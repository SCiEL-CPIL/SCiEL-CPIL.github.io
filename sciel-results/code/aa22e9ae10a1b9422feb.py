import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # block_tsubs[i] is a list of t-subset indices covered by block i
    block_tsubs = [[] for _ in range(n_blocks)]
    # tsub_blocks[j] is a list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            block_tsubs[b_idx].append(idx)
            tsub_blocks[idx].append(b_idx)

    # Convert to numpy arrays for fast access where possible, but lists are often faster for sparse iteration
    # We will use lists for the main loop to avoid numpy overhead on small lists
    # But we can use numpy for the initial greedy selection if we build an incidence matrix.
    # Given n_blocks ~ 50388 and n_tsubsets ~ 3876, an incidence matrix of int8 is ~190MB.
    # Let's stick to lists for memory safety and speed in Python loops, but optimize the inner loops.
    
    # Actually, the current program's weakness is the greedy repair loop being slow in Python.
    # Inspiration 1 uses numpy for the greedy step. Let's try to use numpy for the greedy selection.
    # However, building the full incidence matrix might be slow/memory heavy.
    # Let's use a hybrid: lists for local search, but a vectorized greedy if possible.
    # For v=19, k=7, t=4:
    # n_blocks = C(19,7) = 50388
    # n_tsubsets = C(19,4) = 3876
    # Incidence matrix: 50388 x 3876.
    # Each block covers C(7,4) = 35 t-subsets.
    # Sparse representation is better.

    # Let's refine the greedy step to be faster.
    # The current program uses a heap with lazy deletion, which is O(N log N) per step.
    # Inspiration 1 uses np.argmax on a sum, which is O(N * C(k,t)) per step.
    # For 50k blocks, 35 ops each -> 1.7M ops per step. If we need 150 steps, that's 250M ops.
    # Numpy can do this in ~0.1s. Python heap might be slower due to overhead.

    # Let's implement a numpy-based greedy for the initial construction and repairs.
    # We need an incidence matrix M where M[b, s] = 1 if block b covers t-subset s.
    # But we don't need to store it explicitly if we can compute gains fast.
    # Gains for all blocks given an uncovered mask U:
    # gain[b] = sum(M[b, s] for s in U)
    
    # Let's build the incidence matrix as a sparse structure or dense numpy array.
    # Dense int8 array: 50388 * 3876 * 1 byte = ~190 MB. This is acceptable.
    # Let's build it.
    
    incidence = np.zeros((n_blocks, n_tsubsets), dtype=np.int8)
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            incidence[b_idx, s_idx] = 1

    def greedy_repair_numpy(uncovered_mask_np, existing_blocks_list):
        """
        uncovered_mask_np: numpy bool array of shape (n_tsubsets,)
        existing_blocks_list: list of block indices
        Returns: list of block indices (existing + new)
        """
        if not np.any(uncovered_mask_np):
            return existing_blocks_list

        new_blocks = []
        current_uncovered = uncovered_mask_np.copy()
        
        while np.any(current_uncovered):
            # Calculate gains: for each block, how many currently uncovered t-subsets does it cover?
            # gain[b] = sum(incidence[b, :] * current_uncovered)
            # Since incidence is 0/1, this is just summing rows where current_uncovered is True.
            # We can do: gains = incidence.dot(current_uncovered.astype(np.int8))
            # But dot with bool might be slow. Let's use int8.
            gains = incidence.dot(current_uncovered.astype(np.int8))
            
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break # No block can cover any more uncovered subsets? Should not happen if valid.
                
            new_blocks.append(best_idx)
            
            # Update uncovered mask
            current_uncovered[incidence[best_idx] == 1] = False
            
            # Time check
            if time.time() > deadline - 5:
                break

        return existing_blocks_list + new_blocks

    def reduce_blocks(block_indices):
        """Remove redundant blocks from the list."""
        if not block_indices:
            return []
        
        # Count coverage for each t-subset
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for idx in block_indices:
            for s_idx in block_tsubs[idx]:
                cover_count[s_idx] += 1
        
        new_blocks = []
        for idx in block_indices:
            subs = block_tsubs[idx]
            # Temporarily remove this block's contribution
            for s_idx in subs:
                cover_count[s_idx] -= 1
            
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 0:
                    is_redundant = False
                    break
            
            if is_redundant:
                pass
            else:
                for s_idx in subs:
                    cover_count[s_idx] += 1
                new_blocks.append(idx)
                
        return new_blocks

    # 2. Initial Construction
    full_uncovered = np.ones(n_tsubsets, dtype=bool)
    initial_blocks = greedy_repair_numpy(full_uncovered, [])
    current_blocks = reduce_blocks(initial_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)

    # 3. Local Search Loop
    # We use a hybrid approach:
    # - Fast removal of redundant blocks
    # - Swap moves: replace one block with another that covers its "essential" t-subsets
    
    # Precompute for each block, the set of t-subsets it covers (as a numpy array or list)
    # We already have block_tsubs as lists.
    
    # For swap moves, we need to know which blocks can replace a given block.
    # A block B can replace block A if B covers all t-subsets that are uniquely covered by A.
    # Let U_A be the set of t-subsets where cover_count == 1 and A is the only block covering them.
    # Then B must cover U_A.
    
    # To speed this up, we can precompute for each t-subset, the list of blocks covering it (tsub_blocks).
    
    no_improvement_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Remove redundant blocks
        # We can do this more efficiently by checking each block
        new_blocks_list = []
        for idx in current_blocks:
            # Check if idx is redundant given new_blocks_list
            # This is O(N * C(k,t) * N) which is slow.
            # Better to use the cover_count method from reduce_blocks
            pass
        
        # Let's just call reduce_blocks periodically
        reduced = reduce_blocks(current_blocks)
        if len(reduced) < len(current_blocks):
            current_blocks = reduced
            improved = True
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
        
        if improved:
            no_improvement_count = 0
            continue
        
        # Pass 2: Swap moves
        # Pick a random block to remove
        if not current_blocks:
            break
            
        remove_idx = random.randint(0, len(current_blocks) - 1)
        block_to_remove = current_blocks[remove_idx]
        
        # Calculate essential t-subsets for block_to_remove
        # Essential t-subsets are those covered by block_to_remove but by no other block in current_blocks
        
        # First, calculate cover counts for current_blocks
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for idx in current_blocks:
            for s_idx in block_tsubs[idx]:
                cover_count[s_idx] += 1
        
        # Find essential t-subsets for block_to_remove
        essential_tsubs = []
        for s_idx in block_tsubs[block_to_remove]:
            if cover_count[s_idx] == 1:
                essential_tsubs.append(s_idx)
        
        if not essential_tsubs:
            # Block is redundant, remove it
            current_blocks.pop(remove_idx)
            improved = True
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
            no_improvement_count = 0
            continue
        
        # Find candidate blocks that cover all essential t-subsets
        # A candidate block B must cover all t-subsets in essential_tsubs
        # And B must not be in current_blocks (or if it is, we are just swapping, but usually we look for outside)
        
        candidates = None
        for s_idx in essential_tsubs:
            blocks_covering = tsub_blocks[s_idx]
            if candidates is None:
                candidates = set(blocks_covering)
            else:
                candidates.intersection_update(blocks_covering)
            
            if not candidates:
                break
        
        if not candidates:
            no_improvement_count += 1
            if no_improvement_count > 50:
                break
            continue
        
        # Filter out blocks already in current_blocks (unless we want to swap with an existing one, 
        # but that doesn't change the set size. We want to replace block_to_remove with a new one.)
        current_set = set(current_blocks)
        candidates = [b for b in candidates if b not in current_set]
        
        if not candidates:
            no_improvement_count += 1
            if no_improvement_count > 50:
                break
            continue
        
        # Try a few random candidates
        random.shuffle(candidates)
        for cand in candidates[:5]:
            # Remove block_to_remove, add cand
            new_blocks = list(current_blocks)
            new_blocks.remove(block_to_remove)
            new_blocks.append(cand)
            
            # Check if this is a valid cover (it should be, by construction of essential tsubs)
            # But we should verify that the new block doesn't break anything? No, it only adds coverage.
            # The only risk is if we removed a block that was essential for some t-subset not in essential_tsubs?
            # No, essential_tsubs are the ones ONLY covered by block_to_remove.
            # So any t-subset covered by block_to_remove and at least one other block is still covered.
            # So the cover remains valid.
            
            # Now try to reduce the new set
            reduced_new = reduce_blocks(new_blocks)
            
            if len(reduced_new) < best_size:
                best_size = len(reduced_new)
                best_blocks = list(reduced_new)
                current_blocks = reduced_new
                improved = True
                no_improvement_count = 0
                break
            elif len(reduced_new) == best_size:
                # Accept with some probability to explore
                if random.random() < 0.3:
                    current_blocks = reduced_new
                    improved = True
                    no_improvement_count = 0
                    break
            else:
                # Reject, but maybe keep for exploration?
                pass
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 100:
                break

    # Final reduction
    final_blocks = reduce_blocks(best_blocks)
    final_block_lists = [list(all_blocks[idx]) for idx in final_blocks]
    
    return final_block_lists
