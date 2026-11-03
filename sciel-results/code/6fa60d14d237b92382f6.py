import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 132.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Step 2: Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Step 3: For each block, compute which t-subsets it covers
    # Use Python integers as bitsets for fast bitwise operations
    block_bits = [0] * num_k
    block_t_subsets = [set() for _ in range(num_k)]
    
    for i, block in enumerate(k_blocks):
        bits = 0
        covered = set()
        for combo in itertools.combinations(block, t):
            idx = t_index[combo]
            bits |= (1 << idx)
            covered.add(idx)
        block_bits[i] = bits
        block_t_subsets[i] = covered

    if elapsed() > deadline - 40:
        return [list(b) for b in k_blocks]

    # Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    t_cover_count = np.array([len(lst) for lst in t_to_blocks], dtype=np.int32)

    # Step 4: Weighted greedy set cover
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    t_weights = 1.0 / safe_counts

    # Precompute block coverage as numpy arrays for fast set operations in greedy
    block_cover_arrays = []
    for covered in block_t_subsets:
        if covered:
            block_cover_arrays.append(np.fromiter(covered, dtype=np.int32, count=len(covered)))
        else:
            block_cover_arrays.append(np.array([], dtype=np.int32))

    gain = np.zeros(num_k, dtype=np.float64)
    for i, arr in enumerate(block_cover_arrays):
        if arr.size > 0:
            gain[i] = np.sum(t_weights[arr])

    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)

    while uncovered_count > 0 and elapsed() < deadline - 30:
        masked_gain = gain.copy()
        masked_gain[selected_mask] = -1.0
        best_idx = int(np.argmax(masked_gain))
        best_gain = gain[best_idx]

        if best_gain <= 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True

        # Vectorized update
        new_covered = block_cover_arrays[best_idx]
        if new_covered.size > 0:
            newly_uncovered = new_covered[uncovered[new_covered]]
            if newly_uncovered.size > 0:
                uncovered[newly_uncovered] = False
                uncovered_count -= len(newly_uncovered)
                for idx in newly_uncovered:
                    w = t_weights[idx]
                    covering_blocks = t_to_blocks[idx]
                    mask = np.array([not selected_mask[b] for b in covering_blocks])
                    if mask.any():
                        gain[covering_blocks] -= w

    result = [list(k_blocks[i]) for i in selected]
    
    # Local Search with Bitsets
    if elapsed() < deadline - 10 and result:
        # Initialize cover counts and bitsets
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        
        # Build bitsets for S0, S1, S2
        # S0: count == 0
        # S1: count == 1
        # S2: count == 2
        S0_bits = 0
        S1_bits = 0
        S2_bits = 0
        
        for idx in range(num_t):
            c = cover_count[idx]
            if c == 0:
                S0_bits |= (1 << idx)
            elif c == 1:
                S1_bits |= (1 << idx)
            elif c == 2:
                S2_bits |= (1 << idx)
        
        improved = True
        max_rounds = 20
        
        for _ in range(max_rounds):
            if elapsed() > deadline - 10:
                break
            if not improved:
                break
            improved = False
            
            # 1. Try to drop blocks
            for si in range(len(selected) - 1, -1, -1):
                if elapsed() > deadline - 10:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                
                # Check if any critical
                can_drop = True
                for idx in drop_set:
                    if cover_count[idx] == 1:
                        can_drop = False
                        break
                
                if can_drop:
                    for idx in drop_set:
                        # Update cover_count
                        cover_count[idx] -= 1
                        # Update bitsets
                        if cover_count[idx] == 0:
                            S1_bits &= ~(1 << idx)
                            S0_bits |= (1 << idx)
                        elif cover_count[idx] == 1:
                            S2_bits &= ~(1 << idx)
                            S1_bits |= (1 << idx)
                        elif cover_count[idx] == 2:
                            S2_bits |= (1 << idx) # Wait, if it was 2 and became 1, it moves to S1.
                            # Correction:
                            # If old count was 2, it was in S2. New count is 1. Move to S1.
                            # If old count was 1, it was in S1. New count is 0. Move to S0.
                            # If old count was 3, it was not in S0/S1/S2. New count is 2. Move to S2.
                            pass # Logic below handles this better

                    # Let's redo the bitset update logic carefully
                    # It's safer to rebuild S0, S1, S2 from scratch after a drop? 
                    # No, that's O(num_t). num_t is 11628. That's fast.
                    # Rebuilding is safer and likely fast enough.
                    
                    selected.pop(si)
                    result = [list(k_blocks[i]) for i in selected]
                    improved = True
                    
                    # Rebuild bitsets
                    S0_bits = 0
                    S1_bits = 0
                    S2_bits = 0
                    for idx in range(num_t):
                        c = cover_count[idx]
                        if c == 0:
                            S0_bits |= (1 << idx)
                        elif c == 1:
                            S1_bits |= (1 << idx)
                        elif c == 2:
                            S2_bits |= (1 << idx)
                    break # Restart the drop/swap loop

            if improved:
                continue

            # 2. Try swaps
            # We want to find a swap A -> B that minimizes the new number of criticals (S1)
            # Or simply improves the "droppability".
            # Heuristic: Minimize popcount(new_S1).
            
            best_swap = None
            best_new_crit_count = float('inf')
            
            for si in range(len(selected)):
                if elapsed() > deadline - 10:
                    break
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                drop_bits = block_bits[drop_block]
                
                # Find criticals of drop_block
                critical_bits = 0
                for idx in drop_set:
                    if cover_count[idx] == 1:
                        critical_bits |= (1 << idx)
                
                if critical_bits == 0:
                    # Should have been caught in drop phase
                    continue

                # Find best replacement B
                # B must cover critical_bits
                # Score: popcount(new_S1)
                
                # We can iterate through all blocks not in selected
                # To speed up, we can pre-filter blocks that cover critical_bits?
                # That's hard. Just iterate.
                
                for b in range(num_k):
                    if b in selected_set:
                        continue
                    b_bits = block_bits[b]
                    
                    # Check if B covers criticals
                    if (b_bits & critical_bits) != critical_bits:
                        continue
                    
                    # Calculate new_S1
                    # new_S1 = (S2 & A & ~B) | (S1 & A & B) | (S1 & ~A & ~B)
                    # Note: S1 is current S1. A is drop_block. B is candidate.
                    
                    term1 = S2_bits & drop_bits & ~b_bits
                    term2 = S1_bits & drop_bits & b_bits
                    term3 = S1_bits & ~drop_bits & ~b_bits
                    
                    new_S1 = term1 | term2 | term3
                    new_crit_count = new_S1.bit_count()
                    
                    if new_crit_count < best_new_crit_count:
                        best_new_crit_count = new_crit_count
                        best_swap = (si, b, new_S1)
            
            if best_swap is not None:
                si, b, new_S1_bits = best_swap
                drop_block = selected[si]
                drop_set = block_t_subsets[drop_block]
                add_set = block_t_subsets[b]
                
                # Perform swap
                for idx in drop_set:
                    cover_count[idx] -= 1
                for idx in add_set:
                    cover_count[idx] += 1
                    
                selected[si] = b
                selected_set.discard(drop_block)
                selected_set.add(b)
                result = [list(k_blocks[i]) for i in selected]
                improved = True
                
                # Rebuild bitsets
                S0_bits = 0
                S1_bits = 0
                S2_bits = 0
                for idx in range(num_t):
                    c = cover_count[idx]
                    if c == 0:
                        S0_bits |= (1 << idx)
                    elif c == 1:
                        S1_bits |= (1 << idx)
                    elif c == 2:
                        S2_bits |= (1 << idx)

    # Final validity check
    if result:
        final_cover = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                final_cover[idx] += 1
        if np.any(final_cover == 0):
            # Fallback: add blocks to cover uncovered
            selected_set = set(selected)
            for i in range(num_k):
                if elapsed() > deadline - 5:
                    break
                if i in selected_set:
                    continue
                has_uncovered = False
                for idx in block_t_subsets[i]:
                    if final_cover[idx] == 0:
                        has_uncovered = True
                        break
                if has_uncovered:
                    selected.append(i)
                    selected_set.add(i)
                    for idx2 in block_t_subsets[i]:
                        final_cover[idx2] += 1
            result = [list(k_blocks[i]) for i in selected]

    return result
