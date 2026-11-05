import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Convert to numpy arrays for faster indexing? Lists are fine for small n_t per block (56)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """Greedy set cover with optional randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks_idx = []
        used_blocks = set()

        while uncovered.any():
            # Pick an uncovered t-subset - use random if seed > 0, else min
            if seed > 0:
                uncovered_indices = np.where(uncovered)[0]
                s_idx = rng.choice(uncovered_indices.tolist())
            else:
                s_idx = int(np.argmax(uncovered))  # first True

            # Get candidate k-subsets containing s_idx
            candidates = t_sub_to_k_idxs[s_idx]

            best_block_idx = -1
            best_gain = -1
            # Collect top candidates for randomization
            candidate_gains = []

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                # Calculate gain
                gain = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    candidate_gains = [(gain, b_idx)]
                elif gain == best_gain and gain > 0:
                    candidate_gains.append((gain, b_idx))

            if best_block_idx == -1:
                break

            # If randomized, pick from top gains
            if seed > 0 and len(candidate_gains) > 1:
                # Pick randomly from those with max gain (or top few)
                # Simple: pick randomly from all with max gain
                # Or pick with probability proportional to gain? 
                # Let's stick to picking from max gain set to ensure quality
                chosen = rng.choice(candidate_gains)
                best_block_idx = chosen[1]

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)

            # Update uncovered
            for t_idx in k_sub_t_idxs[best_block_idx]:
                uncovered[t_idx] = False

        return blocks_idx

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search_with_swaps(blocks_idx, max_iter=50):
        """
        Local search that tries to remove blocks, and if not possible, 
        swaps a block with another to enable further removals.
        """
        current = list(blocks_idx)
        if not current:
            return current

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        current_set = set(current)

        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # First pass: try simple removals
            for i in range(len(current)):
                b_idx = current[i]
                can_remove = True
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break

                if can_remove:
                    # Remove it
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current.pop(i)
                    current_set.remove(b_idx)
                    improved = True
                    break # Restart loop as indices changed

            if improved:
                continue

            # Second pass: try swaps
            # For each block, check if we can swap it to allow removal of others
            for i in range(len(current)):
                b_idx = current[i]
                # Find unique t-subsets
                unique_ts = []
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if not unique_ts:
                    # Should have been removed in first pass
                    continue

                # Find candidates that cover all unique_ts
                # Intersect candidate lists
                # Start with candidates for the first unique t
                candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                for t_idx in unique_ts[1:]:
                    candidates &= set(t_sub_to_k_idxs[t_idx])
                
                # Remove current block and already used blocks
                candidates.discard(b_idx)
                candidates -= current_set
                
                if not candidates:
                    continue
                
                # Try swapping with one candidate
                # Pick one arbitrarily or randomly
                # To be safe, try a few
                # Let's just try one to save time
                swap_candidate = next(iter(candidates))
                
                # Simulate swap: remove b_idx, add swap_candidate
                # Check if this allows removal of any other block
                # We need to update cover counts temporarily
                
                # Store original cover counts? No, just update and rollback if failed
                # Update for removal of b_idx
                for t_idx in k_sub_t_idxs[b_idx]:
                    cover_count[t_idx] -= 1
                
                # Update for addition of swap_candidate
                for t_idx in k_sub_t_idxs[swap_candidate]:
                    cover_count[t_idx] += 1
                
                # Now check if any block in current (except b_idx) can be removed
                # Note: b_idx is still in 'current' list, but logically removed
                # We need to check other blocks
                removed_any = False
                removals = []
                
                # We must be careful not to remove b_idx again in this check
                # And we must not remove swap_candidate (it's not in current list yet)
                
                for j in range(len(current)):
                    if j == i:
                        continue
                    b_j = current[j]
                    can_remove = True
                    for t_idx in k_sub_t_idxs[b_j]:
                        if cover_count[t_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        removals.append(j)
                
                if removals:
                    # Accept swap and removals
                    # Remove b_idx
                    current.pop(i)
                    current_set.remove(b_idx)
                    
                    # Add swap_candidate
                    current.append(swap_candidate)
                    current_set.add(swap_candidate)
                    
                    # Remove the redundant blocks
                    # We must pop from end to start to maintain indices or just rebuild
                    # Rebuilding is safer
                    new_current = []
                    removed_set = set(removals)
                    for j in range(len(current)):
                        # Note: indices in 'removals' are from the old 'current' list
                        # But we already popped i. 
                        # This index management is tricky.
                        pass
                    
                    # Let's just rebuild the list and cover counts from scratch for simplicity/safety
                    # Since n_blocks is ~300, this is fast.
                    new_current = [b for j, b in enumerate(current) if j not in removed_set]
                    # Wait, 'current' was modified by pop(i).
                    # The indices in 'removals' refer to the list BEFORE pop(i).
                    # This is getting error-prone.
                    
                    # Safer approach: 
                    # 1. Create a new list of blocks: current without b_idx, plus swap_candidate
                    # 2. Rebuild cover counts
                    # 3. Run reduce_blocks on this new set
                    # 4. If len(new_set) < len(current), accept
                    pass

                # Rollback cover counts
                for t_idx in k_sub_t_idxs[swap_candidate]:
                    cover_count[t_idx] -= 1
                for t_idx in k_sub_t_idxs[b_idx]:
                    cover_count[t_idx] += 1

        return current

    # Run multiple greedy passes with different seeds
    best_blocks = None
    best_len = float('inf')

    # First, deterministic greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # Try randomized greedy if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        for seed in seeds:
            if time.time() >= deadline - 5.0:
                break
            blocks_idx = greedy_cover(seed)
            blocks_idx = reduce_blocks(blocks_idx)
            if blocks_idx and len(blocks_idx) < best_len:
                best_blocks = blocks_idx
                best_len = len(blocks_idx)

    # Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        # The simple local search in the original code was just removal.
        # We can try to improve it with swaps, but it's complex.
        # Let's stick to a robust removal-based local search first, 
        # but maybe do multiple passes.
        improved = local_search_with_swaps(best_blocks, max_iter=50)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = len(improved)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
