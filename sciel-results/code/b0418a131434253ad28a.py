import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for evaluator
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute masks for all k-subsets
    masks = [0] * n_k
    # Also precompute coverers: coverers[j] is a list of k-subset indices covering t-subset j
    coverers = [[] for _ in range(n_t)]
    
    # Precompute block_t_indices for faster local search updates
    block_t_indices = [[] for _ in range(n_k)]
    set_block_t_indices = [set() for _ in range(n_k)]
    
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            coverers[idx].append(i)
            block_t_indices[i].append(idx)
            set_block_t_indices[i].add(idx)
        masks[i] = m
        
    # Precompute rarity weights for greedy strategy
    rarity_weight = [0.0] * n_k
    for i in range(n_k):
        w = 0.0
        for idx in block_t_indices[i]:
            w += 1.0 / len(coverers[idx])
        rarity_weight[i] = w

    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    best_indices = None

    # Helper: Local Search (Redundancy Removal + 1-1 Swap + 2-1 Swap + 1-2 Swap)
    def local_search(initial_indices, seed, time_limit=5.0):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S
            
        # counts[t_idx] = number of blocks in S covering t_idx
        counts = [0] * n_t
        for b in S:
            for idx in block_t_indices[b]:
                counts[idx] += 1
        
        improved = True
        ls_start = time.time()
        ls_iter = 0
        
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            ls_iter += 1
            
            # Pass 1: Remove Redundant Blocks
            to_remove = []
            for b in S:
                redundant = True
                for idx in block_t_indices[b]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
            
            for b in to_remove:
                S.discard(b)
                for idx in block_t_indices[b]:
                    counts[idx] -= 1
            if to_remove:
                improved = True
                continue # Restart loop with new S
            
            # Pass 2: 1-1 Swap
            swapped = False
            list_S = list(S)
            rng.shuffle(list_S)
            
            for b_out in list_S:
                if b_out not in S: continue
                U = []
                for idx in block_t_indices[b_out]:
                    if counts[idx] == 1:
                        U.append(idx)
                
                if not U:
                    S.discard(b_out)
                    for idx in block_t_indices[b_out]:
                        counts[idx] -= 1
                    improved = True
                    swapped = True
                    break # Restart pass with new S
                
                candidates = set(coverers[U[0]])
                for t_idx in U[1:]:
                    candidates.intersection_update(coverers[t_idx])
                candidates -= S
                
                if candidates:
                    b_in = rng.choice(list(candidates))
                    S.discard(b_out)
                    S.add(b_in)
                    
                    for idx in block_t_indices[b_out]:
                        counts[idx] -= 1
                    for idx in block_t_indices[b_in]:
                        counts[idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart pass with new S
            
            if swapped:
                continue # Restart main while loop with new S

            # Pass 3: 2-1 Swap (Remove 2, Add 1)
            if time.time() - ls_start < time_limit - 0.3:
                list_S = list(S)
                rng.shuffle(list_S)
                # Increase max_pairs for more thorough search
                max_pairs = 5000 
                pairs_checked = 0
                
                for i_idx in range(len(list_S)):
                    if pairs_checked > max_pairs: break
                    b1 = list_S[i_idx]
                    if b1 not in S: continue
                    
                    for j_idx in range(i_idx + 1, len(list_S)):
                        if pairs_checked > max_pairs: break
                        b2 = list_S[j_idx]
                        if b2 not in S: continue
                        pairs_checked += 1
                        
                        set_b1 = set_block_t_indices[b1]
                        set_b2 = set_block_t_indices[b2]
                        
                        U = []
                        for idx in set_b1:
                            if counts[idx] == 1:
                                U.append(idx)
                            elif counts[idx] == 2 and idx in set_b2:
                                U.append(idx)
                        
                        for idx in set_b2:
                            if idx in set_b1:
                                continue
                            if counts[idx] == 1:
                                U.append(idx)
                        
                        if not U:
                            # We can remove both b1 and b2 without adding anything
                            S.discard(b1)
                            S.discard(b2)
                            for idx in block_t_indices[b1]:
                                counts[idx] -= 1
                            for idx in block_t_indices[b2]:
                                counts[idx] -= 1
                            improved = True
                            swapped = True
                            break # Restart pass
                            
                        # Find candidate b_in
                        candidates = set(coverers[U[0]])
                        for t_idx in U[1:]:
                            candidates.intersection_update(coverers[t_idx])
                        
                        # b_in must not be in S (we're removing b1 and b2, but b_in should be new)
                        candidates -= S
                        
                        if candidates:
                            b_in = rng.choice(list(candidates))
                            S.discard(b1)
                            S.discard(b2)
                            S.add(b_in)
                            
                            for idx in block_t_indices[b1]:
                                counts[idx] -= 1
                            for idx in block_t_indices[b2]:
                                counts[idx] -= 1
                            for idx in block_t_indices[b_in]:
                                counts[idx] += 1
                            
                            improved = True
                            swapped = True
                            break # Restart pass
                            
                if swapped:
                    continue # Restart main while loop with new S

            # Pass 4: 1-2 Swap (Remove 1, Add 2) - helps escape local minima
            if time.time() - ls_start < time_limit - 0.3 and ls_iter % 3 == 0:
                list_S = list(S)
                rng.shuffle(list_S)
                max_attempts = 1000
                attempts_checked = 0
                
                for b_out in list_S:
                    if attempts_checked > max_attempts: break
                    if b_out not in S: continue
                    attempts_checked += 1
                    
                    # Find t-subsets that would become uncovered
                    U = []
                    for idx in block_t_indices[b_out]:
                        if counts[idx] == 1:
                            U.append(idx)
                    
                    # We need to add 2 blocks that together cover all of U
                    if not U:
                        # Just remove
                        S.discard(b_out)
                        for idx in block_t_indices[b_out]:
                            counts[idx] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    # Try pairs of blocks from coverers of first uncovered
                    cands = list(coverers[U[0]])
                    rng.shuffle(cands)
                    for b_in1 in cands[:50]:  # Limit candidates
                        if b_in1 in S:
                            continue
                        set_in1 = set_block_t_indices[b_in1]
                        # Check what's still uncovered after adding b_in1
                        remaining = []
                        for t_idx in U:
                            if t_idx not in set_in1:
                                remaining.append(t_idx)
                        if not remaining:
                            # b_in1 alone covers all U, so we can do 1-1 swap
                            S.discard(b_out)
                            S.add(b_in1)
                            for idx in block_t_indices[b_out]:
                                counts[idx] -= 1
                            for idx in block_t_indices[b_in1]:
                                counts[idx] += 1
                            improved = True
                            swapped = True
                            break
                        else:
                            # Need one more block to cover remaining
                            cands2 = list(coverers[remaining[0]])
                            for b_in2 in cands2[:30]:
                                if b_in2 in S or b_in2 == b_in1:
                                    continue
                                set_in2 = set_block_t_indices[b_in2]
                                # Check if b_in1 | b_in2 covers all U
                                all_covered = True
                                for t_idx in U:
                                    if t_idx not in set_in1 and t_idx not in set_in2:
                                        all_covered = False
                                        break
                                if all_covered:
                                    S.discard(b_out)
                                    S.add(b_in1)
                                    S.add(b_in2)
                                    for idx in block_t_indices[b_out]:
                                        counts[idx] -= 1
                                    for idx in block_t_indices[b_in1]:
                                        counts[idx] += 1
                                    for idx in block_t_indices[b_in2]:
                                        counts[idx] += 1
                                    improved = True
                                    swapped = True
                                    break
                            if swapped:
                                break
                    if swapped:
                        break
                if swapped:
                    continue

        return S

    # Run multiple randomized greedy attempts with different strategies
    attempts = 30
    for attempt in range(attempts):
        if time.time() > deadline - 12:
            break
            
        rng = random.Random(attempt * 12345 + 67890)
        
        current_uncovered = (1 << n_t) - 1
        selected = set()
        
        # Use different greedy strategies for diversity
        strategy = attempt % 4
        
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            best_score = -1
            
            for i in range(n_k):
                if i in selected:
                    continue
                gain_mask = current_uncovered & masks[i]
                if not gain_mask:
                    continue
                gain = gain_mask.bit_count()
                
                if strategy == 0:
                    # Pure greedy
                    score = gain
                elif strategy == 1:
                    # Greedy with random tie-breaking
                    score = gain + rng.random() * 0.5
                elif strategy == 2:
                    # Weighted greedy: prefer blocks that cover more uncovered
                    score = gain + rng.random() * 0.3
                else:
                    # Strategy 3: Rarity-weighted greedy
                    score = gain + 0.5 * rarity_weight[i]
                
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
            
            if best_idx == -1:
                break
                
            selected.add(best_idx)
            current_uncovered &= ~best_mask
            
            if time.time() > deadline - 15:
                break
        
        if current_uncovered == 0:
            indices = list(selected)
            
            # Apply Local Search with more time for later attempts
            ls_time = 3.0 + min(attempt * 0.2, 7.0)
            ls_indices = local_search(indices, attempt, time_limit=ls_time)
            ls_len = len(ls_indices)
            
            if ls_len < best_len:
                best_len = ls_len
                best_indices = ls_indices
                best_solution = get_blocks(ls_indices)
                
                # Early termination if we hit the lower bound
                if best_len == 69:
                    return best_solution

            if time.time() > deadline - 15:
                break

    # Final Polish: Run a longer local search on the best solution found
    if best_indices is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polished = local_search(best_indices, 99999, time_limit=min(remaining_time, 30.0))
            if len(polished) < best_len:
                best_len = len(polished)
                best_indices = polished
                best_solution = get_blocks(polished)
                
                if best_len == 69:
                    return best_solution

    if best_solution is None:
        # Fallback
        current_uncovered = (1 << n_t) - 1
        selected = []
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            for i in range(n_k):
                if i in selected: continue
                gain_mask = current_uncovered & masks[i]
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
            if best_idx == -1: break
            selected.append(best_idx)
            current_uncovered &= ~best_mask
        best_solution = get_blocks(selected)

    return best_solution
