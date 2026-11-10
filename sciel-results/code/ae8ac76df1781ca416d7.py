import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for evaluator
    
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
    
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            coverers[idx].append(i)
        masks[i] = m

    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    best_indices = None

    # Simulated Annealing polish: escape local optima via temperature-controlled moves
    def simulated_annealing(initial_indices, seed, time_limit=10.0):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S

        counts = [0] * n_t
        for b in S:
            temp = masks[b]
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                temp ^= lsb

        sa_start = time.time()
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        best_S = set(S)
        best_len_sa = len(S)

        while T > T_min and (time.time() - sa_start) < time_limit:
            move_type = rng.random()
            if move_type < 0.25:
                # Try removing a block
                candidates = [b for b in S]
                rng.shuffle(candidates)
                for b in candidates[:20]:
                    # Check if removal is safe (no t-subset would become uncovered)
                    safe = True
                    temp = masks[b]
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        if counts[idx] == 1:
                            safe = False
                            break
                        temp ^= lsb
                    if safe:
                        # Remove it
                        S.discard(b)
                        temp = masks[b]
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            temp ^= lsb
                        if len(S) < best_len_sa:
                            best_len_sa = len(S)
                            best_S = set(S)
                        break
            elif move_type < 0.65:
                # 1-1 swap
                list_S = list(S)
                rng.shuffle(list_S)
                for b_out in list_S[:30]:
                    U = []
                    temp = masks[b_out]
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        if counts[idx] == 1:
                            U.append(idx)
                        temp ^= lsb
                    if not U:
                        # Just remove
                        S.discard(b_out)
                        temp = masks[b_out]
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            temp ^= lsb
                        if len(S) < best_len_sa:
                            best_len_sa = len(S)
                            best_S = set(S)
                        break
                    candidates = set(coverers[U[0]])
                    for t_idx in U[1:]:
                        candidates.intersection_update(coverers[t_idx])
                    candidates -= S
                    if candidates:
                        b_in = rng.choice(list(candidates))
                        S.discard(b_out)
                        S.add(b_in)
                        temp = masks[b_out]
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            temp ^= lsb
                        temp = masks[b_in]
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            temp ^= lsb
                        break
            else:
                # Add a random block (may increase size, but could enable better swaps later)
                all_candidates = list(range(n_k))
                rng.shuffle(all_candidates)
                for b in all_candidates[:50]:
                    if b not in S:
                        S.add(b)
                        temp = masks[b]
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            temp ^= lsb
                        break

            T *= cooling

        # Final cleanup: remove redundant blocks
        improved = True
        while improved:
            improved = False
            to_remove = []
            for b in S:
                m = masks[b]
                redundant = True
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    temp ^= lsb
                if redundant:
                    to_remove.append(b)
            for b in to_remove:
                S.discard(b)
                temp = masks[b]
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    temp ^= lsb
                improved = True

        if len(S) < best_len_sa:
            best_len_sa = len(S)
            best_S = set(S)

        return best_S

    # Helper: Local Search (Redundancy Removal + 1-1 Swap + 2-1 Swap)
    def local_search(initial_indices, seed, time_limit=5.0):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S
            
        # counts[t_idx] = number of blocks in S covering t_idx
        counts = [0] * n_t
        for b in S:
            m = masks[b]
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                temp ^= lsb
        
        improved = True
        ls_start = time.time()
        
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            
            # Pass 1: Remove Redundant Blocks
            to_remove = []
            for b in S:
                m = masks[b]
                redundant = True
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    temp ^= lsb
                if redundant:
                    to_remove.append(b)
            
            for b in to_remove:
                S.discard(b)
                m = masks[b]
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    temp ^= lsb
                if to_remove:
                    improved = True
            if to_remove:
                continue # Restart loop with new S
            
            # Pass 2: 1-1 Swap
            swapped = False
            list_S = list(S)
            rng.shuffle(list_S)
            
            for b_out in list_S:
                if b_out not in S: continue
                U = []
                m_out = masks[b_out]
                temp = m_out
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        U.append(idx)
                    temp ^= lsb
                
                if not U:
                    S.discard(b_out)
                    m = m_out
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
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
                    
                    temp = m_out
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        temp ^= lsb
                        
                    m_in = masks[b_in]
                    temp = m_in
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        temp ^= lsb
                        
                    improved = True
                    swapped = True
                    break # Restart pass with new S
            
            if swapped:
                continue # Restart main while loop with new S

            # Pass 3: 2-1 Swap (Remove 2, Add 1)
            # Only try if we have time, as it's O(N^2)
            if time.time() - ls_start < time_limit - 0.5:
                list_S = list(S)
                rng.shuffle(list_S)
                # Limit pairs checked per iteration to avoid timeout
                max_pairs = 2000 
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
                        
                        m1 = masks[b1]
                        m2 = masks[b2]
                        
                        # Quick check: if m1 & m2 == 0, they share no t-subsets, so removing both is safe if each is redundant
                        # But we need to check if together they're redundant
                        m_union = m1 | m2
                        U = []
                        temp = m_union
                        while temp:
                            lsb = temp & -temp
                            idx = lsb.bit_length() - 1
                            temp ^= lsb
                            
                            c = counts[idx]
                            if m1 & lsb: c -= 1
                            if m2 & lsb: c -= 1
                            
                            if c == 0:
                                U.append(idx)
                        
                        if not U:
                            # We can remove both b1 and b2 without adding anything
                            S.discard(b1)
                            S.discard(b2)
                            temp = m1
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            temp = m2
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
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
                            
                            temp = m1
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            temp = m2
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                temp ^= lsb
                            
                            m_in = masks[b_in]
                            temp = m_in
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                counts[idx] += 1
                                temp ^= lsb
                            
                            improved = True
                            swapped = True
                            break # Restart pass
                            
                if swapped:
                    continue # Restart main while loop with new S

        return S

    # Run multiple randomized greedy attempts with diversification
    attempts = 30
    for attempt in range(attempts):
        if time.time() > deadline - 15:
            break
            
        rng = random.Random(attempt * 12345 + 67890)
        
        # Diversified greedy: use randomized weights for tie-breaking
        # Weight each k-subset randomly to break ties and explore different solutions
        weights = [rng.random() for _ in range(n_k)]
        
        current_uncovered = (1 << n_t) - 1
        selected = set()
        
        while current_uncovered:
            best_score = -1
            best_idx = -1
            best_mask = 0
            
            for i in range(n_k):
                if i in selected:
                    continue
                gain_mask = current_uncovered & masks[i]
                if not gain_mask:
                    continue
                gain = gain_mask.bit_count()
                # Combine gain with small random weight for diversification
                score = gain + weights[i] * 0.01
                
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_mask = gain_mask
            
            if best_idx == -1:
                break
                
            selected.add(best_idx)
            current_uncovered &= ~best_mask
            
            if time.time() > deadline - 20:
                break
        
        if current_uncovered == 0:
            indices = list(selected)
            
            # Apply Local Search
            ls_indices = local_search(indices, attempt, time_limit=4.0)
            ls_len = len(ls_indices)
            
            if ls_len < best_len:
                best_len = ls_len
                best_indices = ls_indices
                best_solution = get_blocks(ls_indices)
                
            if time.time() > deadline - 20:
                break

    # Final Polish: Run local search then simulated annealing on the best solution found
    if best_indices is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            # First: extended local search
            polished = local_search(best_indices, 99999, time_limit=min(remaining_time * 0.5, 15.0))
            if len(polished) < best_len:
                best_len = len(polished)
                best_indices = polished
                best_solution = get_blocks(polished)
            
            # Then: simulated annealing to escape local optima
            remaining_time = deadline - 5 - time.time()
            if remaining_time > 5:
                sa_result = simulated_annealing(best_indices, 424242, time_limit=min(remaining_time - 2, 25.0))
                if len(sa_result) < best_len:
                    best_len = len(sa_result)
                    best_indices = sa_result
                    best_solution = get_blocks(sa_result)

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
