import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for evaluator (120s limit)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Number of 64-bit words needed for n_t bits
    n_words = (n_t + 63) // 64
    
    # Precompute masks as numpy array of uint64
    # masks_np[i] is an array of n_words uint64s
    masks_np = np.zeros((n_k, n_words), dtype=np.uint64)
    
    # Precompute masks as Python integers for local search
    masks_int = [0] * n_k
    
    # Precompute coverers: coverers[j] is a list of k-subset indices covering t-subset j
    coverers = [[] for _ in range(n_t)]
    
    for i, k_sub in enumerate(all_k_subsets):
        m_int = 0
        # We need to set bits in masks_np[i]
        # and also in m_int
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks_np[i, word_idx] |= (1 << bit_idx)
            m_int |= (1 << idx)
            coverers[idx].append(i)
        masks_int[i] = m_int
        
    # Mask for the last word to ignore unused bits
    last_word_bits = n_t % 64
    if last_word_bits == 0:
        last_word_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    else:
        last_word_mask = np.uint64((1 << last_word_bits) - 1)

    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    best_indices = None

    # Helper: Local Search (Redundancy Removal + 1-1 Swap + 2-1 Swap)
    def local_search(initial_indices, seed, time_limit=5.0):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S
            
        # counts[t_idx] = number of blocks in S covering t_idx
        counts = [0] * n_t
        for b in S:
            m = masks_int[b]
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
                m = masks_int[b]
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
                m = masks_int[b]
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
                m_out = masks_int[b_out]
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
                        
                    m_in = masks_int[b_in]
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
                max_pairs = 1000 # Reduced to avoid timeout
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
                        
                        m1 = masks_int[b1]
                        m2 = masks_int[b2]
                        
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
                            
                        candidates = set(coverers[U[0]])
                        for t_idx in U[1:]:
                            candidates.intersection_update(coverers[t_idx])
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
                            
                            m_in = masks_int[b_in]
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

    # Numpy-based Greedy
    def greedy_numpy(seed):
        rng = random.Random(seed)
        # Initialize uncovered as all 1s, masked for last word
        uncovered_np = np.ones(n_words, dtype=np.uint64)
        uncovered_np[-1] = last_word_mask
        
        selected = []
        
        while True:
            # Check if uncovered is zero
            if not np.any(uncovered_np):
                break
            
            # Calculate gains for all blocks
            # bitwise_and is vectorized
            gains = np.bitwise_and(masks_np, uncovered_np).sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if covered, but safety
                break
            
            # Get all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one
            best_idx = int(rng.choice(candidates))
            
            selected.append(best_idx)
            
            # Update uncovered
            uncovered_np = np.bitwise_and(uncovered_np, ~masks_np[best_idx])
            # Re-mask last word to ensure no overflow bits are set
            uncovered_np[-1] &= last_word_mask
            
            if time.time() > deadline - 20:
                break
        
        return selected

    # Run multiple greedy attempts
    attempts = 40
    for attempt in range(attempts):
        if time.time() > deadline - 30:
            break
            
        indices = greedy_numpy(attempt * 12345 + 67890)
        
        if not indices:
            continue
            
        # Quick local search
        ls_indices = local_search(indices, attempt, time_limit=2.0)
        ls_len = len(ls_indices)
        
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices
            best_solution = get_blocks(ls_indices)

    # Final Polish
    if best_indices is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        
        # Extended local search
        if remaining_time > 5:
            polished = local_search(best_indices, 99999, time_limit=min(remaining_time * 0.6, 20.0))
            if len(polished) < best_len:
                best_len = len(polished)
                best_indices = polished
                best_solution = get_blocks(polished)
        
        # Simulated Annealing
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5:
            def simulated_annealing(initial_indices, seed, time_limit=10.0):
                rng = random.Random(seed)
                S = set(initial_indices)
                if not S:
                    return S

                counts = [0] * n_t
                for b in S:
                    temp = masks_int[b]
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
                    if move_type < 0.3:
                        # Try removing a block
                        candidates = list(S)
                        rng.shuffle(candidates)
                        for b in candidates[:20]:
                            safe = True
                            temp = masks_int[b]
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                if counts[idx] == 1:
                                    safe = False
                                    break
                                temp ^= lsb
                            if safe:
                                S.discard(b)
                                temp = masks_int[b]
                                while temp:
                                    lsb = temp & -temp
                                    idx = lsb.bit_length() - 1
                                    counts[idx] -= 1
                                    temp ^= lsb
                                if len(S) < best_len_sa:
                                    best_len_sa = len(S)
                                    best_S = set(S)
                                break
                    elif move_type < 0.7:
                        # 1-1 swap
                        list_S = list(S)
                        rng.shuffle(list_S)
                        for b_out in list_S[:30]:
                            U = []
                            temp = masks_int[b_out]
                            while temp:
                                lsb = temp & -temp
                                idx = lsb.bit_length() - 1
                                if counts[idx] == 1:
                                    U.append(idx)
                                temp ^= lsb
                            if not U:
                                S.discard(b_out)
                                temp = masks_int[b_out]
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
                                temp = masks_int[b_out]
                                while temp:
                                    lsb = temp & -temp
                                    idx = lsb.bit_length() - 1
                                    counts[idx] -= 1
                                    temp ^= lsb
                                temp = masks_int[b_in]
                                while temp:
                                    lsb = temp & -temp
                                    idx = lsb.bit_length() - 1
                                    counts[idx] += 1
                                    temp ^= lsb
                                break
                    else:
                        # Add random block
                        all_candidates = list(range(n_k))
                        rng.shuffle(all_candidates)
                        for b in all_candidates[:50]:
                            if b not in S:
                                S.add(b)
                                temp = masks_int[b]
                                while temp:
                                    lsb = temp & -temp
                                    idx = lsb.bit_length() - 1
                                    counts[idx] += 1
                                    temp ^= lsb
                                break

                    T *= cooling

                # Final cleanup
                improved = True
                while improved:
                    improved = False
                    to_remove = []
                    for b in S:
                        m = masks_int[b]
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
                        temp = masks_int[b]
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

            sa_result = simulated_annealing(best_indices, 424242, time_limit=min(remaining_time - 2, 20.0))
            if len(sa_result) < best_len:
                best_len = len(sa_result)
                best_indices = sa_result
                best_solution = get_blocks(sa_result)

    if best_solution is None:
        # Fallback: Simple greedy
        current_uncovered = (1 << n_t) - 1
        selected = []
        while current_uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0
            for i in range(n_k):
                if i in selected: continue
                gain_mask = current_uncovered & masks_int[i]
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
