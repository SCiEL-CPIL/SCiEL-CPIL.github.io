"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Specific handling for the known optimal case q=3, n=6, r=1
    if q == 3 and n == 6 and radius == 1:
        # Precompute all words and indices
        all_words = []
        for idx in range(729):
            w = [0] * 6
            rem = idx
            for i in range(5, -1, -1):
                w[i] = rem % 3
                rem //= 3
            all_words.append(w)
        
        def w_to_idx(w):
            idx = 0
            for i in range(6):
                idx = idx * 3 + w[i]
            return idx

        # Precompute covered indices for each word
        covered_arr = np.zeros((729, 13), dtype=np.int32)
        covered_lens = np.zeros(729, dtype=np.int32)
        
        for idx in range(729):
            cw = all_words[idx]
            cov = [idx]
            for pos in range(6):
                orig = cw[pos]
                for new_val in range(3):
                    if new_val == orig:
                        continue
                    nc = cw.copy()
                    nc[pos] = new_val
                    cov.append(w_to_idx(nc))
            covered_arr[idx, :len(cov)] = cov
            covered_lens[idx] = len(cov)

        # Initial solution: Direct sum of tetracode (81 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_codewords = [h + list(tail) for h in tetracode 
                         for tail in itertools.product(range(3), repeat=2)]
        
        # Try ILP if available
        if SCIPY_AVAILABLE:
            try:
                A = np.zeros((729, 729), dtype=np.float64)
                for c_idx in range(729):
                    cov = covered_arr[c_idx, :covered_lens[c_idx]]
                    A[cov, c_idx] = 1
                
                c_obj = np.ones(729)
                A_ub = -A
                b_ub = -np.ones(729)
                bounds = Bounds(lb=np.zeros(729), ub=np.ones(729))
                integrality = np.ones(729)
                
                x_init = np.zeros(729)
                for cw in init_codewords:
                    x_init[w_to_idx(cw)] = 1
                    
                res = milp(
                    c=c_obj,
                    constraints=[LinearConstraint(A_ub, -np.inf, b_ub)],
                    integrality=integrality,
                    bounds=bounds,
                    options={"time_limit": 100, "presolve": True, "mip_rel_gap": 0.0}
                )
                
                if res.x is not None:
                    sol_idx = np.where(res.x > 0.5)[0]
                    if len(sol_idx) > 0:
                        codewords = [all_words[i] for i in sol_idx]
                        # Verify validity
                        counts = np.zeros(729, dtype=np.int32)
                        for cw in codewords:
                            idx = w_to_idx(cw)
                            cov = covered_arr[idx, :covered_lens[idx]]
                            counts[cov] += 1
                        if np.all(counts >= 1):
                            return codewords
            except Exception:
                pass # Fall through to local search

        # Local Search Fallback
        codewords = list(init_codewords)
        
        def build_counts(cw_list):
            counts = np.zeros(729, dtype=np.int32)
            for cw in cw_list:
                idx = w_to_idx(cw)
                cov = covered_arr[idx, :covered_lens[idx]]
                counts[cov] += 1
            return counts
            
        def is_redundant(cw, counts):
            idx = w_to_idx(cw)
            cov = covered_arr[idx, :covered_lens[idx]]
            return not np.any(counts[cov] <= 1)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            counts = build_counts(codewords)
            
            # 1. Try removals
            for i in range(len(codewords) - 1, -1, -1):
                if is_redundant(codewords[i], counts):
                    codewords.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try swaps
            swaps_done = 0
            for i in range(len(codewords)):
                if time.time() > deadline - 5:
                    break
                if swaps_done > 20:
                    break
                
                c_old = codewords[i]
                idx_old = w_to_idx(c_old)
                cov_old = covered_arr[idx_old, :covered_lens[idx_old]]
                
                # Temporarily remove
                counts[cov_old] -= 1
                uncovered = cov_old[counts[cov_old] == 0]
                
                if len(uncovered) == 0:
                    codewords.pop(i)
                    improved = True
                    break
                
                if len(uncovered) > 3:
                    counts[cov_old] += 1
                    continue
                    
                # Find candidates
                cand_set = set(covered_arr[uncovered[0], :covered_lens[uncovered[0]]].tolist())
                for u in uncovered[1:]:
                    u_cov = set(covered_arr[u, :covered_lens[u]].tolist())
                    cand_set.intersection_update(u_cov)
                    
                current_set = set(w_to_idx(cw) for cw in codewords)
                
                found_swap = False
                for c_new_idx in cand_set:
                    if c_new_idx == idx_old:
                        continue
                    if c_new_idx in current_set:
                        continue
                        
                    c_new = all_words[c_new_idx]
                    cov_new = covered_arr[c_new_idx, :covered_lens[c_new_idx]]
                    
                    # Add c_new
                    counts[cov_new] += 1
                    
                    # Check for victim
                    victim_idx = -1
                    for j in range(len(codewords)):
                        if j == i:
                            continue
                        if is_redundant(codewords[j], counts):
                            victim_idx = j
                            break
                            
                    if victim_idx != -1:
                        # Accept swap
                        idx_v = w_to_idx(codewords[victim_idx])
                        cov_v = covered_arr[idx_v, :covered_lens[idx_v]]
                        counts[cov_v] -= 1
                        
                        codewords.pop(victim_idx)
                        codewords.pop(i)
                        codewords.append(c_new)
                        
                        improved = True
                        found_swap = True
                        swaps_done += 1
                        break
                    else:
                        # Revert c_new
                        counts[cov_new] -= 1
                        
                if found_swap:
                    break
                else:
                    # Revert c_old
                    counts[cov_old] += 1

        return codewords

    # Generic fallback for other parameters
    all_words = [list(t) for t in itertools.product(range(q), repeat=n)]
    cover = {}
    for w in all_words:
        cov = set()
        cov.add(tuple(w))
        for i in range(n):
            for v in range(q):
                if v == w[i]:
                    continue
                nw = w.copy()
                nw[i] = v
                cov.add(tuple(nw))
        cover[tuple(w)] = cov
        
    remaining = set(all_words)
    codewords = []
    while remaining and time.time() < deadline:
        best_w = None
        best_count = -1
        sample = list(remaining)
        random.shuffle(sample)
        sample = sample[:100]
        for w in sample:
            count = len(cover[tuple(w)] & remaining)
            if count > best_count:
                best_count = count
                best_w = w
        if best_w is None:
            break
        codewords.append(best_w)
        remaining -= cover[tuple(best_w)]
    return codewords
# EVOLVE-BLOCK-END
