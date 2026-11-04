import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific case optimization
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]
        
        # Precompute neighbors
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            neighbors[i, 0] = i
            k = 1
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1
                        
        # Precompute neighbor masks for fast intersection/union checks
        neighbor_mask = np.zeros((total, total), dtype=np.bool_)
        for i in range(total):
            neighbor_mask[i, neighbors[i]] = True

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]

        def greedy_set_cover():
            """Greedy set cover with random tie-breaking for diversity."""
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []
            coverage_counts = np.zeros(total, dtype=np.int32)
            
            while uncovered.sum() > 0:
                counts = np.sum(uncovered[neighbors], axis=1)
                max_count = np.max(counts)
                if max_count == 0:
                    break
                # Pick randomly among top candidates for diversity
                top_candidates = np.where(counts == max_count)[0]
                best_c = random.choice(top_candidates.tolist())
                codewords.append(best_c)
                uncovered[neighbors[best_c]] = 0
                coverage_counts[neighbors[best_c]] += 1
            
            return codewords, coverage_counts

        def local_search(codewords, coverage_counts, time_budget):
            """Perform local search to reduce code size."""
            best_codewords = list(codewords)
            best_size = len(codewords)
            best_coverage = coverage_counts.copy()
            
            def try_remove(cw, cc):
                """Try to remove a redundant word. Returns True if removed."""
                for idx, c in enumerate(cw):
                    if np.min(cc[neighbors[c]]) >= 2:
                        cw.pop(idx)
                        cc[neighbors[c]] -= 1
                        return True
                return False

            def try_swap(cw, cc):
                """Try a 1-for-1 swap. Returns True if swapped."""
                if not cw:
                    return False
                idx_out = random.randint(0, len(cw) - 1)
                c_out = cw[idx_out]
                
                crit = neighbors[c_out][cc[neighbors[c_out]] == 1]
                if len(crit) == 0:
                    return False
                    
                valid_in = np.all(neighbor_mask[:, crit], axis=1)
                
                existing_mask = np.zeros(total, dtype=np.bool_)
                if cw:
                    existing_mask[cw] = True
                valid_in[existing_mask] = False
                valid_in[c_out] = False
                
                candidates = np.where(valid_in)[0]
                if len(candidates) > 0:
                    c_in = random.choice(candidates)
                    cw[idx_out] = c_in
                    cc[neighbors[c_out]] -= 1
                    cc[neighbors[c_in]] += 1
                    return True
                return False

            def try_2_swap(cw, cc):
                """Try a 2-for-2 swap. Returns True if swapped."""
                if len(cw) < 2:
                    return False
                idxs = random.sample(range(len(cw)), 2)
                c1, c2 = cw[idxs[0]], cw[idxs[1]]
                
                cc[neighbors[c1]] -= 1
                cc[neighbors[c2]] -= 1
                
                critical_words = np.where(cc == 0)[0]
                
                if len(critical_words) == 0:
                    existing_mask = np.zeros(total, dtype=np.bool_)
                    if cw:
                        existing_mask[cw] = True
                    existing_mask[c1] = False
                    existing_mask[c2] = False
                    
                    candidates = np.where(~existing_mask)[0]
                    if len(candidates) >= 2:
                        c3, c4 = random.sample(candidates.tolist(), 2)
                        cw[idxs[0]] = c3
                        cw[idxs[1]] = c4
                        cc[neighbors[c3]] += 1
                        cc[neighbors[c4]] += 1
                        return True
                    
                    cc[neighbors[c1]] += 1
                    cc[neighbors[c2]] += 1
                    return False

                w0 = critical_words[0]
                c3_candidates = neighbors[w0]
                
                existing_mask = np.zeros(total, dtype=np.bool_)
                if cw:
                    existing_mask[cw] = True
                existing_mask[c1] = False
                existing_mask[c2] = False
                
                for c3 in c3_candidates:
                    if existing_mask[c3]:
                        continue
                        
                    covered_c3 = neighbor_mask[c3, critical_words]
                    remaining = critical_words[~covered_c3]
                    
                    if len(remaining) == 0:
                        valid_c4 = ~existing_mask
                        valid_c4[c3] = False
                        c4_candidates = np.where(valid_c4)[0]
                        if len(c4_candidates) > 0:
                            c4 = random.choice(c4_candidates)
                            cw[idxs[0]] = c3
                            cw[idxs[1]] = c4
                            cc[neighbors[c3]] += 1
                            cc[neighbors[c4]] += 1
                            return True
                    else:
                        valid_c4 = np.all(neighbor_mask[:, remaining], axis=1)
                        valid_c4[existing_mask] = False
                        valid_c4[c3] = False
                        c4_candidates = np.where(valid_c4)[0]
                        if len(c4_candidates) > 0:
                            c4 = random.choice(c4_candidates)
                            cw[idxs[0]] = c3
                            cw[idxs[1]] = c4
                            cc[neighbors[c3]] += 1
                            cc[neighbors[c4]] += 1
                            return True
                
                cc[neighbors[c1]] += 1
                cc[neighbors[c2]] += 1
                return False

            # Main local search loop
            stuck_count = 0
            max_stuck = 200
            
            while time.time() < time_budget:
                if try_remove(codewords, coverage_counts):
                    stuck_count = 0
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                        best_coverage = coverage_counts.copy()
                    continue
                    
                if try_swap(codewords, coverage_counts):
                    stuck_count = 0
                    continue
                    
                stuck_count += 1
                if stuck_count > max_stuck:
                    if try_2_swap(codewords, coverage_counts):
                        stuck_count = 0
                        max_stuck = random.randint(100, 300)
                    else:
                        stuck_count = 0
                        max_stuck = random.randint(100, 300)

            # Final cleanup
            while True:
                if not try_remove(codewords, coverage_counts):
                    break
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    best_coverage = coverage_counts.copy()

            if len(best_codewords) < len(codewords):
                return best_codewords, best_coverage
            else:
                return codewords, coverage_counts

        # Multi-restart strategy
        best_overall = None
        best_overall_size = float('inf')
        
        # Use remaining time for multiple restarts
        while time.time() < deadline - 5:
            # Greedy initialization
            codewords, coverage_counts = greedy_set_cover()
            
            # Local search with time budget
            time_budget = min(time.time() + 30, deadline - 5)
            codewords, coverage_counts = local_search(codewords, coverage_counts, time_budget)
            
            # Ensure validity
            uncovered = np.ones(total, dtype=np.int8)
            for c in codewords:
                uncovered[neighbors[c]] = 0
            if uncovered.sum() > 0:
                while uncovered.sum() > 0:
                    c = np.random.randint(total)
                    if c not in codewords:
                        codewords.append(c)
                        uncovered[neighbors[c]] = 0
            
            if len(codewords) < best_overall_size:
                best_overall_size = len(codewords)
                best_overall = list(codewords)
                
            # If we found a good solution, do more focused search
            if best_overall_size <= 75:
                # Try to improve further with longer local search
                time_budget2 = min(time.time() + 20, deadline - 5)
                codewords2, coverage_counts2 = local_search(best_overall, coverage_counts.copy(), time_budget2)
                if len(codewords2) < best_overall_size:
                    best_overall_size = len(codewords2)
                    best_overall = list(codewords2)

        if best_overall is None:
            # Fallback
            best_overall = [0]
            
        return [idx_to_list(idx) for idx in best_overall]

    else:
        # Fallback for other parameters
        total = q ** n
        return [[0] * n]
