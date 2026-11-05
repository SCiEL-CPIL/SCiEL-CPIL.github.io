import numpy as np
import random
import time

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0

    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n
        ball_size = 1 + n * (q - 1)

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute balls as a NumPy array
        balls_list = []
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls_list.append(neighbors)
        
        balls_np = np.array(balls_list, dtype=np.int32)

        # Precompute cover_map: for each word w, list of codewords c such that w in ball(c)
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls_list[c]:
                cover_map[w].append(c)
        
        # Convert cover_map to numpy arrays for faster iteration? 
        # Or keep as lists of numpy arrays?
        cover_map_np = [np.array(cm, dtype=np.int32) for cm in cover_map]

        def greedy_cover(seed):
            rng = random.Random(seed)
            uncovered = np.ones(N, dtype=bool)
            code = []
            
            while np.any(uncovered):
                # Vectorized gain calculation
                # gains[i] = number of uncovered words in ball i
                gains = uncovered[balls_np].sum(axis=1)
                
                # Find candidates with max gain
                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                
                # Random tie-breaking
                c = rng.choice(candidates.tolist())
                code.append(c)
                
                # Update uncovered
                uncovered[balls_np[c]] = False

            return code

        def refine(code):
            if not code:
                return code
            
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[balls_np[c]] += 1
            
            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Check if removing c leaves any word uncovered
                    # i.e., min(cover_count[ball[c]]) > 1
                    if np.min(cover_count[balls_np[c]]) > 1:
                        cover_count[balls_np[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                cover_count[balls_np[c]] += 1

            in_code = np.zeros(N, dtype=bool)
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    ball_rm = balls_np[c_remove]
                    
                    # Tentatively remove
                    cover_count[ball_rm] -= 1
                    
                    # Find newly uncovered words
                    newly_uncovered = ball_rm[cover_count[ball_rm] == 0]
                    
                    if len(newly_uncovered) == 0:
                        # Safe deletion
                        in_code[c_remove] = False
                        code_list.pop(i)
                        improved = True
                        break
                    else:
                        # Try to find a replacement
                        # Candidate pool: intersection of cover_map[w] for w in newly_uncovered
                        # Exclude c_remove and current code
                        cand_pool = None
                        for w in newly_uncovered:
                            cands_w = cover_map_np[w]
                            # Filter out in_code and c_remove
                            mask = ~in_code
                            mask[c_remove] = False
                            cands_w = cands_w[mask]
                            
                            if cand_pool is None:
                                cand_pool = set(cands_w.tolist())
                            else:
                                cand_pool.intersection_update(cands_w.tolist())
                        
                        found = -1
                        if cand_pool:
                            for c_add in cand_pool:
                                # Check if c_add covers all newly_uncovered
                                # Since c_add is in cover_map[w] for all w in newly_uncovered, it does.
                                found = c_add
                                break
                        
                        if found >= 0:
                            # Commit swap
                            in_code[c_remove] = False
                            in_code[found] = True
                            cover_count[balls_np[found]] += 1
                            code_list[i] = found
                            improved = True
                            break
                        else:
                            # Restore
                            cover_count[ball_rm] += 1

                # Try 2-for-1 move
                if time.time() < time_limit and len(code_list) >= 2:
                    n_codes = len(code_list)
                    # Sample pairs to avoid O(N^2) full scan if time is tight
                    # But N is ~170, so N^2 is ~30k, which is fast.
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c1 = code_list[i2]
                            c2 = code_list[j2]
                            b1 = balls_np[c1]
                            b2 = balls_np[c2]
                            
                            # Tentatively remove both
                            cover_count[b1] -= 1
                            cover_count[b2] -= 1
                            
                            # Find newly uncovered
                            # Union of b1 and b2 where count is 0
                            uncov_mask = np.zeros(N, dtype=bool)
                            uncov_mask[b1] = (cover_count[b1] == 0)
                            uncov_mask[b2] = (cover_count[b2] == 0)
                            newly_uncovered = np.where(uncov_mask)[0]
                            
                            if len(newly_uncovered) == 0:
                                # Safe deletion of both
                                in_code[c1] = False
                                in_code[c2] = False
                                code_list.pop(j2) # pop larger index first
                                code_list.pop(i2)
                                improved = True
                                break
                            else:
                                # Find replacement covering all newly_uncovered
                                cand_pool = None
                                for w in newly_uncovered:
                                    cands_w = cover_map_np[w]
                                    mask = ~in_code
                                    mask[c1] = False
                                    mask[c2] = False
                                    cands_w = cands_w[mask]
                                    
                                    if cand_pool is None:
                                        cand_pool = set(cands_w.tolist())
                                    else:
                                        cand_pool.intersection_update(cands_w.tolist())
                                
                                found = -1
                                if cand_pool:
                                    found = next(iter(cand_pool))
                                
                                if found >= 0:
                                    # Commit 2-for-1
                                    in_code[c1] = False
                                    in_code[c2] = False
                                    in_code[found] = True
                                    cover_count[balls_np[found]] += 1
                                    code_list[j2] = found
                                    code_list.pop(i2)
                                    improved = True
                                    break
                                else:
                                    # Restore
                                    cover_count[b1] += 1
                                    cover_count[b2] += 1
                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # 1. Multiple Greedy Restarts
        num_restarts = 100 # Can do more because greedy is fast now
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined
        
        # 2. Local Search on Best
        if best_code is not None:
            if time.time() < deadline - 10:
                ls_code = local_search(best_code, 42, deadline - 5)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code
            
            # 3. Kick / Re-greedy
            for kick_iter in range(20):
                if time.time() > deadline - 5:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k = min(5, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                
                # Re-greedy to fill gap
                # We can modify greedy to start with 'remaining'
                # But greedy_cover doesn't support start_code in this version.
                # Let's just run greedy_cover and take the union? No, that's not right.
                # We need a greedy that accepts initial code.
                
                # Let's implement a simple re-greedy:
                # Start with 'remaining', mark covered, then greedy add.
                uncovered = np.ones(N, dtype=bool)
                for c in remaining:
                    uncovered[balls_np[c]] = False
                
                re_code = list(remaining)
                while np.any(uncovered):
                    gains = uncovered[balls_np].sum(axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0: break
                    candidates = np.where(gains == max_gain)[0]
                    c = rng_k.choice(candidates.tolist())
                    re_code.append(c)
                    uncovered[balls_np[c]] = False
                
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                
                if time.time() < deadline - 3:
                    ls_code = local_search(re_refined, 2000 + kick_iter, deadline - 2)
                    if len(ls_code) < best_len:
                        best_len = len(ls_code)
                        best_code = ls_code

        if best_code is None:
            return []

        final_code = refine(best_code)
        result = [int_to_base3(c, n) for c in final_code]
        return result

    # Generic case
    return []
