import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 150s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

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

        # Precompute balls (neighbors including self)
        balls = []
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
            balls.append(neighbors)

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)
        
        # Convert cover_map to sets for faster intersection
        cover_map_sets = [set(lst) for lst in cover_map]

        # Precompute ball sets for fast membership
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for fast iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with heap-based candidate selection."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            # Build heap: (-cnt[i], i) for max-heap behavior
            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)

            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-cnt[c2], c2))

            while uncovered_count > 0:
                # Pop valid candidates from heap
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not covered[i]:
                        break
                else:
                    break

                c = i
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion, and randomized k-exchanges (remove k, add 1)."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                
                # Phase 1: Exhaustive Deletion
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break
                    else:
                        for w in ball_indices[c_remove]:
                            cover_count[w] += 1

                if improved:
                    continue

                # Phase 2: Randomized 2-Swap (Remove 2, Add 1)
                # Sample random pairs
                n_codes = len(code_list)
                if n_codes >= 3:
                    for _ in range(500): # Try 500 random pairs
                        if time.time() > time_limit:
                            break
                        i1, i2 = rng.sample(range(n_codes), 2)
                        c_rem1 = code_list[i1]
                        c_rem2 = code_list[i2]
                        
                        uncov = []
                        for w in ball_indices[c_rem1]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.append(w)
                        for w in ball_indices[c_rem2]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                if w not in uncov: # Avoid duplicates if balls overlap
                                    uncov.append(w)
                            elif cover_count[w] == 0 and w in uncov:
                                pass # Already added
                            # Note: if w was in uncov from rem1, and rem2 also uncovered it, 
                            # cover_count[w] would be 0 from rem1, then decremented to -1? 
                            # No, cover_count tracks total. 
                            # If cover_count[w] was 1, rem1 makes it 0. rem2 makes it -1? 
                            # Wait, if w is in both balls, and cover_count was 1, 
                            # it means only one of the two codewords covered it? 
                            # No, cover_count[w] is the sum of all codewords covering w.
                            # If cover_count[w] == 1, and we remove c1, it becomes 0. 
                            # If we remove c2, and c2 also covered w, it becomes -1? 
                            # This implies my logic for "newly_uncovered" needs care.
                            
                        # Let's restart the logic for multi-removal:
                        # We need to know if ANY word has cover_count 0 after removing the set.
                        # Let's just collect words that hit 0.
                        
                        # Actually, simpler:
                        # Decrement all.
                        # Check if any is 0.
                        # If yes, we have a set U of uncovered words.
                        # Try to find a codeword c_add that covers all of U.
                        
                        # Let's redo the decrement/check properly.
                        
                        # Restore first
                        for w in ball_indices[c_rem1]:
                            cover_count[w] += 1
                        for w in ball_indices[c_rem2]:
                            cover_count[w] += 1

                        # Now try again
                        uncov_set = set()
                        for w in ball_indices[c_rem1]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov_set.add(w)
                        for w in ball_indices[c_rem2]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov_set.add(w)
                            elif cover_count[w] < 0:
                                # This shouldn't happen if cover_count was accurate
                                pass

                        if uncov_set:
                            # Find c_add in intersection of cover_map[w] for w in uncov_set
                            # Exclude c_rem1, c_rem2
                            candidates = None
                            for w in uncov_set:
                                cands_w = cover_map_sets[w] - {c_rem1, c_rem2}
                                if candidates is None:
                                    candidates = cands_w
                                else:
                                    candidates &= cands_w
                                if not candidates:
                                    break
                            
                            if candidates:
                                c_add = rng.choice(list(candidates))
                                # Apply swap
                                # Remove c_rem1, c_rem2
                                # Add c_add
                                
                                # We already decremented. 
                                # We need to increment for c_add.
                                for w in ball_indices[c_add]:
                                    cover_count[w] += 1
                                
                                # Update code_list
                                # Remove indices i1, i2
                                # Add c_add
                                # Be careful with indices if i1 < i2
                                if i1 < i2:
                                    code_list[i1] = c_add
                                    code_list.pop(i2)
                                else:
                                    code_list[i2] = c_add
                                    code_list.pop(i1)
                                
                                in_code[c_rem1] = False
                                in_code[c_rem2] = False
                                in_code[c_add] = True
                                
                                improved = True
                                break
                        else:
                            # Should not happen if we checked uncov_set
                            pass
                        
                        # Restore
                        for w in ball_indices[c_rem1]:
                            cover_count[w] += 1
                        for w in ball_indices[c_rem2]:
                            cover_count[w] += 1

                if improved:
                    continue

                # Phase 3: Randomized 3-Swap (Remove 3, Add 1)
                if n_codes >= 4:
                    for _ in range(200): # Try 200 random triples
                        if time.time() > time_limit:
                            break
                        i1, i2, i3 = rng.sample(range(n_codes), 3)
                        c_rem1 = code_list[i1]
                        c_rem2 = code_list[i2]
                        c_rem3 = code_list[i3]
                        
                        uncov_set = set()
                        for w in ball_indices[c_rem1]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov_set.add(w)
                        for w in ball_indices[c_rem2]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov_set.add(w)
                        for w in ball_indices[c_rem3]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov_set.add(w)

                        if uncov_set:
                            candidates = None
                            for w in uncov_set:
                                cands_w = cover_map_sets[w] - {c_rem1, c_rem2, c_rem3}
                                if candidates is None:
                                    candidates = cands_w
                                else:
                                    candidates &= cands_w
                                if not candidates:
                                    break
                            
                            if candidates:
                                c_add = rng.choice(list(candidates))
                                for w in ball_indices[c_add]:
                                    cover_count[w] += 1
                                
                                # Update code_list
                                # Remove i1, i2, i3, add c_add
                                # Create a new list to avoid index shifting issues
                                new_list = []
                                for idx, c in enumerate(code_list):
                                    if idx not in (i1, i2, i3):
                                        new_list.append(c)
                                new_list.append(c_add)
                                code_list = new_list
                                
                                in_code[c_rem1] = False
                                in_code[c_rem2] = False
                                in_code[c_rem3] = False
                                in_code[c_add] = True
                                
                                improved = True
                                break
                        else:
                            pass
                        
                        # Restore
                        for w in ball_indices[c_rem1]:
                            cover_count[w] += 1
                        for w in ball_indices[c_rem2]:
                            cover_count[w] += 1
                        for w in ball_indices[c_rem3]:
                            cover_count[w] += 1

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts
        num_restarts = 50
        for i in range(num_restarts):
            if time.time() > deadline - 15:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 10:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 8, time.time() + 5.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 12:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 10:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 4.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(3):
                if time.time() > deadline - 2:
                    break
                if len(final_code) <= 150:
                    break
                ls_code2 = local_search(final_code, 99 + polish_iter * 13, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
                else:
                    break
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((N, nb_count), dtype=np.int32)
    for idx in range(N):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        best_idx = -1
        max_gain = 0
        for i in range(N):
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if max_gain == 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
