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

        # Precompute ball sets for fast membership (swap search)
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for faster iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        # Precompute gain matrix row-sums for greedy: gain[c] = number of
        # currently-uncovered words in ball c.  We track it incrementally.
        # gain[c] = sum over w in ball c of (1 - covered[w])
        gain = [ball_size] * N
        # For quick update: when w becomes covered, decrement gain[c] for c in cover_map[w]

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with random tie-breaking using a heap.
            If start_code is given, start from that partial cover and continue
            covering the remaining words (kick / re-greedy)."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N
            
            # Heap elements: (-gain, tie_breaker, idx)
            heap = []
            for i in range(N):
                heapq.heappush(heap, (-cnt[i], rng.random(), i))
                
            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                                heapq.heappush(heap, (-cnt[c2], rng.random(), c2))

            while uncovered_count > 0:
                # Find valid max
                valid_found = False
                while heap:
                    neg_gain, _, c = heapq.heappop(heap)
                    gain = -neg_gain
                    # Check if this entry is still valid (cnt matches) and gain is positive
                    if cnt[c] == gain and gain > 0:
                        valid_found = True
                        break
                    # If gain is 0 or stale, discard and continue popping
                
                if not valid_found:
                    # Should not happen if uncovered_count > 0, but safety break
                    break
                    
                code.append(c)
                
                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            heapq.heappush(heap, (-cnt[c2], rng.random(), c2))

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
            """Local search with deletion, 1-swap, and selective 2-swap moves."""
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

            # Precompute for each codeword the set of words it uniquely covers
            # (words where cover_count == 1) - these are the "critical" words
            def get_critical_words(c):
                """Return list of words in ball[c] that are only covered by c."""
                crit = []
                for w in ball_indices[c]:
                    if cover_count[w] == 1:
                        crit.append(w)
                return crit

            improved = True
            stall_rounds = 0
            while improved and time.time() < time_limit:
                improved = False
                stall_rounds = 0
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                # Pass 1: deletion and 1-swap
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Try deletion first
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

                    # Try 1-swap
                    found = -1
                    if len(newly_uncovered) <= 6:
                        # Sort by rarity of covering codewords for efficiency
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        cand_pool = None
                        for w in crit_sorted:
                            candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if cand_pool is None:
                                cand_pool = set(candidates_w)
                            else:
                                cand_pool.intersection_update(candidates_w)
                            if not cand_pool:
                                break
                        if cand_pool:
                            for c_add in cand_pool:
                                bs = ball_sets[c_add]
                                if all(w in bs for w in newly_uncovered):
                                    found = c_add
                                    break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in ball_indices[found]:
                            cover_count[w] += 1
                        improved = True
                        break
                    else:
                        for w in ball_indices[c_remove]:
                            cover_count[w] += 1
                        stall_rounds += 1

                # Pass 2: selective 2-swap (only when we've stalled and have time)
                if not improved and stall_rounds > len(code_list) // 2 and time.time() < time_limit - 1.0 and len(code_list) >= 4:
                    n_codes = len(code_list)
                    # Try a limited number of random 2-swap attempts
                    num_attempts = min(50, n_codes * (n_codes - 1) // 4)
                    attempted = set()
                    for _ in range(num_attempts):
                        if time.time() > time_limit - 0.5:
                            break
                        i2 = rng.randint(0, n_codes - 1)
                        j2 = rng.randint(0, n_codes - 1)
                        if i2 == j2 or (i2, j2) in attempted or (j2, i2) in attempted:
                            continue
                        attempted.add((i2, j2))
                        
                        c_rem1 = code_list[i2]
                        c_rem2 = code_list[j2]
                        
                        uncov = set()
                        for w in ball_indices[c_rem1]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.add(w)
                        for w in ball_indices[c_rem2]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.add(w)
                        
                        if uncov and len(uncov) <= 7:
                            cand_pool = None
                            for w in sorted(uncov, key=lambda x: len(cover_map[x])):
                                candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_rem1 and c2 != c_rem2]
                                if cand_pool is None:
                                    cand_pool = set(candidates_w)
                                else:
                                    cand_pool.intersection_update(candidates_w)
                                if not cand_pool:
                                    break
                            found2 = -1
                            if cand_pool:
                                for c_add in cand_pool:
                                    bs = ball_sets[c_add]
                                    if all(w in bs for w in uncov):
                                        found2 = c_add
                                        break
                            if found2 >= 0:
                                # Apply the 2-swap
                                code_list[j2] = found2
                                in_code[c_rem2] = False
                                in_code[found2] = True
                                for w in ball_indices[found2]:
                                    cover_count[w] += 1
                                code_list.pop(i2)
                                in_code[c_rem1] = False
                                improved = True
                                break
                            else:
                                for w in ball_indices[c_rem1]:
                                    cover_count[w] += 1
                                for w in ball_indices[c_rem2]:
                                    cover_count[w] += 1
                        else:
                            for w in ball_indices[c_rem1]:
                                cover_count[w] += 1
                            for w in ball_indices[c_rem2]:
                                cover_count[w] += 1
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts with different seeds, then local search each
        # refined solution.  Also do kick/re-greedy: remove a few codewords
        # from the best solution and re-greedy to fill the gap.
        num_restarts = 80
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 15:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 12, time.time() + 5.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts on the best solution found so far
        if best_code is not None:
            for kick_iter in range(15):
                if time.time() > deadline - 18:
                    break
                rng_k = random.Random(1000 + kick_iter)
                # Remove k random codewords (kick), then re-greedy the rest
                k = min(6, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 14:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 12, time.time() + 4.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Aggressive final local search + refine
        if best_code is not None:
            # Long final local search
            if time.time() < deadline - 8:
                ls_code = local_search(best_code, 42, deadline - 6)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code
            final_code = refine(best_code)
            # Multiple polish passes
            polish_seed = 99
            for _ in range(3):
                if time.time() < deadline - 2 and len(final_code) > 180:
                    ls_code2 = local_search(final_code, polish_seed, deadline - 1.5)
                    if len(ls_code2) < len(final_code):
                        final_code = refine(ls_code2)
                    polish_seed += 13
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case (shouldn't be called for this task, but included for completeness)
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

    # Generic greedy
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
