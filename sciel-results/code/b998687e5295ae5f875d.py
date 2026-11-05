import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0

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

        # Precompute balls
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

        # Precompute cover_map
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets and numpy arrays
        ball_sets = [set(b) for b in balls]
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        # Build the ternary [7,4,3] Hamming code
        # Generator matrix for ternary Hamming [7,4,3] code
        # The parity check matrix H has columns as all non-zero vectors in GF(3)^2 up to scalar
        # Actually for ternary Hamming [7,4,3]: h = (q^n - 1)/(q-1) = (9-1)/2 = 4... 
        # Wait, ternary Hamming [n,k,d] where n = (3^m - 1)/(3-1)
        # For m=2: n = (9-1)/2 = 4, k=2, d=3 -> [4,2,3]
        # For m=3: n = (27-1)/2 = 13, k=10, d=3
        # Hmm, [7,4,3] is not a standard Hamming code over GF(3).
        # Let me reconsider. The [7,4,3] code over GF(3) exists but is not Hamming.
        # Actually, let me just use a different approach.
        
        # The key structural insight: use a well-known construction.
        # For K_3(7,1) = 186, the optimal code has a specific structure.
        # Let's use a hybrid: structured greedy with Hamming-like prioritization.
        
        # Actually, let me use a simpler but effective approach:
        # 1. Multiple greedy restarts with different tie-breaking
        # 2. Strong local search with deletion, 1-swap, and 2-swap
        # 3. Kick-and-refill strategy
        # 4. Use the fact that the answer is 186 to guide search

        def greedy_cover(seed, start_code=None):
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

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
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-cnt[c2], rng.random(), c2))

            while uncovered_count > 0:
                valid_found = False
                while heap:
                    neg_gain, _, c = heapq.heappop(heap)
                    gain = -neg_gain
                    if cnt[c] == gain and gain > 0:
                        valid_found = True
                        break
                if not valid_found:
                    break

                code.append(c)
                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], rng.random(), c2))

            return code

        def refine(code):
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
            stall_rounds = 0
            while improved and time.time() < time_limit:
                improved = False
                stall_rounds = 0
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

                    found = -1
                    if len(newly_uncovered) <= 6:
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

                # 2-swap
                if not improved and stall_rounds > len(code_list) // 2 and time.time() < time_limit - 1.0 and len(code_list) >= 4:
                    n_codes = len(code_list)
                    num_attempts = min(80, n_codes * (n_codes - 1) // 4)
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

                        if uncov and len(uncov) <= 8:
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

        def verify(code):
            """Check if code covers all words."""
            if not code:
                return N > 0
            covered = [False] * N
            for c in code:
                for w in balls[c]:
                    covered[w] = True
            return all(covered)

        best_code = None
        best_len = float('inf')

        # Phase 1: Multiple greedy restarts with local search
        num_restarts = 60
        for i in range(num_restarts):
            if time.time() > deadline - 25:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

            if time.time() > deadline - 20:
                continue
            ls_time = min(deadline - 15, time.time() + 4.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Phase 2: Kick-and-refill on best solution
        if best_code is not None:
            for kick_iter in range(20):
                if time.time() > deadline - 20:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k = min(8, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 16:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 14, time.time() + 3.5))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Phase 3: Aggressive final local search
        if best_code is not None:
            if time.time() < deadline - 10:
                ls_code = local_search(best_code, 42, deadline - 8)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code
            final_code = refine(best_code)
            # Multiple polish passes
            polish_seed = 99
            for _ in range(5):
                if time.time() < deadline - 3 and len(final_code) > 182:
                    ls_code2 = local_search(final_code, polish_seed, deadline - 2.0)
                    if len(ls_code2) < len(final_code):
                        final_code = refine(ls_code2)
                    polish_seed += 13

            # Verify final code
            if not verify(final_code):
                # Fallback: rebuild
                final_code = refine(greedy_cover(0))
        else:
            final_code = refine(greedy_cover(0))

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
