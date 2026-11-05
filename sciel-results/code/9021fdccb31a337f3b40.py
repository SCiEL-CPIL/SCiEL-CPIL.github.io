import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 140.0

    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        # Precompute ball membership as bitsets (Python ints for speed)
        balls = []
        for i in range(N):
            digits = [0] * n
            tmp = i
            for pos in range(n - 1, -1, -1):
                digits[pos] = tmp % 3
                tmp //= 3
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        new_int = 0
                        for d in new_digits:
                            new_int = new_int * 3 + d
                        neighbors.append(new_int)
            balls.append(neighbors)

        # Bitset representation for fast set operations
        ball_bitsets = []
        for b in balls:
            bs = 0
            for w in b:
                bs |= (1 << w)
            ball_bitsets.append(bs)

        # cover_map: for each word, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # cover_bitsets: for each word, bitset of codewords that cover it
        cover_bitsets = []
        for w in range(N):
            bs = 0
            for c in cover_map[w]:
                bs |= (1 << c)
            cover_bitsets.append(bs)

        def greedy_cover(seed, start_code=None):
            rng = random.Random(seed)
            covered_bits = 0
            code = []

            if start_code:
                for c in start_code:
                    code.append(c)
                    covered_bits |= ball_bitsets[c]

            uncovered_count = N - bin(covered_bits).count('1')

            # Build initial gain array
            gain = [0] * N
            for c in range(N):
                gain[c] = bin(ball_bitsets[c] & ~covered_bits).count('1')

            heap = [(-gain[i], i) for i in range(N) if gain[i] > 0]
            heapq.heapify(heap)
            in_heap = [False] * N
            for _, i in heap:
                in_heap[i] = True

            while uncovered_count > 0:
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if not in_heap[i]:
                        continue
                    if gain[i] == -neg_val:
                        break
                else:
                    break

                c = i
                in_heap[c] = False
                code.append(c)

                # Update gains
                newly_covered = ball_bitsets[c] & ~covered_bits
                covered_bits |= ball_bitsets[c]
                uncovered_count -= bin(newly_covered).count('1')

                # Update gain for all codewords that share words with newly_covered
                # For each newly covered word w, decrement gain for all c2 in cover_map[w]
                tmp = newly_covered
                while tmp:
                    low = tmp & (-tmp)
                    w = low.bit_length() - 1
                    tmp ^= low
                    for c2 in cover_map[w]:
                        if c2 != c and gain[c2] > 0:
                            old_gain = gain[c2]
                            gain[c2] -= 1
                            if gain[c2] > 0:
                                heapq.heappush(heap, (-gain[c2], c2))
                                in_heap[c2] = True
                            else:
                                in_heap[c2] = False

            return code

        def refine(code):
            if not code:
                return code

            # Count how many times each word is covered
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Check if removing c leaves any word uncovered
                    all_covered = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            all_covered = False
                            break
                    if all_covered:
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

            # Use bitset for in_code tracking
            in_code_bits = 0
            for c in code_list:
                in_code_bits |= (1 << c)

            # Cover count array
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)

                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Try deletion
                    newly_uncovered = []
                    for w in balls[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code_bits &= ~(1 << c_remove)
                        improved = True
                        break

                    # Try 1-swap
                    found = -1
                    if len(newly_uncovered) <= 8:
                        # Find candidate that covers all newly_uncovered
                        # Intersect cover sets
                        cand_bits = None
                        for w in newly_uncovered:
                            cb = cover_bitsets[w] & ~in_code_bits & ~(1 << c_remove)
                            if cand_bits is None:
                                cand_bits = cb
                            else:
                                cand_bits &= cb
                            if cand_bits == 0:
                                break
                        if cand_bits:
                            # Pick one candidate
                            low = cand_bits & (-cand_bits)
                            c_add = low.bit_length() - 1
                            # Verify it covers all
                            bs = ball_bitsets[c_add]
                            all_cov = True
                            for w in newly_uncovered:
                                if not (bs >> w) & 1:
                                    all_cov = False
                                    break
                            if all_cov:
                                found = c_add

                    if found >= 0:
                        code_list[i] = found
                        in_code_bits &= ~(1 << c_remove)
                        in_code_bits |= (1 << found)
                        for w in balls[found]:
                            cover_count[w] += 1
                        improved = True
                        break
                    else:
                        for w in balls[c_remove]:
                            cover_count[w] += 1

                # 2-for-1 swap: remove two, add one
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    pairs_done = 0
                    max_pairs = min(50, n_codes * (n_codes - 1) // 2)
                    for i2 in range(n_codes):
                        if time.time() > time_limit or pairs_done >= max_pairs:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit or pairs_done >= max_pairs:
                                break
                            pairs_done += 1
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]

                            uncov = []
                            for w in balls[c_rem1]:
                                cover_count[w] -= 1
                                if cover_count[w] == 0:
                                    uncov.append(w)
                            for w in balls[c_rem2]:
                                cover_count[w] -= 1
                                if cover_count[w] == 0 and w not in uncov:
                                    uncov.append(w)

                            if uncov and len(uncov) <= 10:
                                cand_bits = None
                                for w in uncov:
                                    cb = cover_bitsets[w] & ~in_code_bits & ~(1 << c_rem1) & ~(1 << c_rem2)
                                    if cand_bits is None:
                                        cand_bits = cb
                                    else:
                                        cand_bits &= cb
                                    if cand_bits == 0:
                                        break
                                if cand_bits:
                                    low = cand_bits & (-cand_bits)
                                    c_add = low.bit_length() - 1
                                    bs = ball_bitsets[c_add]
                                    all_cov = True
                                    for w in uncov:
                                        if not (bs >> w) & 1:
                                            all_cov = False
                                            break
                                    if all_cov:
                                        # Execute swap
                                        code_list[j2] = c_add
                                        in_code_bits &= ~(1 << c_rem2)
                                        in_code_bits |= (1 << c_add)
                                        for w in balls[c_add]:
                                            cover_count[w] += 1
                                        code_list.pop(i2)
                                        in_code_bits &= ~(1 << c_rem1)
                                        improved = True
                                        break
                            # Undo
                            for w in balls[c_rem1]:
                                cover_count[w] += 1
                            for w in balls[c_rem2]:
                                cover_count[w] += 1
                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Phase 1: Multiple greedy restarts with local search
        num_restarts = 80
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined

            if time.time() > deadline - 15:
                continue

            ls_time = min(deadline - 10, time.time() + 3.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Phase 2: Kick/re-greedy on best solution
        if best_code is not None:
            kick_configs = [(3, 500), (4, 501), (5, 502), (6, 503), (4, 504), (5, 505), (3, 506), (6, 507), (4, 508), (5, 509)]
            for kick_iter, (k, seed_k) in enumerate(kick_configs):
                if time.time() > deadline - 15:
                    break
                rng_k = random.Random(seed_k)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(1000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 12:
                    break
                ls_code = local_search(re_refined, 2000 + kick_iter, min(deadline - 10, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Phase 3: Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(5):
                if time.time() > deadline - 2:
                    break
                if len(final_code) <= 155:
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
            digits = [0] * n
            tmp = c
            for pos in range(n - 1, -1, -1):
                digits[pos] = tmp % 3
                tmp //= 3
            result.append(digits)
        return result

    # Generic fallback
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    neighbors = []
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
        neighbors.append(nb_list)

    uncovered = [True] * N
    code = []
    while any(uncovered):
        if time.time() > deadline:
            break
        best_idx = -1
        max_gain = 0
        for i in range(N):
            gain = sum(1 for w in neighbors[i] if uncovered[w])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if max_gain == 0:
            break
        code.append(best_idx)
        for w in neighbors[best_idx]:
            uncovered[w] = False

    return [idx_to_word(i) for i in code]
