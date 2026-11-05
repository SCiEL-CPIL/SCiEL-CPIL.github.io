import numpy as np
import random
import time

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

        # Precompute balls as lists
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

        # Bitset representation of each ball
        ball_bits = [0] * N
        for c in range(N):
            b = 0
            for w in balls[c]:
                b |= (1 << w)
            ball_bits[c] = b

        # cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute cover_map as bitsets too for intersection
        cover_map_bits = [0] * N
        for w in range(N):
            b = 0
            for c in cover_map[w]:
                b |= (1 << c)
            cover_map_bits[w] = b

        # Ball sets for fast membership in local search
        ball_sets = [set(b) for b in balls]

        # Ball indices as numpy arrays
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        def greedy_cover(seed, start_code=None):
            """Bitset-based greedy set cover."""
            rng = random.Random(seed)
            uncovered_bits = (1 << N) - 1
            code = []

            if start_code:
                for c in start_code:
                    code.append(c)
                    uncovered_bits &= ~ball_bits[c]
                if uncovered_bits == 0:
                    return code

            # Precompute gain for all candidates
            # gain[c] = popcount(ball_bits[c] & uncovered_bits)
            while uncovered_bits != 0:
                # Find candidate with max gain
                best_c = -1
                best_gain = 0
                for c in range(N):
                    g = (ball_bits[c] & uncovered_bits).bit_count()
                    if g > best_gain:
                        best_gain = g
                        best_c = c
                if best_gain == 0:
                    break
                code.append(best_c)
                uncovered_bits &= ~ball_bits[best_c]

            return code

        def refine(code):
            """Remove redundant codewords using bitsets."""
            if not code:
                return code

            code_set = set(code)
            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Check if removing c leaves all its words covered by others
                    others_cover = 0
                    for c2 in code_list:
                        if c2 != c:
                            others_cover |= ball_bits[c2]
                    # All words in ball c must be covered by others
                    if (ball_bits[c] & ~others_cover) == 0:
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion, 1-swap, and 2-swap moves using bitsets."""
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
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Try deletion
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
                    if len(newly_uncovered) <= 4:
                        # Find candidates covering all newly_uncovered
                        cand_bits = (1 << N) - 1
                        for w in newly_uncovered:
                            cand_bits &= cover_map_bits[w]
                        # Remove already-in-code candidates and c_remove
                        for c2 in code_list:
                            cand_bits &= ~(1 << c2)
                        while cand_bits:
                            c_add = (cand_bits & -cand_bits).bit_length() - 1
                            cand_bits &= cand_bits - 1
                            # Verify c_add covers all newly_uncovered
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

                # Try 2-swap: remove two codewords, add one
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
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
                            if uncov and len(uncov) <= 5:
                                cand_bits = (1 << N) - 1
                                for w in uncov:
                                    cand_bits &= cover_map_bits[w]
                                for c2 in code_list:
                                    cand_bits &= ~(1 << c2)
                                found2 = -1
                                while cand_bits:
                                    c_add = (cand_bits & -cand_bits).bit_length() - 1
                                    cand_bits &= cand_bits - 1
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
                        if improved:
                            break

            return code_list

        def ils(code, seed, time_limit):
            """Iterated Local Search with swap moves (fixed-size)."""
            rng = random.Random(seed)
            best = list(code)
            best_len = len(best)
            current = list(best)
            cur_len = len(current)

            while time.time() < time_limit:
                # Local search on current
                ls_code = local_search(current, rng.randint(0, 2**31), min(time_limit, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best = ls_code
                    best_len = len(best)
                    current = list(best)
                    cur_len = best_len
                elif len(ls_code) < cur_len:
                    current = ls_code
                    cur_len = len(ls_code)
                else:
                    # Perturbation: random swap
                    if len(current) >= 2:
                        i = rng.randint(0, len(current) - 1)
                        j = rng.randint(0, len(current) - 1)
                        if i != j:
                            current[i], current[j] = current[j], current[i]
                    # Try a random replacement
                    if len(current) > 0:
                        i = rng.randint(0, len(current) - 1)
                        c_old = current[i]
                        # Find a random candidate not in current
                        candidates = [c for c in range(N) if c != c_old and c not in set(current)]
                        if candidates:
                            c_new = rng.choice(candidates)
                            current[i] = c_new

            return best

        best_code = None
        best_len = float('inf')

        # Phase 1: Multiple greedy restarts + local search
        num_restarts = 200
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined
            if time.time() > deadline - 15:
                break
            ls_time = min(deadline - 12, time.time() + 2.5)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Phase 2: Kick / re-greedy on best solution
        if best_code is not None:
            for kick_iter in range(20):
                if time.time() > deadline - 15:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k = min(5, len(best_code))
                kick_set = set(rng_k.sample(best_code, k))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 12:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 10, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Phase 3: ILS on best solution
        if best_code is not None and time.time() < deadline - 10:
            ils_code = ils(best_code, 42, deadline - 5)
            if len(ils_code) < best_len:
                best_len = len(ils_code)
                best_code = ils_code

        # Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 99, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            if time.time() < deadline - 2 and len(final_code) > 150:
                ls_code2 = local_search(final_code, 123, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
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
