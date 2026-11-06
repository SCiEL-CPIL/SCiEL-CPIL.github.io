import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius)

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

    # Precompute cover_map: for each word w, which codewords cover it
    cover_map = [list() for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    ball_sets = [set(b) for b in balls]

    # Precompute for each c, the set of words NOT in ball[c]
    # For local search swap: candidates for c_remove must cover all newly_uncovered words
    # i.e., c_add must be in intersection of cover_map[w] for w in newly_uncovered
    # Precompute nothing extra; use cover_map at runtime

    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = [False] * N
        cnt = [ball_size] * N
        code = []
        uncovered_count = N

        # Use a simpler approach: repeatedly pick the word with max uncovered coverage
        # Use numpy for speed
        uncovered_mask = np.ones(N, dtype=bool)
        cnt_arr = np.array(cnt, dtype=np.int32)

        in_code = [False] * N

        while uncovered_count > 0:
            # Compute marginal coverage for each candidate
            # For each c, marginal = number of uncovered words in ball[c]
            # Use precomputed ball membership
            # balls_np: shape (N, ball_size) with ball indices
            best_c = -1
            best_count = -1

            # Vectorized: for each c, count how many of its ball members are uncovered
            # balls as numpy array
            # This is O(N * ball_size) per iteration = 2187 * 15 = ~33k ops
            # With numpy: sum of uncovered_mask[ball] for each ball

            # Actually, let's use the heap approach but with numpy for initial setup
            # and a simpler loop

            # Simple O(N * ball_size) scan
            for c in range(N):
                if in_code[c]:
                    continue
                count = 0
                for w in balls[c]:
                    if uncovered_mask[w]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_c = c

            if best_c < 0:
                break

            c = best_c
            code.append(c)
            in_code[c] = True

            ball_c = balls[c]
            for w in ball_c:
                if uncovered_mask[w]:
                    uncovered_mask[w] = False
                    uncovered_count -= 1

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
            for w in balls[c]:
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
                for w in balls[c_remove]:
                    cover_count[w] -= 1
                    if cover_count[w] == 0:
                        newly_uncovered.append(w)

                if not newly_uncovered:
                    code_list.pop(i)
                    in_code[c_remove] = False
                    improved = True
                    break

                # Try 1-swap: find c_add not in code that covers all newly_uncovered
                # Precompute: intersection of cover_map[w] for w in newly_uncovered
                # Then check if any candidate in that intersection is not in_code
                if len(newly_uncovered) == 1:
                    candidates = cover_map[newly_uncovered[0]]
                elif len(newly_uncovered) == 2:
                    s1 = set(cover_map[newly_uncovered[0]])
                    s2 = cover_map[newly_uncovered[1]]
                    candidates = [c for c in s2 if c in s1]
                elif len(newly_uncovered) == 3:
                    s1 = set(cover_map[newly_uncovered[0]])
                    s2 = set(cover_map[newly_uncovered[1]])
                    s3 = cover_map[newly_uncovered[2]]
                    candidates = [c for c in s3 if c in s1 and c in s2]
                else:
                    # For 4+, use iterative filtering
                    s = set(cover_map[newly_uncovered[0]])
                    for w in newly_uncovered[1:]:
                        s &= set(cover_map[w])
                        if not s:
                            break
                    candidates = list(s)

                found = -1
                for c_add in candidates:
                    if c_add == c_remove or in_code[c_add]:
                        continue
                    found = c_add
                    break

                if found >= 0:
                    # Perform swap
                    code_list[i] = found
                    in_code[c_remove] = False
                    in_code[found] = True
                    for w in balls[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    # Revert deletion
                    for w in balls[c_remove]:
                        cover_count[w] += 1

        return code_list

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_code = None
    best_len = float('inf')

    # Run multiple restarts
    num_restarts = 80
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        code = greedy_cover(i)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None and time.time() < deadline - 5:
        # Local search on the best code
        ls_code = local_search(best_code, 42, deadline - 2)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

    if best_code is not None:
        final_code = refine(best_code)
    else:
        final_code = []

    result = []
    for c in final_code:
        result.append(int_to_base3(c, n))

    return result

def _generic_greedy(q, n, radius):
    N = q ** n
    def int_to_base(num, length, base):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % base
            num //= base
        return res

    balls = []
    for i in range(N):
        digits = int_to_base(i, n, q)
        neighbors = set([i])
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    val = 0
                    for d in new_digits:
                        val = val * q + d
                    neighbors.add(val)
        balls.append(list(neighbors))

    covered = set()
    code = []
    remaining = set(range(N))
    while remaining:
        best_c = None
        best_count = -1
        for c in remaining:
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        covered.update(balls[best_c])
        remaining -= set(balls[best_c])

    result = []
    for c in code:
        digits = int_to_base(c, n, q)
        result.append(digits)
    return result
