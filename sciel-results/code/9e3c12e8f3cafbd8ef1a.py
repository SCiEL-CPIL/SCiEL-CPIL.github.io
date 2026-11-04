import numpy as np
import random
import time

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

    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = [False] * N
        cnt = [ball_size] * N
        code = []
        uncovered_count = N

        while uncovered_count > 0:
            max_val = max(cnt)
            candidates = [i for i in range(N) if cnt[i] == max_val]
            c = rng.choice(candidates)
            code.append(c)

            ball_c = balls[c]
            for w in ball_c:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1

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

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                for w in balls[c_remove]:
                    cover_count[w] -= 1
                uncovered = [w for w in range(N) if cover_count[w] == 0]
                if not uncovered:
                    code_list.pop(i)
                    improved = True
                    break

                found = -1
                for c_add in range(N):
                    if c_add == c_remove or c_add in code_list:
                        continue
                    ball_add = set(balls[c_add])
                    if all(w in ball_add for w in uncovered):
                        found = c_add
                        break
                if found >= 0:
                    code_list[i] = found
                    for w in balls[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    for w in balls[c_remove]:
                        cover_count[w] += 1

        return code_list

    start_time = time.time()
    deadline = start_time + 110

    best_code = None
    best_len = float('inf')

    num_restarts = 40
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        code = greedy_cover(i)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None and time.time() < deadline - 20:
        ls_code = local_search(best_code, 42, deadline - 5)
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
