import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 150s wall-clock

    # ---- Generic fallback for non-target parameters ----
    if not (q == 3 and n == 7 and radius == 1):
        return _generic(q, n, radius)

    N = 3 ** n                       # 2187
    ball_size = 1 + n * (q - 1)      # 15

    # ---------- base-3 <-> int helpers ----------
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

    # ---------- precompute balls (radius-1 neighborhoods) ----------
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

    # cover_map[w] = list of codewords c whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # numpy matrix B[c][w] = 1 if w in ball(c)  -> vectorized gain sums
    B = np.zeros((N, N), dtype=np.int32)
    for c in range(N):
        B[c, balls[c]] = 1

    # ---------- greedy set-cover with randomized tie-break ----------
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = np.zeros(N, dtype=bool)
        cnt = np.array([ball_size] * N, dtype=np.int32)
        code = []
        uncovered_count = N
        while uncovered_count > 0:
            max_val = int(cnt.max())
            cands = np.flatnonzero(cnt == max_val)
            c = int(rng.choice(cands))
            code.append(c)
            ball_c = balls[c]
            newly = [w for w in ball_c if not covered[w]]
            if newly:
                covered[newly] = True
                uncovered_count -= len(newly)
                for w in newly:
                    cnt[cover_map[w]] -= 1
        return code

    # ---------- pure redundancy removal ----------
    def refine(code):
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if not (cover_count[balls[c]] > 1).all():
                    continue
                cover_count[balls[c]] -= 1
                code_list.pop(i)
                changed = True
                break
        return code_list

    # ---------- local search: remove-one + relocate/swap ----------
    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        code_set = set(code_list)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                ball_r = balls[c_remove]
                cover_count[ball_r] -= 1
                uncovered = [w for w in ball_r if cover_count[w] == 0]
                if not uncovered:
                    # pure removal
                    code_list.pop(i)
                    code_set.discard(c_remove)
                    improved = True
                    break

                # relocation: find a codeword whose ball covers all uncovered
                found = -1
                for c_add in range(N):
                    if c_add == c_remove or c_add in code_set:
                        continue
                    ball_a = balls[c_add]
                    ok = True
                    for w in uncovered:
                        if cover_count[w] > 0:
                            continue
                        if not B[c_add, w]:
                            ok = False
                            break
                    if ok:
                        found = c_add
                        break

                if found >= 0:
                    # swap: relocate c_remove -> found
                    cover_count[balls[found]] += 1
                    code_list[i] = found
                    code_set.discard(c_remove)
                    code_set.add(found)
                    improved = True
                    break
                else:
                    # rollback
                    cover_count[ball_r] += 1

        return code_list

    # ---------- driver: restarts -> best -> local search ----------
    best_code = None
    best_len = float('inf')

    # greedy restarts (cheap, vectorized)
    while time.time() < deadline - 25.0:
        seed = int((time.time() - start_time) * 100000) % (10 ** 9)
        code = greedy_cover(seed)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None and time.time() < deadline - 2.0:
        ls_code = local_search(best_code, 12345, deadline - 2.0)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

    if best_code is not None:
        final_code = refine(best_code)
    else:
        final_code = []

    return [int_to_base3(c, n) for c in final_code]


def _generic(q, n, radius):
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

    # light refinement
    cover_count = [0] * N
    for c in code:
        for w in balls[c]:
            cover_count[w] += 1
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            c = code[i]
            if all(cover_count[w] > 1 for w in balls[c]):
                for w in balls[c]:
                    cover_count[w] -= 1
                code.pop(i)
                changed = True
                break

    return [int_to_base(c, n, q) for c in code]
