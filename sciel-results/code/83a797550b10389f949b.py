import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    ball_set = [set(b) for b in ball_indices]

    # Greedy construction
    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = []
        while not np.all(covered):
            best_c = -1
            max_new = -1
            for c in range(total):
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code

    current_code = set(greedy_build())
    best_code = list(current_code)
    best_size = len(best_code)

    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1

    def run_delete_phase(code_set, cnt_arr):
        changed = False
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed

    def try_swap(c_remove, c_add, code_set, cnt_arr):
        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        # Only need: for x in ball_rem \ ball_add, cnt[x] >= 2
        in_rem_not_add = [x for x in ball_rem if x not in ball_set[c_add]]
        if in_rem_not_add:
            if np.min(cnt_arr[in_rem_not_add]) < 2:
                return False
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        return True

    def delete_after_swap(code_set, cnt_arr):
        while run_delete_phase(code_set, cnt_arr):
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)
            if best_size <= 160:
                break

    # Local Search Loop
    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        while run_delete_phase(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)

        # 2. Swap Phase with critical-word targeting
        swaps_attempted = 0
        max_swaps = 500
        no_improve = 0
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break

            c = random.choice(list(current_code))
            ball_rem = ball_indices_np[c]
            # Critical words: covered only by c
            critical = [x for x in ball_rem if cnt[x] == 1]

            if not critical:
                # c is redundant; delete it
                current_code.remove(c)
                cnt[ball_rem] -= 1
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                continue

            # Find candidates that cover all critical words
            found = False
            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break
                c_prime = random.randint(0, total - 1)
                if c_prime == c or c_prime in current_code:
                    continue
                # Check coverage of all critical words
                ok = all(x in ball_set[c_prime] for x in critical)
                if ok and try_swap(c, c_prime, current_code, cnt):
                    delete_after_swap(current_code, cnt)
                    found = True
                    no_improve = 0
                    break

            if not found:
                no_improve += 1

            if best_size <= 160:
                break

        # Perturb if stuck
        if no_improve >= 30 and len(current_code) < total:
            rand_cw = random.randint(0, total - 1)
            if rand_cw not in current_code:
                current_code.add(rand_cw)
                cnt[ball_indices_np[rand_cw]] += 1
                no_improve = 0

        if best_size <= 160:
            break

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
