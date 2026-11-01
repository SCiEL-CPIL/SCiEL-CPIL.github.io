import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin under 120s

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as numpy int32 arrays for fast indexing
    balls = [None] * total
    for i in range(total):
        temp = i
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        ball = [i]
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                ball.append(i + (v - orig) * base[coord])
        balls[i] = np.array(ball, dtype=np.int32)

    # Greedy construction with numpy
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            best_c = -1
            max_new = -1
            # Randomized scan: shuffle order to break symmetry
            idxs = np.arange(total)
            random.shuffle(idxs.tolist())
            idxs = np.array(idxs, dtype=np.int32)

            # Vectorized: compute new coverage for all candidates at once
            # For each candidate c, count how many points in balls[c] are uncovered
            # Use advanced indexing: for each c, balls[c] gives the ball indices
            # We need covered[balls[c]] == 0 count for each c
            # This is tricky to vectorize fully, so we batch it
            batch_size = 256
            for start in range(0, total, batch_size):
                batch = idxs[start:start + batch_size]
                # For each c in batch, get its ball and count uncovered
                for c in batch:
                    b = balls[c]
                    new_cov = np.count_nonzero(covered[b] == 0)
                    if new_cov > max_new:
                        max_new = new_cov
                        best_c = c
                        if max_new == ball_size:
                            break
                if max_new == ball_size:
                    break

            if best_c < 0:
                break
            code.append(best_c)
            covered[balls[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # Local search with delete + 1-swap + 2-swap + 3-swap
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[balls[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    b = balls[c]
                    if np.min(cnt[b]) >= 2:
                        code_set.remove(c)
                        cnt[b] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        no_success = 0
        max_streak = 600

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # Choose move type: 1-swap (50%), 2-swap (30%), 3-swap (20%)
            r = random.random()
            if r < 0.5:
                # 1-for-1 swap
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)

                if c_add in code_set or c_add == c_remove:
                    continue

                b_rem = balls[c_remove]
                b_add = balls[c_add]

                # Check validity: for all x in b_rem \ b_add, cnt[x] >= 2
                mask = np.isin(b_rem, b_add)
                critical = b_rem[~mask]
                valid = np.all(cnt[critical] >= 2) if len(critical) > 0 else True

                if valid:
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[b_rem] -= 1
                    cnt[b_add] += 1
                    no_success = 0
                    try_delete()
                else:
                    no_success += 1
            elif r < 0.8:
                # 2-for-2 swap
                code_list_now = list(code_set)
                if len(code_list_now) < 2:
                    no_success += 1
                    continue

                idx1, idx2 = random.sample(range(len(code_list_now)), 2)
                c_rem1 = code_list_now[idx1]
                c_rem2 = code_list_now[idx2]

                c_add1 = random.randint(0, total - 1)
                c_add2 = random.randint(0, total - 1)

                if c_add1 in code_set or c_add2 in code_set:
                    continue
                if c_add1 == c_rem1 or c_add1 == c_rem2 or c_add2 == c_rem1 or c_add2 == c_rem2:
                    continue
                if c_add1 == c_add2:
                    continue

                b_rem1 = balls[c_rem1]
                b_rem2 = balls[c_rem2]
                b_add1 = balls[c_add1]
                b_add2 = balls[c_add2]

                # Combined removed ball
                combined_rem = np.unique(np.concatenate([b_rem1, b_rem2]))
                # Combined added ball
                combined_add = np.unique(np.concatenate([b_add1, b_add2]))

                # Points only covered by removed codewords (not by added)
                mask = np.isin(combined_rem, combined_add)
                critical = combined_rem[~mask]

                # For critical points, they must be covered by at least one other codeword
                if len(critical) > 0:
                    in_rem1 = np.isin(critical, b_rem1).astype(np.int32)
                    in_rem2 = np.isin(critical, b_rem2).astype(np.int32)
                    required = 1 + in_rem1 + in_rem2
                    valid = np.all(cnt[critical] >= required)
                else:
                    valid = True

                if valid:
                    code_set.discard(c_rem1)
                    code_set.discard(c_rem2)
                    code_set.add(c_add1)
                    code_set.add(c_add2)
                    cnt[b_rem1] -= 1
                    cnt[b_rem2] -= 1
                    cnt[b_add1] += 1
                    cnt[b_add2] += 1
                    no_success = 0
                    try_delete()
                else:
                    no_success += 1
            else:
                # 3-for-3 swap
                code_list_now = list(code_set)
                if len(code_list_now) < 3:
                    no_success += 1
                    continue

                idx1, idx2, idx3 = random.sample(range(len(code_list_now)), 3)
                c_rem1 = code_list_now[idx1]
                c_rem2 = code_list_now[idx2]
                c_rem3 = code_list_now[idx3]

                c_add1 = random.randint(0, total - 1)
                c_add2 = random.randint(0, total - 1)
                c_add3 = random.randint(0, total - 1)

                if c_add1 in code_set or c_add2 in code_set or c_add3 in code_set:
                    continue
                rems = {c_rem1, c_rem2, c_rem3}
                if c_add1 in rems or c_add2 in rems or c_add3 in rems:
                    continue
                if c_add1 == c_add2 or c_add1 == c_add3 or c_add2 == c_add3:
                    continue

                b_rem1 = balls[c_rem1]
                b_rem2 = balls[c_rem2]
                b_rem3 = balls[c_rem3]
                b_add1 = balls[c_add1]
                b_add2 = balls[c_add2]
                b_add3 = balls[c_add3]

                # Combined removed ball
                combined_rem = np.unique(np.concatenate([b_rem1, b_rem2, b_rem3]))
                # Combined added ball
                combined_add = np.unique(np.concatenate([b_add1, b_add2, b_add3]))

                # Points only covered by removed codewords (not by added)
                mask = np.isin(combined_rem, combined_add)
                critical = combined_rem[~mask]

                # For critical points, they must be covered by at least one other codeword
                if len(critical) > 0:
                    in_rem1 = np.isin(critical, b_rem1).astype(np.int32)
                    in_rem2 = np.isin(critical, b_rem2).astype(np.int32)
                    in_rem3 = np.isin(critical, b_rem3).astype(np.int32)
                    required = 1 + in_rem1 + in_rem2 + in_rem3
                    valid = np.all(cnt[critical] >= required)
                else:
                    valid = True

                if valid:
                    code_set.discard(c_rem1)
                    code_set.discard(c_rem2)
                    code_set.discard(c_rem3)
                    code_set.add(c_add1)
                    code_set.add(c_add2)
                    code_set.add(c_add3)
                    cnt[b_rem1] -= 1
                    cnt[b_rem2] -= 1
                    cnt[b_rem3] -= 1
                    cnt[b_add1] += 1
                    cnt[b_add2] += 1
                    cnt[b_add3] += 1
                    no_success = 0
                    try_delete()
                else:
                    no_success += 1

            if no_success > max_streak:
                no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=10.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart
    seed = 0
    while time.time() < deadline - 6.0 and best_size_overall > 158:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 6.0 - time.time()
        tb = min(8.0, max(1.0, time_left / 5.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Convert to list of lists
    result = []
    for idx in best_code_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
