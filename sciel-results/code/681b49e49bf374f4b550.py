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

            for c in idxs:
                b = balls[c]
                # Count uncovered points in this ball
                new_cov = np.count_nonzero(covered[b] == 0)
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break

            if best_c < 0:
                break
            code.append(best_c)
            covered[balls[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # Local search with delete + swap
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
        max_streak = 800

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # Alternate between 1-for-1, 2-for-1, and 1-for-2 swaps
            move_type = random.randint(0, 2)
            
            valid_move = False
            
            if move_type == 0:
                # 1-for-1 swap
                if len(code_set) < 2:
                    continue
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)

                if c_add in code_set or c_add == c_remove:
                    continue

                b_rem = balls[c_remove]
                b_add = balls[c_add]

                # Check validity: for all x in b_rem \ b_add, cnt[x] >= 2
                valid = True
                add_set = set(b_add.tolist())
                for x in b_rem:
                    if x not in add_set:
                        if cnt[x] < 2:
                            valid = False
                            break

                if valid:
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[b_rem] -= 1
                    cnt[b_add] += 1
                    valid_move = True

            elif move_type == 1:
                # 2-for-1 swap: remove 2, add 1
                if len(code_set) < 3:
                    continue
                rem_list = list(code_set)
                c1, c2 = random.sample(rem_list, 2)
                c_add = random.randint(0, total - 1)

                if c_add in code_set:
                    continue

                b1 = balls[c1]
                b2 = balls[c2]
                b_add = balls[c_add]
                
                # We need to check if (b1 U b2) \ b_add is covered by other codewords
                # i.e., for all x in (b1 U b2) \ b_add, cnt[x] >= 2 (since c1, c2 are removed)
                # Actually, if x is in b1 and not b_add, it must be covered by another codeword in code_set \ {c1, c2}
                # So cnt[x] - (1 if x in b1 else 0) - (1 if x in b2 else 0) >= 1?
                # No, cnt[x] includes c1 and c2 if they cover x.
                # If we remove c1 and c2, the new count is cnt[x] - [x in b1] - [x in b2].
                # We need new_count >= 1 for all x in (b1 U b2) \ b_add?
                # Wait, the union of balls of remaining codewords must cover all points.
                # Points in b_add are covered by c_add.
                # Points in (b1 U b2) \ b_add must be covered by other codewords.
                # So for x in (b1 U b2) \ b_add, we need (cnt[x] - [x in b1] - [x in b2]) >= 1.
                
                # Optimization: Only check points in b1 U b2.
                # We can iterate over b1 and b2.
                
                valid = True
                # Check b1
                for x in b1:
                    if x in set(b_add.tolist()):
                        continue
                    # x is in b1, so cnt[x] includes c1.
                    # If x is also in b2, cnt[x] includes c2.
                    # New count = cnt[x] - 1 - (1 if x in b2 else 0)
                    # We need New count >= 1 => cnt[x] - 1 - (1 if x in b2 else 0) >= 1 => cnt[x] >= 2 + (1 if x in b2 else 0)
                    if x in b2:
                        if cnt[x] < 3:
                            valid = False
                            break
                    else:
                        if cnt[x] < 2:
                            valid = False
                            break
                if valid:
                    # Check b2 (points not in b1 are already covered? No, we need to check points in b2 not in b1 and not in b_add)
                    # Actually, if x is in b2 and not b1, we checked it in the b1 loop? No.
                    # We need to check x in b2 \ b1 \ b_add.
                    b1_set = set(b1.tolist())
                    for x in b2:
                        if x in b1_set:
                            continue
                        if x in set(b_add.tolist()):
                            continue
                        # x is in b2, not b1.
                        # New count = cnt[x] - 1 (for c2)
                        # We need New count >= 1 => cnt[x] >= 2
                        if cnt[x] < 2:
                            valid = False
                            break

                if valid:
                    code_set.remove(c1)
                    code_set.remove(c2)
                    code_set.add(c_add)
                    cnt[b1] -= 1
                    cnt[b2] -= 1
                    cnt[b_add] += 1
                    valid_move = True

            else:
                # 1-for-2 swap: remove 1, add 2
                if len(code_set) < 1:
                    continue
                c_remove = random.choice(list(code_set))
                c_add1 = random.randint(0, total - 1)
                c_add2 = random.randint(0, total - 1)
                
                # Ensure distinct and not in code_set
                if c_add1 == c_add2 or c_add1 in code_set or c_add2 in code_set or c_add1 == c_remove or c_add2 == c_remove:
                    continue

                b_rem = balls[c_remove]
                b_add1 = balls[c_add1]
                b_add2 = balls[c_add2]
                
                # Points in b_rem \ (b_add1 U b_add2) must be covered by other codewords.
                # So for x in b_rem \ (b_add1 U b_add2), cnt[x] - 1 >= 1 => cnt[x] >= 2.
                
                valid = True
                add_set = set(b_add1.tolist()) | set(b_add2.tolist())
                for x in b_rem:
                    if x not in add_set:
                        if cnt[x] < 2:
                            valid = False
                            break

                if valid:
                    code_set.remove(c_remove)
                    code_set.add(c_add1)
                    code_set.add(c_add2)
                    cnt[b_rem] -= 1
                    cnt[b_add1] += 1
                    cnt[b_add2] += 1
                    valid_move = True

            if valid_move:
                no_success = 0
                try_delete()
            else:
                no_success += 1
                if no_success > max_streak:
                    # Reset streak, keep going
                    no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=12.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart
    seed = 0
    while time.time() < deadline - 8.0 and best_size_overall > 158:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 8.0 - time.time()
        tb = min(10.0, max(1.5, time_left / 4.0))
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
