import time
import numpy as np
import random
import math

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin under 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base for index <-> word
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as index lists and boolean coverage matrix
    ball_indices = [None] * total
    cov = np.zeros((total, total), dtype=np.bool_)

    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q

        ball = [c]
        cov[c, c] = True

        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                ball.append(w)
                cov[c, w] = True

        ball_indices[c] = ball

    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            # Vectorized gain computation
            gains = cov @ (1 - covered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = random.choice(candidates.tolist())
            else:
                best_c = int(candidates[0])

            code.append(best_c)
            covered[cov[best_c]] = 1
            uncovered = int(np.sum(covered == 0))

        return code

    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        in_code = np.zeros(total, dtype=np.bool_)
        for c in code_set:
            in_code[c] = True
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    if np.min(cnt[cov[c]]) >= 2:
                        code_set.remove(c)
                        in_code[c] = False
                        cnt[cov[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        no_success = 0
        max_streak = 1500
        temperature = 0.05  # SA-style acceptance probability for neutral/worsening moves

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # 1. Try deletion
            deleted = False
            for c in list(code_set):
                if np.min(cnt[cov[c]]) >= 2:
                    code_set.remove(c)
                    in_code[c] = False
                    cnt[cov[c]] -= 1
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break

            if deleted:
                continue

            # 2. Try 1-for-1 swap using critical words
            c_remove = random.choice(list(code_set))
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                # Can delete directly
                code_set.remove(c_remove)
                in_code[c_remove] = False
                cnt[cov[c_remove]] -= 1
                continue

            critical_words = np.where(critical_mask)[0]

            # Find candidates that cover all critical words
            candidates_mask = np.all(cov[:, critical_words], axis=1)
            candidates_mask &= ~in_code
            candidates_mask[c_remove] = False

            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                c_add = int(random.choice(candidate_indices))
                code_set.remove(c_remove)
                code_set.add(c_add)
                in_code[c_remove] = False
                in_code[c_add] = True
                cnt[cov[c_remove]] -= 1
                cnt[cov[c_add]] += 1
                no_success = 0
                try_delete()
            else:
                no_success += 1
                # SA-style: occasionally accept a random swap even if it doesn't cover all criticals
                if no_success > 500 and random.random() < temperature:
                    c_add = random.randint(0, total - 1)
                    if not in_code[c_add] and c_add != c_remove:
                        code_set.remove(c_remove)
                        code_set.add(c_add)
                        in_code[c_remove] = False
                        in_code[c_add] = True
                        cnt[cov[c_remove]] -= 1
                        cnt[cov[c_add]] += 1
                        # Try to recover by deleting redundant codewords
                        try_delete()

            # 3. Perturb if stuck - add a random codeword and try to delete
            if no_success > max_streak:
                rand_c = random.randint(0, total - 1)
                if not in_code[rand_c]:
                    code_set.add(rand_c)
                    in_code[rand_c] = True
                    cnt[cov[rand_c]] += 1
                    try_delete()
                no_success = 0
                # Decay temperature
                temperature *= 0.95

        return best_code, best_size

    def final_polish(code_list, time_budget=3.0):
        """Aggressive final pass with multiple deletion and swap rounds."""
        code_set = set(code_list)
        in_code = np.zeros(total, dtype=np.bool_)
        for c in code_set:
            in_code[c] = True
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        # Multiple rounds of deletion
        for _ in range(10):
            if time.time() - t_start > time_budget:
                break
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    if np.min(cnt[cov[c]]) >= 2:
                        code_set.remove(c)
                        in_code[c] = False
                        cnt[cov[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        # Targeted swaps
        for _ in range(50):
            if time.time() - t_start > time_budget:
                break
            if len(code_set) == 0:
                break

            c_remove = random.choice(list(code_set))
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                code_set.remove(c_remove)
                in_code[c_remove] = False
                cnt[cov[c_remove]] -= 1
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                continue

            critical_words = np.where(critical_mask)[0]
            candidates_mask = np.all(cov[:, critical_words], axis=1)
            candidates_mask &= ~in_code
            candidates_mask[c_remove] = False

            candidate_indices = np.where(candidates_mask)[0]
            if len(candidate_indices) > 0:
                c_add = int(random.choice(candidate_indices))
                code_set.remove(c_remove)
                code_set.add(c_add)
                in_code[c_remove] = False
                in_code[c_add] = True
                cnt[cov[c_remove]] -= 1
                cnt[cov[c_add]] += 1
                # Try delete after swap
                for c2 in list(code_set):
                    if np.min(cnt[cov[c2]]) >= 2:
                        code_set.remove(c2)
                        in_code[c2] = False
                        cnt[cov[c2]] -= 1
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=18.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Final polish on best result
    if time.time() < deadline - 5.0:
        polished_code, polished_size = final_polish(best_code_overall, time_budget=3.0)
        if polished_size < best_size_overall:
            best_size_overall = polished_size
            best_code_overall = polished_code

    # Multi-restart with adaptive time allocation
    seed = 0
    restarts = 0
    while time.time() < deadline - 8.0:
        seed += 1
        restarts += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 8.0 - time.time()
        if time_left < 3.0:
            break
        # Allocate more time to earlier restarts, less to later ones
        tb = min(10.0, max(2.0, time_left / max(1, (10 - restarts) / 2.0)))
        best_code, best_size = local_search(greedy_code, time_budget=tb)

        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

            # Polish the new best
            if time.time() < deadline - 3.0:
                polished_code, polished_size = final_polish(best_code_overall, time_budget=2.0)
                if polished_size < best_size_overall:
                    best_size_overall = polished_size
                    best_code_overall = polished_code

        # Early exit if we're close to optimal
        if best_size_overall <= 180:
            break

    # Final conversion
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
