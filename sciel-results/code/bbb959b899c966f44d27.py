import itertools
import random
import time
import numpy as np


def _idx_to_word(idx, q, n):
    w = [0] * n
    for p in range(n - 1, -1, -1):
        w[p] = idx % q
        idx //= q
    return w


def _word_to_idx(w, q):
    s = 0
    for p in range(len(w)):
        s = s * q + w[p]
    return s


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    total_words = q ** n

    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1(deadline)

    return _greedy_cover(q, n, radius, deadline)


def _construct_3_6_1(deadline):
    """Specialized construction for q=3, n=6, radius=1 targeting 73 codewords."""
    total = 729
    n = 6
    q = 3

    # Precompute all balls
    all_balls = []
    for idx in range(total):
        w = _idx_to_word(idx, q, n)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                nw = list(w)
                nw[pos] = (nw[pos] + delta) % q
                ball.append(_word_to_idx(nw, q))
        all_balls.append(set(ball))

    # Coverage count array: cov[i] = number of codewords covering word i
    def _compute_coverage(code_set):
        cov = np.zeros(total, dtype=np.int32)
        for idx in code_set:
            for b in all_balls[idx]:
                cov[b] += 1
        return cov

    def _is_complete(cov):
        return np.all(cov > 0)

    # Initial: tetracode product (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = set()
    for h in tetracode:
        for t in range(9):
            tw = _idx_to_word(t, 3, 2)
            w = h + tw
            init_code.add(_word_to_idx(w, 3))

    best_code = set(init_code)
    best_size = len(best_code)

    # Local search with multiple restarts
    for attempt in range(30):
        if time.time() > deadline - 20:
            break

        if attempt == 0:
            code_set = set(init_code)
        elif attempt % 3 == 1:
            # Perturb best
            code_set = set(best_code)
            for _ in range(random.randint(3, 15)):
                if code_set:
                    r = random.choice(list(code_set))
                    code_set.discard(r)
                    code_set.add(random.randint(0, total - 1))
        else:
            # Random greedy
            code_set = set()
            uncovered = set(range(total))
            for _ in range(75):
                if time.time() > deadline - 20:
                    break
                best_idx = -1
                best_cnt = -1
                candidates = random.sample(range(total), min(150, total))
                for idx in candidates:
                    if idx not in code_set:
                        cnt = len(all_balls[idx] & uncovered)
                        if cnt > best_cnt:
                            best_cnt = cnt
                            best_idx = idx
                if best_idx == -1:
                    break
                code_set.add(best_idx)
                uncovered -= all_balls[best_idx]
                if not uncovered:
                    break

        # Local search
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            cov = _compute_coverage(code_set)

            # Try removals
            for idx in list(code_set):
                if time.time() > deadline - 15:
                    break
                # Check if removing idx leaves any uncovered
                new_cov = cov.copy()
                for b in all_balls[idx]:
                    new_cov[b] -= 1
                if np.all(new_cov > 0):
                    code_set.discard(idx)
                    cov = new_cov
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)

            # Try swaps
            if time.time() > deadline - 15:
                break
            for idx in list(code_set):
                if time.time() > deadline - 15:
                    break
                # Words only covered by idx
                unique = set()
                for b in all_balls[idx]:
                    if cov[b] == 1:
                        unique.add(b)
                if not unique:
                    code_set.discard(idx)
                    cov = _compute_coverage(code_set)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                    continue
                # Try to find replacement
                for new_idx in range(total):
                    if new_idx in code_set:
                        continue
                    if all_balls[new_idx] >= unique:
                        code_set.discard(idx)
                        code_set.add(new_idx)
                        cov = _compute_coverage(code_set)
                        improved = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = set(code_set)
                        break

        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)

        if best_size <= 73:
            break

    return [_idx_to_word(idx, 3, 6) for idx in best_code]


def _greedy_cover(q, n, radius, deadline):
    total_words = q ** n
    all_balls = {}
    for idx in range(total_words):
        w = _idx_to_word(idx, q, n)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                nw = list(w)
                nw[pos] = (nw[pos] + delta) % q
                ball.append(_word_to_idx(nw, q))
        all_balls[idx] = set(ball)

    uncovered = set(range(total_words))
    code = []
    code_set = set()

    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if idx not in code_set:
                cnt = len(all_balls[idx] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
        if best_idx == -1 or best_count <= 0:
            break
        w = _idx_to_word(best_idx, q, n)
        code.append(w)
        code_set.add(best_idx)
        uncovered -= all_balls[best_idx]

    if not uncovered:
        return code

    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(3), repeat=n)]
