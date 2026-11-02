"""Covering code K_3(7,1): vectorized ball-mask representation + simulated annealing."""
import time
import random
import numpy as np

_DEADLINE = 120.0
_t0 = 0.0


def construct(q, n, radius):
    global _t0
    _t0 = time.time()

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1()
    return _construct_general(q, n, radius)


def _construct_3_7_1():
    N = 2187
    deadline = _t0 + _DEADLINE - 3.0  # safety margin

    # Precompute digits for all words
    digits = np.zeros((N, 7), dtype=np.int8)
    for i in range(N):
        tmp = i
        for pos in range(7):
            digits[i, pos] = tmp % 3
            tmp //= 3

    # Precompute pos_val_mask: pos_val_mask[pos][val] = bool array where digit at pos == val
    pos_val_mask = [[None] * 3 for _ in range(7)]
    for pos in range(7):
        for val in range(3):
            pos_val_mask[pos][val] = (digits[:, pos] == val)

    # Precompute ball_mask: ball_mask[i] = bool array, True at j if dist(i,j) <= 1
    # ball_mask[i] = OR over pos of (same as i except pos, any val at pos) ... but that includes i itself
    # Actually: ball(i) = {i} ∪ {j : j differs from i in exactly one position}
    # = union over pos of {j : j[p] = i[p] for all p != pos}  (this includes i when val=i[pos])
    # So ball_mask[i] = OR_{pos} (AND_{p != pos} pos_val_mask[p][i[p]])
    ball_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        row = np.zeros(N, dtype=bool)
        for pos in range(7):
            m = np.ones(N, dtype=bool)
            for p in range(7):
                if p != pos:
                    m = m & pos_val_mask[p][digits[i, p]]
            row |= m
        ball_mask[i] = row

    ball_mask_int = ball_mask.astype(np.int8)

    # Greedy initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        selected_set = set()

        while np.any(uncovered) and time.time() < deadline - 5.0:
            best_idx = -1
            best_count = 0
            # Scan all candidates
            for i in range(N):
                if i in selected_set:
                    continue
                cnt = np.sum(uncovered & ball_mask[i])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_mask[best_idx]

        return selected

    # Simulated annealing on a code
    def anneal(initial_code, seed, time_budget):
        rng = random.Random(seed)
        code = list(initial_code)
        code_set = set(code)

        # Coverage count
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt += ball_mask_int[c]

        best_code = code[:]
        best_size = len(code)
        best_cnt = cnt.copy()

        # Temperature schedule
        T_start = 2.0
        T_end = 0.05
        n_iter_max = 50000
        iter_budget = time_budget

        t_end = time.time() + time_budget
        T = T_start

        for it in range(n_iter_max):
            if time.time() > t_end:
                break
            # Cool temperature
            T = T_start * (T_end / T_start) ** (it / n_iter_max)

            move_type = rng.random()
            if move_type < 0.4:
                # Try remove
                if len(code) == 0:
                    continue
                c = rng.choice(code)
                # Check if removing c leaves all words covered
                new_cnt = cnt - ball_mask_int[c]
                if np.any(new_cnt == 0):
                    # Not redundant, skip
                    continue
                # Accept removal
                cnt = new_cnt
                code.remove(c)
                code_set.discard(c)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cnt = cnt.copy()
            elif move_type < 0.8:
                # Try swap: remove c, add d
                if len(code) < 2:
                    continue
                c = rng.choice(code)
                # Find words uniquely covered by c
                unique_words = np.where((cnt == 1) & ball_mask[c])[0]
                if len(unique_words) == 0:
                    # c is redundant, just remove
                    cnt -= ball_mask_int[c]
                    code.remove(c)
                    code_set.discard(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        best_cnt = cnt.copy()
                    continue
                # Find candidates d that cover all unique_words
                # d must be within dist 1 of each unique_word
                # candidate_mask[d] = AND over w in unique_words of ball_mask[d, w]
                candidate_mask = np.ones(N, dtype=bool)
                for w in unique_words:
                    # d covers w iff ball_mask[d, w] is True, i.e., dist(d,w)<=1
                    # ball_mask is symmetric, so ball_mask[d,w] = ball_mask[w,d]
                    # We need: for all w in unique_words, d is in ball(w)
                    # i.e., d is in intersection of ball(w) for w in unique_words
                    candidate_mask = candidate_mask & ball_mask[w]  # ball_mask[w] is the row for w
                # Exclude already in code
                candidate_mask = candidate_mask & ~np.array([x in code_set for x in range(N)])
                candidates = np.where(candidate_mask)[0]
                if len(candidates) == 0:
                    continue
                d = int(rng.choice(candidates))
                # Apply swap
                cnt = cnt - ball_mask_int[c] + ball_mask_int[d]
                code.remove(c)
                code_set.discard(c)
                code.append(d)
                code_set.add(d)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cnt = cnt.copy()
            else:
                # Try add (to enable future removes)
                # Pick a random word, add a codeword near it
                if len(code) >= 250:
                    continue
                w = rng.randint(0, N - 1)
                # Pick a random neighbor of w (including w itself)
                neighbors = np.where(ball_mask[w])[0]
                d = int(rng.choice(neighbors))
                if d in code_set:
                    continue
                cnt = cnt + ball_mask_int[d]
                code.append(d)
                code_set.add(d)
                # Don't track best here (size increased)

        return best_code, best_size

    # Run multiple annealing restarts
    best_overall = None
    best_overall_size = float('inf')

    # First greedy
    code0 = greedy_init(42)
    size0 = len(code0)
    best_overall = code0
    best_overall_size = size0

    remaining = deadline - time.time()
    n_restarts = max(1, int(remaining / 8.0))
    n_restarts = min(n_restarts, 15)

    for r in range(n_restarts):
        if time.time() > deadline - 2.0:
            break
        budget = (deadline - time.time() - 2.0) / (n_restarts - r)
        seed = 1000 + r * 7
        # Greedy with different seed
        code_init = greedy_init(seed)
        code, size = anneal(code_init, seed + 10000, budget)
        if size < best_overall_size:
            best_overall_size = size
            best_overall = code

    # Final cleanup: remove any redundant
    cnt = np.zeros(N, dtype=np.int32)
    for c in best_overall:
        cnt += ball_mask_int[c]
    changed = True
    while changed:
        changed = False
        new_code = []
        for c in best_overall:
            if np.any((cnt == 1) & ball_mask[c]):
                new_code.append(c)
            else:
                cnt -= ball_mask_int[c]
                changed = True
        best_overall = new_code

    # Convert to word lists
    result = []
    for idx in best_overall:
        w = []
        x = idx
        for _ in range(7):
            w.append(x % 3)
            x //= 3
        w.reverse()
        result.append(w)
    return result


def _construct_general(q, n, radius):
    """General fallback: simple greedy."""
    total = q ** n
    deadline = _t0 + _DEADLINE - 3.0
    uncovered = np.ones(total, dtype=bool)
    selected = []
    selected_set = set()

    def ball_of(idx):
        """Return set of flat indices within distance 1 of idx."""
        out = {idx}
        for pos in range(n):
            stride = q ** (n - 1 - pos)
            d = (idx // stride) % q
            for d2 in range(q):
                if d2 != d:
                    out.add(idx + (d2 - d) * stride)
        return out

    while np.any(uncovered) and time.time() < deadline:
        best = -1
        best_count = 0
        for idx in range(total):
            if idx in selected_set:
                continue
            b = ball_of(idx)
            cnt = sum(1 for x in b if uncovered[x])
            if cnt > best_count:
                best_count = cnt
                best = idx
        if best == -1:
            break
        selected.append(best)
        selected_set.add(best)
        b = ball_of(best)
        for x in b:
            uncovered[x] = False

    result = []
    for idx in selected:
        w = []
        x = idx
        for _ in range(n):
            w.append(x % q)
            x //= q
        w.reverse()
        result.append(w)
    return result
