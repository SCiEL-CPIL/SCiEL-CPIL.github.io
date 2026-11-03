import time
import itertools
import numpy as np
import random

_DEADLINE = 145.0  # Safety margin vs 150s wall-clock kill

def _all_words(q, n):
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)

def construct(q, n, radius):
    t0 = time.time()
    total = q ** n

    if total > 10_000_000:
        if q == 3 and n >= 4:
            tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tet for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(range(n))]

    base = _all_words(q, n)
    powers = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    idx_of = (base * powers).sum(axis=1)

    if radius == 1:
        ball = np.zeros((total, 1 + n * (q - 1)), dtype=np.int64)
        ball[:, 0] = idx_of
        k = 1
        for pos in range(n):
            for delta in range(1, q):
                flipped = base.copy()
                flipped[:, pos] = (base[:, pos] + delta) % q
                ball[:, k] = (flipped * powers).sum(axis=1)
                k += 1
        balls = ball
    else:
        balls_list = []
        for i in range(total):
            seen = {i}
            frontier = [i]
            for _ in range(radius):
                nxt = []
                for widx in frontier:
                    w = base[widx]
                    for pos in range(n):
                        for v in range(q):
                            if v == w[pos]:
                                continue
                            nw = w.copy()
                            nw[pos] = v
                            j = int((nw * powers).sum())
                            if j not in seen:
                                seen.add(j)
                                nxt.append(j)
                frontier = nxt
            balls_list.append(list(seen))
        maxb = max(len(b) for b in balls_list)
        balls = np.array([[x for x in b] + [0] * (maxb - len(b)) for b in balls_list], dtype=np.int64)

    ball_size = int(balls[0].size)

    def time_left():
        return _DEADLINE - (time.time() - t0)

    # Precompute ball membership matrix
    ball_member = np.zeros((total, total), dtype=np.int8)
    for i in range(total):
        ball_member[i, balls[i]] = 1

    # Reverse cover: for each word j, list of codewords i whose ball covers j
    reverse_cover = []
    for j in range(total):
        idxs = np.where(ball_member[:, j] == 1)[0]
        reverse_cover.append(idxs)

    # Greedy Set Cover
    cov = np.zeros(total, dtype=np.int32)
    chosen = []
    in_code = np.zeros(total, dtype=bool)

    while time_left() > 15.0:
        uncovered = (cov == 0).astype(np.int32)
        gains = ball_member @ uncovered
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        chosen.append(best)
        in_code[best] = True
        cov[balls[best]] += 1

    # Local search state
    covcnt = np.zeros(total, dtype=np.int32)
    for i in chosen:
        covcnt[balls[i]] += 1

    def try_remove(i):
        b = balls[i]
        return (covcnt[b] >= 2).all()

    def do_remove(i):
        b = balls[i]
        covcnt[b] -= 1
        in_code[i] = False
        for idx, v in enumerate(chosen):
            if v == i:
                chosen[idx] = chosen[-1]
                chosen.pop()
                break

    def do_add(j):
        b = balls[j]
        covcnt[b] += 1
        in_code[j] = True
        chosen.append(j)

    def try_swap(i, j):
        b_i = balls[i]
        b_j = balls[j]
        in_j = np.zeros(total, dtype=bool)
        in_j[b_j] = True
        critical = b_i[~in_j[b_i]]
        if critical.size == 0:
            return True
        return (covcnt[critical] >= 2).all()

    # Local search
    improved = True
    while improved and time_left() > 5.0:
        improved = False

        # Pass A: Redundancy Removal
        for idx in range(len(chosen)):
            if time_left() < 5.0:
                break
            i = chosen[idx]
            if try_remove(i):
                do_remove(i)
                improved = True
                break

        if improved:
            continue

        # Pass B: 1-for-1 Swap
        if time_left() > 10.0:
            n_chosen = len(chosen)
            chosen_indices = list(range(n_chosen))
            random.shuffle(chosen_indices)

            for idx in chosen_indices:
                if time_left() < 10.0:
                    break
                i = chosen[idx]
                b_i = balls[i]
                unique_words = b_i[covcnt[b_i] == 1]

                found = False
                for _ in range(30):
                    j = random.randint(0, total - 1)
                    if in_code[j]:
                        continue
                    if try_swap(i, j):
                        do_remove(i)
                        do_add(j)
                        improved = True
                        found = True
                        break

                if not found and unique_words.size > 0:
                    candidates = set()
                    for w in unique_words:
                        for c in reverse_cover[int(w)]:
                            if not in_code[c] and c != i:
                                candidates.add(int(c))
                    candidates = list(candidates)
                    random.shuffle(candidates)
                    for j in candidates:
                        if time_left() < 10.0:
                            break
                        if try_swap(i, j):
                            do_remove(i)
                            do_add(j)
                            improved = True
                            found = True
                            break

                if not found and n_chosen < 300:
                    for j in range(total):
                        if time_left() < 10.0:
                            break
                        if in_code[j]:
                            continue
                        if try_swap(i, j):
                            do_remove(i)
                            do_add(j)
                            improved = True
                            found = True
                            break
                    if found:
                        break

        if improved:
            continue

        # Pass C: 2-for-1 Exchange
        if time_left() > 30.0:
            n_chosen = len(chosen)
            pairs_to_check = []
            if n_chosen < 200:
                for a in range(n_chosen):
                    for b in range(a + 1, n_chosen):
                        pairs_to_check.append((chosen[a], chosen[b]))
            else:
                for _ in range(2000):
                    a = random.randint(0, n_chosen - 1)
                    b = random.randint(0, n_chosen - 1)
                    if a != b:
                        pairs_to_check.append((chosen[a], chosen[b]))

            random.shuffle(pairs_to_check)

            for i, k in pairs_to_check:
                if time_left() < 30.0:
                    break
                if i not in chosen or k not in chosen:
                    continue

                b_i = balls[i]
                b_k = balls[k]

                in_i = np.zeros(total, dtype=bool)
                in_i[b_i] = True
                in_k = np.zeros(total, dtype=bool)
                in_k[b_k] = True

                lost1 = in_i & in_k & (covcnt == 2)
                lost2 = (in_i ^ in_k) & (covcnt == 1)
                lost = lost1 | lost2

                if not lost.any():
                    do_remove(i)
                    do_remove(k)
                    improved = True
                    break

                lost_words = np.where(lost)[0]
                if len(lost_words) > 50:
                    continue

                found_j = -1
                candidates = set(reverse_cover[int(lost_words[0])])
                for w in lost_words[1:]:
                    candidates &= set(reverse_cover[int(w)])
                    if not candidates:
                        break
                candidates = [c for c in candidates if not in_code[c] and c != i and c != k]
                random.shuffle(candidates)
                for j in candidates:
                    if time_left() < 30.0:
                        break
                    b_j = balls[j]
                    in_j = np.zeros(total, dtype=bool)
                    in_j[b_j] = True
                    if in_j[lost].all():
                        found_j = j
                        break

                if found_j != -1:
                    do_remove(i)
                    do_remove(k)
                    do_add(found_j)
                    improved = True
                    break

    # Final verification and repair
    if time_left() > 1.0:
        covered = np.zeros(total, dtype=bool)
        for i in chosen:
            covered[balls[i]] = True
        if not covered.all():
            while time_left() > 0.5:
                g = (~covered)
                if not g.any():
                    break
                best = -1
                bestg = 0
                for i in range(total):
                    if in_code[i]:
                        continue
                    ng = g[balls[i]].sum()
                    if ng > bestg:
                        bestg = ng
                        best = i
                if best < 0 or bestg == 0:
                    break
                do_add(best)
                covered[balls[best]] = True

    return [base[i].tolist() for i in chosen]
