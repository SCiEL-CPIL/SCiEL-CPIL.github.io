"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0
_START = [time.time()]

def _elapsed():
    return time.time() - _START[0]

def _time_left():
    return _DEADLINE - _elapsed()

def construct(q, n, radius):
    _START[0] = time.time()
    if (q, n, radius) != (3, 8, 1):
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    rng = random.Random(12345)
    N = 3 ** 8  # 6561

    # Precompute digits and idx conversions
    def idx_to_digits(i):
        d = [0] * 8
        for p in range(8):
            d[p] = i % 3
            i //= 3
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(8):
            v += d[p] * mul
            mul *= 3
        return v

    # Precompute balls as a 2D array: ball_matrix[i] contains the 17 indices in ball of i
    # Ball size = 1 + 8*2 = 17
    BALL_SIZE = 17
    ball_matrix = np.zeros((N, BALL_SIZE), dtype=np.int32)
    for i in range(N):
        d = idx_to_digits(i)
        ball_matrix[i, 0] = i
        idx = 1
        for p in range(8):
            orig = d[p]
            for nv in range(3):
                if nv != orig:
                    d[p] = nv
                    ball_matrix[i, idx] = digits_to_idx(d)
                    idx += 1
                    d[p] = orig

    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_set = []
    for h in tetracode:
        for t in tetracode:
            w = h + t
            seed_set.append(digits_to_idx(w))
    seed_set = list(set(seed_set))

    # Greedy fill with vectorized gain computation
    covered = np.zeros(N, dtype=np.int8)
    chosen = list(seed_set)
    for i in seed_set:
        covered[ball_matrix[i]] = 1

    def greedy_fill(chosen, covered, time_budget=30.0):
        t_start = time.time()
        while time.time() - t_start < time_budget and _time_left() > 25.0:
            # Vectorized: for each candidate i, gain = sum of (1 - covered[j]) for j in ball_matrix[i]
            # = BALL_SIZE - sum(covered[ball_matrix[i]])
            gains = BALL_SIZE - covered[ball_matrix].sum(axis=1)
            best_i = int(np.argmax(gains))
            if gains[best_i] <= 0:
                break
            chosen.append(best_i)
            covered[ball_matrix[best_i]] = 1
        return chosen, covered

    chosen, covered = greedy_fill(chosen, covered)

    if not covered.all():
        for i in range(N):
            if not covered[i]:
                chosen.append(i)
                covered[ball_matrix[i]] = 1

    # Build coverage count
    covcount = np.zeros(N, dtype=np.int32)
    for i in chosen:
        covcount[ball_matrix[i]] += 1

    # Local search: removals
    chosen_set = set(chosen)
    def try_removals(chosen_set, covcount, time_budget=20.0):
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_budget and _time_left() > 15.0:
            improved = False
            order = list(chosen_set)
            rng.shuffle(order)
            for i in order:
                if i not in chosen_set:
                    continue
                if np.min(covcount[ball_matrix[i]]) >= 2:
                    covcount[ball_matrix[i]] -= 1
                    chosen_set.discard(i)
                    improved = True
        return chosen_set, covcount

    chosen_set, covcount = try_removals(chosen_set, covcount)

    # Simulated annealing for swaps
    def simulated_annealing(chosen_set, covcount, time_budget=40.0):
        t_start = time.time()
        T = 2.0
        cooling = 0.9999
        best_set = set(chosen_set)
        best_size = len(chosen_set)
        cur_set = set(chosen_set)
        cur_size = len(cur_set)

        while time.time() - t_start < time_budget and _time_left() > 10.0:
            T *= cooling
            if T < 0.01:
                break

            # Pick a random codeword to remove
            if not cur_set:
                break
            i = rng.choice(list(cur_set))
            b = ball_matrix[i]
            would_uncover = b[covcount[b] == 1]

            if len(would_uncover) == 0:
                # Pure removal
                covcount[b] -= 1
                cur_set.discard(i)
                cur_size -= 1
                if cur_size < best_size:
                    best_size = cur_size
                    best_set = set(cur_set)
                continue

            # Try to find a replacement j
            # Sample a few candidates
            target = set(would_uncover.tolist())
            found = -1
            # Try random candidates first
            for _ in range(20):
                j = rng.randint(0, N - 1)
                if j in cur_set:
                    continue
                bj = ball_matrix[j]
                if target.issubset(set(bj.tolist())):
                    found = j
                    break

            if found == -1:
                # Try a few more targeted
                for _ in range(10):
                    j = rng.randint(0, N - 1)
                    if j in cur_set:
                        continue
                    bj = ball_matrix[j]
                    if target.issubset(set(bj.tolist())):
                        found = j
                        break

            if found >= 0:
                bj = ball_matrix[found]
                # Compute delta potential: change in number of words with covcount==1
                old_ones = np.sum(covcount == 1)
                covcount[b] -= 1
                covcount[bj] += 1
                new_ones = np.sum(covcount == 1)
                delta = new_ones - old_ones

                if delta < 0 or rng.random() < np.exp(-delta / T):
                    cur_set.discard(i)
                    cur_set.add(found)
                    if len(cur_set) < best_size:
                        best_size = len(cur_set)
                        best_set = set(cur_set)
                else:
                    # Revert
                    covcount[bj] -= 1
                    covcount[b] += 1

        # Restore best
        cur_set = best_set
        return cur_set, covcount

    chosen_set, covcount = simulated_annealing(chosen_set, covcount)

    # Final removals pass
    chosen_set, covcount = try_removals(chosen_set, covcount, time_budget=10.0)

    # Final verification
    final_covered = np.zeros(N, dtype=np.int8)
    for i in chosen_set:
        final_covered[ball_matrix[i]] = 1
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen_set.add(i)

    chosen = list(chosen_set)
    return [idx_to_digits(i) for i in chosen]
