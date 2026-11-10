import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin vs 90s kill

    N = 12
    DEGREES = np.arange(N)
    # Pairs (i,j) with i<=j, used to build square from coefficients
    _pair_i, _pair_j = [], []
    for i in range(N):
        for j in range(i, N):
            _pair_i.append(i)
            _pair_j.append(j)
    PAIR_I = np.array(_pair_i, dtype=np.int64)
    PAIR_J = np.array(_pair_j, dtype=np.int64)
    PAIR_DEG = PAIR_I + PAIR_J
    PAIR_MULT = np.where(PAIR_I == PAIR_J, 1, 2)
    NPAIR = len(PAIR_I)

    def count_nonzero_square(c):
        sq = np.zeros(2 * N - 1, dtype=np.int64)
        np.add.at(sq, PAIR_DEG, PAIR_MULT * c[PAIR_I] * c[PAIR_J])
        return int(np.count_nonzero(sq))

    # Fast incremental: square of (c + delta*e_k) = sq + delta*(2*c_k + 2*cross_k)
    # cross_k[d] = sum_{l != k, l+k=d} c_l
    def fast_square(c, k, delta, sq):
        ck = c[k]
        cross = np.zeros(2 * N - 1, dtype=np.int64)
        other = np.ones(N, dtype=bool)
        other[k] = False
        idx = other.nonzero()[0]
        np.add.at(cross, DEGREES[k] + DEGREES[idx], c[idx])
        new_sq = sq + delta * (2 * ck + 2 * cross)
        return new_sq

    # Collision graph: edge (i,j) if degrees collide (2i or 2j equals i+j)
    def build_graph():
        A = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i + 1, N):
                d = i + j
                if d == 2 * i or d == 2 * j:
                    A[i, j] = A[j, i] = True
        return A

    GRAPH = build_graph()
    DEGREE = GRAPH.sum(axis=1)
    ORDER = np.argsort(DEGREE)  # low degree first for coloring

    # Greedy coloring -> independent-set upper bound
    def color_bound():
        color = np.full(N, -1, dtype=np.int64)
        for i in ORDER:
            used = set()
            for j in range(N):
                if GRAPH[i, j] and color[j] >= 0:
                    used.add(color[j])
            c = 0
            while c in used:
                c += 1
            color[i] = c
        return int(color.max()) + 1

    COLOURS = color_bound()
    LOWER = N - COLOURS  # minimum number of nonzero square terms (collision bound)

    # Branch-and-bound over independent sets of the collision graph.
    # Score of an independent set = number of distinct sums i+j over the set
    # (each distinct sum is a potentially nonzero square term).
    def best_independent_set():
        best = [N, list(range(N))]
        adj = [set(int(j) for j in range(N) if GRAPH[i, j]) for i in range(N)]

        def rec(cand, current, sums):
            if len(current) == N:
                if len(sums) < best[0]:
                    best[0] = len(sums)
                    best[1] = list(current)
                return
            # bound: need at least (N - len(current)) more vertices, each adds >=1 new sum
            if len(sums) + (N - len(current)) >= best[0]:
                return
            if not cand:
                return
            # branch on a candidate
            v = int(cand[0])
            # include v
            new_sums = set(sums)
            for u in current:
                new_sums.add(v + u)
            new_sums.add(2 * v)
            new_cand = cand[1:]
            new_cand = [x for x in new_cand if not GRAPH[v, x]]
            rec(new_cand, current + [v], new_sums)
            # exclude v
            rec(cand[1:], current, sums)

        rec(list(range(N)), [], set())
        return best[1], best[0]

    try:
        IS_SUPPORT, IS_SCORE = best_independent_set()
    except Exception:
        IS_SUPPORT, IS_SCORE = list(range(N)), 2 * N - 1

    # ---- Coefficient search machinery ----
    def make_initial(signs):
        c = np.zeros(N, dtype=np.int64)
        for i in IS_SUPPORT:
            c[i] = signs[i % len(signs)]
        return c

    def local_search(c, max_iters):
        c = c.copy()
        score = count_nonzero_square(c)
        sq = None
        T = 4.0
        Tmin = 0.05
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            k = random.randrange(N)
            old = int(c[k])
            r = random.random()
            if r < 0.7:
                new = random.choice([-3, -2, -1, 1, 2, 3])
            else:
                new = random.randint(-64, 64)
            if new == 0:
                new = 1
            if new == old:
                continue
            delta = new - old
            if sq is None:
                sq = np.zeros(2 * N - 1, dtype=np.int64)
                np.add.at(sq, PAIR_DEG, PAIR_MULT * c[PAIR_I] * c[PAIR_J])
            new_sq = fast_square(c, k, delta, sq)
            new_score = int(np.count_nonzero(new_sq))
            d = new_score - score
            if d < 0 or random.random() < np.exp(-d / T):
                c[k] = new
                score = new_score
                sq = new_sq
            else:
                pass
            T = max(Tmin, T * 0.999)
        return c, score

    def targeted_cancel(c):
        """Try to zero specific square terms by adjusting one coefficient."""
        c = c.copy()
        score = count_nonzero_square(c)
        for k in random.sample(range(N), N):
            if time.time() > deadline - 3:
                break
            sq = np.zeros(2 * N - 1, dtype=np.int64)
            np.add.at(sq, PAIR_DEG, PAIR_MULT * c[PAIR_I] * c[PAIR_J])
            for d in range(len(sq)):
                if sq[d] == 0:
                    continue
                # find i such that i + (d-i) = d with i in support, solve for c_i
                for i in IS_SUPPORT:
                    j = d - i
                    if j < 0 or j >= N or j == i:
                        continue
                    if c[j] == 0:
                        continue
                    # sq[d] contains 2*c_i*c_j (i!=j). To zero, need c_i such that
                    # new sq[d]=0: (2*c_i*c_j + rest) = 0 -> c_i = -rest/(2*c_j)
                    rest = int(sq[d]) - 2 * int(c[i]) * int(c[j])
                    denom = 2 * int(c[j])
                    if denom == 0:
                        continue
                    if rest % denom == 0:
                        new = -rest // denom
                        if new != 0 and -64 <= new <= 64 and new != int(c[i]):
                            c[i] = new
                            ns = count_nonzero_square(c)
                            if ns < score:
                                score = ns
                                sq = np.zeros(2 * N - 1, dtype=np.int64)
                                np.add.at(sq, PAIR_DEG, PAIR_MULT * c[PAIR_I] * c[PAIR_J])
                                break
        return c, score

    best_score = float('inf')
    best_c = None

    # Structured initial candidates
    for signs in ([1] * N, [-1, 1] * (N // 2), [1, -1] * (N // 2),
                  [1, 1, -1, -1] * (N // 4), [1, -1, -1, 1] * (N // 4)):
        c = make_initial(signs)
        s = count_nonzero_square(c)
        if s < best_score:
            best_score = s
            best_c = c.copy()

    # Random restarts with simulated annealing
    random.seed(12345)
    n_restarts = 60
    for r in range(n_restarts):
        if time.time() > deadline - 8:
            break
        signs = [random.choice([-1, 1]) for _ in range(N)]
        c = make_initial(signs)
        c, s = local_search(c, max_iters=3000)
        if s < best_score:
            best_score = s
            best_c = c.copy()

    # Focused refinement on best
    if best_c is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            c, s = local_search(best_c, max_iters=4000)
            if s < best_score:
                best_score = s
                best_c = c.copy()
            c, s = targeted_cancel(c if c is not None else best_c)
            if s < best_score:
                best_score = s
                best_c = c.copy()

    # ---- Build final polynomial (ascending degree, length<=41) ----
    if best_c is None:
        best_c = np.ones(N, dtype=np.int64)

    # Ensure exactly 12 nonzero, coefficients in [-64,64], last nonzero
    final = np.zeros(N, dtype=np.int64)
    nonzero_idx = [i for i in range(N) if best_c[i] != 0]
    if len(nonzero_idx) != N:
        # fill missing with 1
        for i in range(N):
            if best_c[i] == 0:
                best_c[i] = 1
    for i in range(N):
        v = int(best_c[i])
        if v == 0:
            v = 1
        if v > 64:
            v = 64
        if v < -64:
            v = -64
        final[i] = v
    # ensure last nonzero
    if final[-1] == 0:
        final[-1] = 1

    return final.tolist()
# EVOLVE-BLOCK-END
