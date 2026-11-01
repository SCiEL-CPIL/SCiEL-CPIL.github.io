"""Suite task: ternary covering code K_3(6,1). Specialized fast greedy + local search."""
import time
import itertools
import numpy as np

_TIME_LIMIT = 140.0
_START = time.time()
_cache = {}


def construct(q, n, radius):
    key = (q, n, radius)
    if key in _cache:
        return [list(w) for w in _cache[key]]
    if (q, n, radius) == (3, 6, 1):
        res = _solve_q3n6r1()
    else:
        res = _generic(q, n, radius)
    _cache[key] = res
    return [list(w) for w in res]


def _generic(q, n, radius):
    # Fallback: direct product of a radius-1 q-ary code of length 4 with the full tail.
    # For q=3, length 4, radius 1 the Hamming code has 9 words; general q uses a
    # 2*(q-1)*4 + 1 = 8q-7 length-4 code (parity on first 4 coords).
    if n >= 4:
        base = [[a, b, c, d, (a + b + c + d) % q] for a in range(q)
                for b in range(q) for c in range(q) for d in range(q)]
        base = base[: 2 * (q - 1) * 4 + 1]
        return [h[:4] + list(tail) for h in base
                for tail in itertools.product(range(q), repeat=n - 4)]
    return [list(t) for t in itertools.product(range(q), repeat=n)]


def _solve_q3n6r1():
    q, n = 3, 6
    N = q ** n
    # Precompute the 19-ball (self + 18 single-coordinate neighbours) as flat indices.
    ball = [0]
    for i in range(n):
        for d in range(1, q):
            ball.append(d * (q ** i))
    ball = np.array(ball, dtype=np.int64)

    # Base-3 digits of each flat index, shape (N, n).
    digits = np.zeros((N, n), dtype=np.int64)
    for i in range(n):
        digits[:, i] = (np.arange(N, dtype=np.int64) // (q ** i)) % q

    # Coverage: cov[j] = number of selected codewords within the ball of j.
    # A word j is covered iff cov[j] >= 1.
    def add(cov, c):
        idx = c + ball
        cov[idx] += 1

    def remove(cov, c):
        idx = c + ball
        cov[idx] -= 1

    def uncovered(cov):
        return int(np.count_nonzero(cov == 0))

    def greedy():
        cov = np.zeros(N, dtype=np.int32)
        selected = []
        for _ in range(75):
            if time.time() - _START > _TIME_LIMIT:
                break
            # gain = number of currently-uncovered words in this candidate's ball
            gain = np.count_nonzero(cov[ball] == 0)
            best = int(np.argmax(gain))
            if gain[best] <= 0:
                break
            add(cov, best)
            selected.append(best)
        return selected, cov

    def local_search(selected, cov, budget):
        selected = list(selected)
        while time.time() - _START < budget:
            improved = False

            # 1) Remove any redundant codeword (cov stays >= 1 everywhere).
            for c in selected:
                remove(cov, c)
                if np.all(cov >= 1):
                    selected.remove(c)
                    improved = True
                else:
                    add(cov, c)
                if improved:
                    break
            if improved:
                continue

            # 2) Add a codeword that reduces the uncovered count.
            uc = np.count_nonzero(cov == 0)
            if uc == 0:
                break
            gain = np.count_nonzero(cov[ball] == 0)
            best = int(np.argmax(gain))
            if gain[best] > 0:
                add(cov, best)
                selected.append(best)
                improved = True
            if improved:
                continue

            # 3) 1-for-1 swap: drop c, add x, if it reduces uncovered count.
            done = False
            for c in selected:
                if time.time() - _START > budget:
                    break
                remove(cov, c)
                # new uncovered = words in ball(c) that are now 0
                before = np.count_nonzero(cov == 0)
                # candidate x must cover at least one of those newly-uncovered words
                cand = np.where(cov[ball] == 0)[0]
                if cand.size == 0:
                    add(cov, c)
                    continue
                # pick the x that covers the most currently-uncovered words
                # (cov currently has c removed)
                g = np.count_nonzero(cov[ball] == 0)
                x = int(np.argmax(g))
                if g[x] > 0:
                    # after adding x, net uncovered change = -g[x] + (cov[x+ball] that were 1 and become 2 don't matter)
                    # just apply and check
                    add(cov, x)
                    after = np.count_nonzero(cov == 0)
                    if after < before:
                        # commit the swap
                        selected.remove(c)
                        selected.append(x)
                        done = True
                        break
                    else:
                        remove(cov, x)
                add(cov, c)
            if done:
                continue

            # 4) No improving single move: restart with a fresh greedy seed.
            sel2, cov2 = greedy()
            if len(sel2) < len(selected):
                selected, cov = sel2, cov2
            else:
                # small random perturbation then re-local-search
                if selected and np.random.rand() < 0.5:
                    c = selected[int(np.random.randint(len(selected)))]
                    remove(cov, c)
                    selected.remove(c)
                # try to repair
                for _ in range(50):
                    uc = np.count_nonzero(cov == 0)
                    if uc == 0:
                        break
                    gain = np.count_nonzero(cov[ball] == 0)
                    best = int(np.argmax(gain))
                    if gain[best] <= 0:
                        break
                    add(cov, best)
                    selected.append(best)
                # reduce again
                for c in list(selected):
                    remove(cov, c)
                    if np.all(cov >= 1):
                        selected.remove(c)
                    else:
                        add(cov, c)
        return selected

    budget = _START + _TIME_LIMIT - 5.0
    best = None
    best_cov = None
    # First a deterministic greedy to get a solid seed.
    sel, cov = greedy()
    sel = local_search(sel, cov, budget)
    best, best_cov = list(sel), cov.copy()

    # Try a couple of randomized restarts if time remains.
    rng = np.random.default_rng(0)
    while time.time() - _START < budget - 2.0:
        # randomized greedy: among top-k gain, pick randomly
        cov = np.zeros(N, dtype=np.int32)
        selected = []
        for _ in range(75):
            if time.time() - _START > budget:
                break
            gain = np.count_nonzero(cov[ball] == 0)
            gmax = gain.max()
            if gmax <= 0:
                break
            top = np.where(gain == gmax)[0]
            c = int(top[int(rng.integers(len(top)))])
            add(cov, c)
            selected.append(c)
        selected = local_search(selected, cov, budget)
        if len(selected) < len(best):
            best, best_cov = list(selected), cov.copy()
        if len(best) <= 73:
            break

    # Final polish pass on the best: a tight reduce.
    cov = np.zeros(N, dtype=np.int32)
    for c in best:
        add(cov, c)
    for c in list(best):
        remove(cov, c)
        if np.all(cov >= 1):
            best.remove(c)
        else:
            add(cov, c)
    return best


if __name__ == "__main__":
    res = construct(3, 6, 1)
    print(len(res))
