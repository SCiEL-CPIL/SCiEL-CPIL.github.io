import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = N * N  # 256


def _build_cover():
    """Vectorized build of COVER[p] = list of (a,b) with a<b, a,b!=p,
    such that triangle (p,a,b) is isosceles."""
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    iu_a, iu_b = np.triu_indices(M, k=1)
    cover = []
    for p in range(M):
        d_pa = D2[p, A][:, None]   # M x 1
        d_pb = D2[p, A][None, :]   # 1 x M
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        cover.append(list(zip(a[sel].tolist(), b[sel].tolist())))
    return cover


COVER = _build_cover()


def _make_inS(S):
    inS = [False] * M
    for v in S:
        inS[v] = True
    return inS


def _compatible(v, inS):
    """v can be added to S without creating an isosceles triangle."""
    for a, b in COVER[v]:
        if inS[a] and inS[b]:
            return False
    return True


def _is_maximal(inS):
    """S is maximal iff every outside point is blocked by some in-S pair."""
    for p in range(M):
        if inS[p]:
            continue
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                break
        else:
            return False
    return True


def _shrink(S):
    """Remove any point whose removal keeps maximality (inclusion-minimal maximal)."""
    S = set(S)
    inS = _make_inS(S)
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            inS[x] = False
            S.discard(x)
            if _is_maximal(inS):
                improved = True
            else:
                inS[x] = True
                S.add(x)
    return S


def _repair(S):
    """Greedy-complete S to a maximal no-isosceles set."""
    S = set(S)
    inS = _make_inS(S)
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if inS[v]:
            continue
        if _compatible(v, inS):
            S.add(v)
            inS[v] = True
    return S


def _greedy_maximal():
    S = set()
    inS = _make_inS(S)
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if _compatible(v, inS):
            S.add(v)
            inS[v] = True
    return S


def _unblocked(inS):
    """Return list of points outside S that are NOT blocked by any in-S pair."""
    out = []
    for p in range(M):
        if inS[p]:
            continue
        blocked = False
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                blocked = True
                break
        if not blocked:
            out.append(p)
    return out


def _energy(S, inS):
    """Objective: small size, penalize non-maximality (unblocked outside points)."""
    return len(S) + 2.0 * len(_unblocked(inS))


def _anneal(S, inS, deadline, t0):
    """Simulated annealing: try flips (add/remove) accepted by Metropolis on _energy."""
    S = set(S)
    inS = _make_inS(S)
    cur_e = _energy(S, inS)
    best = set(S)
    best_e = cur_e
    T = 3.0
    T_min = 0.05
    cool = 0.9985
    while time.time() < deadline:
        if T < T_min:
            break
        # Choose a random flip: pick a random point, toggle it if it keeps validity
        v = random.randrange(M)
        if inS[v]:
            # try removal
            inS[v] = False
            S.discard(v)
            e = _energy(S, inS)
            if e < cur_e or random.random() < np.exp(-(e - cur_e) / T):
                cur_e = e
                if e < best_e:
                    best_e = e
                    best = set(S)
            else:
                inS[v] = True
                S.add(v)
        else:
            # try addition (must keep no-isosceles)
            if _compatible(v, inS):
                inS[v] = True
                S.add(v)
                e = _energy(S, inS)
                if e < cur_e or random.random() < np.exp(-(e - cur_e) / T):
                    cur_e = e
                    if e < best_e:
                        best_e = e
                        best = set(S)
                else:
                    inS[v] = False
                    S.discard(v)
        T *= cool
    return best, best_e


def construct():
    t0 = time.time()
    deadline = t0 + 285.0  # safety margin well under the 330s kill

    # Initial solutions: diverse greedy maximal sets, shrink each, keep best.
    best = None
    best_size = M + 1
    for _ in range(6):
        S = _shrink(_greedy_maximal())
        if len(S) < best_size:
            best = S
            best_size = len(S)

    # Simulated-annealing search over multiple restarts.
    plateau = 0
    while time.time() < deadline:
        # Perturb the best solution to create a diverse starting point for annealing.
        S = set(best)
        if random.random() < 0.5 and S:
            k = random.randint(1, max(1, len(S) // 3))
            for _ in range(k):
                if not S:
                    break
                x = random.choice(list(S))
                S.discard(x)
        S = _repair(S)

        # Anneal from this start.
        S, e = _anneal(S, _make_inS(S), deadline, t0)
        S = _shrink(S)

        if len(S) < best_size:
            best = S
            best_size = len(S)
            plateau = 0
        else:
            plateau += 1
            if plateau >= 40:
                # Diversify: restart from a fresh greedy solution.
                S = _shrink(_greedy_maximal())
                if len(S) < best_size:
                    best = S
                    best_size = len(S)
                plateau = 0

    # Final shrink if time permits.
    if time.time() < deadline - 2.0:
        best = _shrink(best)

    return [POINTS[i] for i in sorted(best)]
