import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = len(POINTS)  # 256

# ---------- Precompute ----------
# COVER[p] = list of (a,b), a<b, a,b != p, such that triangle (p,a,b) is isosceles.
def _build_cover():
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    iu_a, iu_b = np.triu_indices(M, k=1)
    cover = []
    for p in range(M):
        d_pa = D2[p, A][:, None]
        d_pb = D2[p, A][None, :]
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        cover.append(list(zip(a[sel].tolist(), b[sel].tolist())))
    return cover

COVER = _build_cover()

# For fast vectorized cover counting:
#   C_a[p] = np.array of a's, C_b[p] = np.array of b's
C_a = [np.array([ab[0] for ab in COVER[p]]) for p in range(M)]
C_b = [np.array([ab[1] for ab in COVER[p]]) for p in range(M)]

# For fast compatibility check:
#   COMPAT[p] = set of pairs (a,b) that would make p incompatible (i.e. a,b in S -> p blocked)
COMPAT = [set(COVER[p]) for p in range(M)]


def _make_inS(S):
    inS = [False] * M
    for v in S:
        inS[v] = True
    return inS


def _compatible(v, inS):
    for a, b in COVER[v]:
        if inS[a] and inS[b]:
            return False
    return True


def _is_maximal(inS):
    for p in range(M):
        if inS[p]:
            continue
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                break
        else:
            return False
    return True


def _cover_counts(S):
    """Vectorized: for each outside point p, count isosceles pairs (a,b) in S covering p."""
    Ss = sorted(S)
    if len(Ss) < 2:
        return np.zeros(M, dtype=np.int32)
    Sa = np.array(Ss)
    iu_a, iu_b = np.triu_indices(len(Ss), k=1)
    pa = Sa[iu_a]; pb = Sa[iu_b]
    counts = np.zeros(M, dtype=np.int32)
    for p in range(M):
        if p in S:
            continue
        ca = C_a[p]; cb = C_b[p]
        if ca.size == 0:
            continue
        counts[p] = int(np.sum((ca[:, None] == pa[None, :]) & (cb[:, None] == pb[None, :])))
    return counts


def _shrink(S):
    """Inclusion-minimal maximal: remove any point whose removal keeps maximality.
    Vectorized cover-count check: maximality holds iff every outside point has cover>=1."""
    S = set(S)
    # fast-path: if already maximal, use vectorized cover counting
    inS = _make_inS(S)
    if _is_maximal(inS):
        improved = True
        while improved:
            improved = False
            counts = _cover_counts(S)
            Sset = set(S)
            # Try to remove points that are "least critical" (low cover count) first?
            # Actually, just iterate. If counts[x] == 0, it's a candidate for removal?
            # No, counts[x] is the number of pairs in S covering x.
            # If we remove x, we need to check if S\{x} is still maximal.
            # This is equivalent to checking if every point outside S\{x} is covered by S\{x}.
            # A point p outside S\{x} is covered by S\{x} if it was covered by S and not solely by pairs involving x.
            # Or more simply: just recompute counts for S\{x} and check if all outside points have count >= 1.
            
            # Optimization: Only check removal of points that are not "essential" for covering some outside point?
            # If x is in S, removing x might uncover some points.
            # We can just iterate through S and try to remove.
            
            # To make it faster, we can randomize the order of checking removals.
            items = list(S)
            random.shuffle(items)
            for x in items:
                # If counts[x] > 0, it means x is covered by S. This doesn't tell us if x is needed to cover others.
                # We must check if S - {x} is maximal.
                S2 = Sset - {x}
                c2 = _cover_counts(S2)
                outside = [p for p in range(M) if p not in S2]
                # Check if all outside points have c2[p] >= 1
                # Vectorized check:
                if len(outside) == 0:
                    # S2 is all points? Unlikely.
                    pass
                else:
                    out_arr = np.array(outside)
                    if np.all(c2[out_arr] >= 1):
                        S = S2
                        improved = True
                        break
        return S
    # fallback: generic (may need repair first)
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            inS = _make_inS(S)
            inS[x] = False
            S.discard(x)
            if _is_maximal(inS):
                improved = True
            else:
                S.add(x)
    return S


def _repair(S):
    """Greedy-complete S to a maximal no-isosceles set (adds compatible points)."""
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


def construct():
    t0 = time.time()
    deadline = t0 + 290.0

    # Initial: diverse greedy maximal sets, shrink, keep best.
    best = None
    best_size = M + 1
    for _ in range(5):
        S = _shrink(_greedy_maximal())
        if len(S) < best_size:
            best = S
            best_size = len(S)

    # Simulated Annealing
    current = set(best)
    current_size = len(current)
    
    T_start = 5.0
    T_end = 0.1
    alpha = 0.995
    
    # Estimate steps
    # _shrink + _repair + _cover_counts takes ~1-2ms? 
    # Let's assume ~5ms per iteration.
    # Time available: 290s.
    # Max iterations: 290 / 0.005 = 58000.
    
    t = T_start
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline - 2.0 and steps < max_steps:
        # Perturbation: Remove k random points
        # k depends on temperature? Or just random 1-3.
        k = random.randint(1, 3)
        S_perturbed = set(current)
        for _ in range(k):
            if S_perturbed:
                x = random.choice(list(S_perturbed))
                S_perturbed.discard(x)
        
        # Repair and Shrink
        S_new = _repair(S_perturbed)
        S_new = _shrink(S_new)
        new_size = len(S_new)
        
        # Acceptance
        delta = new_size - current_size
        if delta < 0:
            accept = True
        else:
            if t > 0:
                r = random.random()
                # Probability of accepting worse solution
                if r < np.exp(-delta / t):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            current = S_new
            current_size = new_size
            if current_size < best_size:
                best = current
                best_size = current_size
        
        # Cooling
        t *= alpha
        steps += 1

    # Final shrink
    if time.time() < deadline - 1.0:
        best = _shrink(best)

    return [POINTS[i] for i in sorted(best)]
