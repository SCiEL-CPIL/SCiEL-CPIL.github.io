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

def _get_violations(S, inS):
    """
    Returns (indep_violations, max_violations)
    indep_violations: number of isosceles triangles in S
    max_violations: number of points p not in S that are not covered by S
    """
    indep_viol = 0
    # Count isosceles triangles in S
    # For each p in S, count pairs (a,b) in S such that (p,a,b) is isosceles.
    # Each triangle is counted 3 times (once for each vertex).
    # So indep_viol = total_counts // 3
    total_counts = 0
    for p in S:
        # Check pairs (a,b) in S
        # We can use COVER[p]
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                total_counts += 1
    indep_viol = total_counts // 3
    
    max_viol = 0
    for p in range(M):
        if inS[p]:
            continue
        covered = False
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                covered = True
                break
        if not covered:
            max_viol += 1
            
    return indep_viol, max_viol

def _objective(S, inS):
    indep_viol, max_viol = _get_violations(S, inS)
    # Penalty should be large enough to keep us feasible, but allow exploration
    # If we want strictly feasible, we can use a huge penalty.
    # But for SA, we want to explore infeasible.
    return len(S) + 100 * (indep_viol + max_viol)

def construct():
    t0 = time.time()
    deadline = t0 + 285.0
    
    # Start with a greedy maximal set
    def _greedy_maximal():
        S = set()
        inS = _make_inS(S)
        order = list(range(M))
        random.shuffle(order)
        for v in order:
            # Check compatibility
            compatible = True
            for a, b in COVER[v]:
                if inS[a] and inS[b]:
                    compatible = False
                    break
            if compatible:
                S.add(v)
                inS[v] = True
        return S, inS

    S, inS = _greedy_maximal()
    current_obj = _objective(S, inS)
    
    best_S = set(S)
    best_obj = current_obj
    best_inS = list(inS)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = random.random()
        S_new = set(S)
        inS_new = list(inS)
        
        if move_type < 0.33:
            # Add
            candidates = [v for v in range(M) if not inS[v]]
            if candidates:
                v = random.choice(candidates)
                S_new.add(v)
                inS_new[v] = True
        elif move_type < 0.66:
            # Remove
            if S:
                v = random.choice(list(S))
                S_new.discard(v)
                inS_new[v] = False
        else:
            # Swap
            if S:
                v_in = random.choice(list(S))
                candidates = [v for v in range(M) if not inS[v]]
                if candidates:
                    v_out = random.choice(candidates)
                    S_new.discard(v_in)
                    S_new.add(v_out)
                    inS_new[v_in] = False
                    inS_new[v_out] = True
        
        new_obj = _objective(S_new, inS_new)
        delta = new_obj - current_obj
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            S = S_new
            inS = inS_new
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_S = set(S)
                best_inS = list(inS)
                
        T *= cooling
        
    # Ensure the best solution is valid (independent and maximal)
    # If best_obj has violations, we need to repair it.
    # But SA with high penalty should converge to feasible.
    # Let's verify best_S
    inS_best = _make_inS(best_S)
    indep, maxv = _get_violations(best_S, inS_best)
    
    if indep > 0 or maxv > 0:
        # Repair: 
        # 1. Remove points to fix independence
        # 2. Add points to fix maximality
        # This is a fallback. Ideally SA finds a feasible solution.
        # For now, if infeasible, use the greedy one if it was better?
        # No, the objective includes penalties.
        # If the best found is infeasible, it's bad.
        # We should try to repair best_S.
        
        # Remove conflicting points
        changed = True
        while changed:
            changed = False
            for v in list(best_S):
                # Check if v is part of an isosceles triangle
                for a, b in COVER[v]:
                    if a in best_S and b in best_S:
                        best_S.discard(v)
                        changed = True
                        break
        
        # Add points to ensure maximality
        inS_best = _make_inS(best_S)
        order = list(range(M))
        random.shuffle(order)
        for v in order:
            if inS_best[v]:
                continue
            # Check if v is covered
            covered = False
            for a, b in COVER[v]:
                if inS_best[a] and inS_best[b]:
                    covered = True
                    break
            if not covered:
                # Check if v can be added (independence)
                compatible = True
                for a, b in COVER[v]:
                    if inS_best[a] and inS_best[b]:
                        compatible = False
                        break
                if compatible:
                    best_S.add(v)
                    inS_best[v] = True
                else:
                    # We are stuck? This shouldn't happen if we start from maximal.
                    # But we removed points.
                    # We might need to add a point that is NOT compatible? No, must be independent.
                    # If no compatible point can cover v, we might have broken maximality irreparably?
                    # Actually, if v is not covered, we MUST add some point to cover it?
                    # No, we need to add points to S such that v is covered.
                    # This is a set cover problem.
                    # For now, this repair is naive.
                    pass

    return [POINTS[i] for i in sorted(best_S)]
