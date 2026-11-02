"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

DEADLINE = 120.0
SAFETY = 10.0


def _unit_vectors():
    """Return the 2*D unit vectors +/-e_i as a list of tuples."""
    out = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            out.append(tuple(v))
    return out


def _candidate_norm2_vectors():
    """Generate candidate integer vectors of norm 2 (support size 2 or 3)."""
    cands = []
    # support size 2: e_i + s*e_j
    for i, j in combinations(range(D), 2):
        for s in (1, -1):
            v = [0] * D
            v[i], v[j] = 1, s
            cands.append(tuple(v))
    # support size 3: e_i + s1*e_j + s2*e_k
    for i, j, k in combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i], v[j], v[k] = 1, s1, s2
                cands.append(tuple(v))
    return cands


def _sq_norm(v):
    return sum(x * x for x in v)


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY

    # Start with the 220 unit vectors (norm 1). Pairwise distances >= 1 = max_norm.
    selected = _unit_vectors()
    selected_set = set(selected)

    # Greedy add norm-2 vectors. A norm-2 vector v is compatible with all
    # selected vectors u iff for each u, ||v - u||^2 >= ||v||^2 = 4.
    # Since ||u||^2 is 1 (unit) or 4 (norm-2), this is a dot-product check.
    # We precompute compatibility with the unit vectors via support.

    # For a norm-2 candidate v with support S:
    #   - with unit vector u: need v.u <= 1  (since ||v||^2+||u||^2-2v.u >= 4 -> 5-2v.u>=4 -> v.u<=1)
    #     v.u is in {-2,-1,0,1,2}; so need v.u <= 1, i.e., v.u not in {2}.
    #     v.u = 2 happens iff u = v/2, impossible for unit u. Actually v.u=2 requires
    #     u aligned with two coords of v at value 1 each -> u would need two 1s, not unit.
    #     For unit u, v.u = v[i] if u = e_i (i in support), else 0. v[i] is +/-1.
    #     So v.u in {-1,0,1} always for unit u -> always compatible! 
    #   Wait: that means norm-2 vectors are ALWAYS compatible with unit vectors.
    #   Let's verify: v = e_0+e_1, u = e_0: v.u=1, ||v-u||^2 = ||e_1||^2 = 1 < 4. NOT compatible!
    #   Correction: need ||v-u||^2 >= ||v||^2 = 4. ||v-u||^2 = ||v||^2+||u||^2-2v.u = 4+1-2*1 = 3 < 4. Fail.
    #   So need 4+1-2v.u >= 4 -> 1 >= 2v.u -> v.u <= 0.5 -> v.u <= 0 (integer).
    #   So v.u must be <= 0. For unit u, v.u = v[i] if u=e_i, = -v[i] if u=-e_i, else 0.
    #   Need v[i] <= 0 for u=e_i, and -v[i] <= 0 (i.e., v[i]>=0) for u=-e_i.
    #   Both can't hold unless v[i]=0. So u must be ORTHOGONAL to v's support.

    # Thus: a norm-2 vector v with support S is compatible with unit vector u iff
    #   u's support is disjoint from S.

    # For compatibility between two norm-2 vectors v, w: need ||v-w||^2 >= 4.
    # ||v-w||^2 = ||v||^2+||w||^2-2v.w = 4+4-2v.w = 8-2v.w >= 4 -> v.w <= 2.

    # Greedy: add norm-2 vectors one by one, checking against all selected.
    cands = _candidate_norm2_vectors()

    # Sort candidates: more constrained (larger support) first for better packing.
    cands.sort(key=lambda v: -sum(1 for x in v if x != 0))

    for v in cands:
        if time.time() > deadline:
            break
        # Check compatibility with all selected vectors
        ok = True
        for u in selected:
            # compute ||v-u||^2 and ||v||^2
            # We need ||v-u||^2 >= max(||v||^2, ||u||^2)
            # Actually the constraint is min pairwise dist >= max norm over ALL vectors.
            # So we need ||v-u|| >= max_norm where max_norm is the max norm in the set.
            # Currently max_norm^2 = 4 (from norm-2 vectors) or 1 (only units).
            # If we add norm-2 vectors, max_norm^2 = 4.
            # So need ||v-u||^2 >= 4 for all u.
            dot = sum(a * b for a, b in zip(v, u))
            norm_u_sq = sum(b * b for b in u)
            norm_v_sq = _sq_norm(v)
            dist_sq = norm_v_sq + norm_u_sq - 2 * dot
            if dist_sq < 4:  # max_norm^2 = 4 once we have any norm-2 vector
                ok = False
                break
        if ok:
            # Also check v is not already in set
            if v not in selected_set:
                selected.append(v)
                selected_set.add(v)

    result = np.array(selected, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
