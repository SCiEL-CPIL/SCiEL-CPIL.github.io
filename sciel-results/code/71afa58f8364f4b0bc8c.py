import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 15.0

def _check_cap_set(vecs, n):
    """Check if a set of vectors is a cap set (no 3-term AP)."""
    if len(vecs) < 3:
        return True
    s = set(vecs)
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a = vecs[i]
            b = vecs[j]
            # 3-term AP: a, (a+b)/2, b in Z_3^n means a + b + c = 0 (mod 3)
            # i.e., c = -(a+b) mod 3 = (3 - (a+b) % 3) % 3
            c = tuple((-a[k] - b[k]) % 3 for k in range(n))
            if c in s and c != a and c != b:
                return False
    return True

def _fourier_energy_penalty(vec, n):
    """
    Compute the sum of squared DFT coefficients at non-zero frequencies.
    For a single vector in Z_3^n, this measures its 'structure'.
    A vector with more uniform frequency content has lower penalty.
    """
    # For a single point, the DFT is trivial, so we use a proxy:
    # the variance of the coordinate values (how non-uniform the vector is)
    # and the sum of pairwise coordinate correlations
    coords = np.array(vec, dtype=float)
    # Variance penalty: prefer vectors with balanced coordinate values
    var_penalty = np.var(coords)
    # Pairwise correlation penalty
    if n >= 2:
        # Sum of products of coordinate pairs (measures linear structure)
        pair_sum = sum(coords[i] * coords[j] for i in range(n) for j in range(i + 1, n))
        corr_penalty = abs(pair_sum) / (n * (n - 1) / 2)
    else:
        corr_penalty = 0.0
    return var_penalty + 0.5 * corr_penalty

def _collision_penalty(vec, existing_set, n):
    """
    Estimate the number of 3-term APs that would be formed if vec is added.
    For each pair (a, b) in existing_set, check if vec completes an AP.
    Also check pairs where vec is one of the endpoints.
    """
    count = 0
    vecs = list(existing_set)
    for a in vecs:
        for b in vecs:
            if a == b:
                continue
            # Check if (a, b, vec) or (a, vec, b) or (vec, a, b) forms AP
            # In Z_3^n, a 3-term AP is x, y, z where x + z = 2y (mod 3)
            # i.e., x + z - 2y = 0 (mod 3)
            # For triple (a, b, vec): a + vec = 2*b mod 3
            if all((a[k] + vec[k] - 2 * b[k]) % 3 == 0 for k in range(n)):
                count += 1
            # For triple (a, vec, b): a + b = 2*vec mod 3
            if all((a[k] + b[k] - 2 * vec[k]) % 3 == 0 for k in range(n)):
                count += 1
            # For triple (vec, a, b): vec + b = 2*a mod 3
            if all((vec[k] + b[k] - 2 * a[k]) % 3 == 0 for k in range(n)):
                count += 1
    return count

def _priority_score(vec, existing_set, n, iteration):
    """
    Compute the priority score for a vector.
    Lower score = higher priority (we want to minimize structure/collisions).
    """
    # Fourier-based structure penalty
    fourier_pen = _fourier_energy_penalty(vec, n)
    
    # Collision penalty with existing set
    if len(existing_set) > 0:
        collision_pen = _collision_penalty(vec, existing_set, n)
    else:
        collision_pen = 0.0
    
    # L2 balance: prefer vectors that are "spread out"
    coords = np.array(vec, dtype=float)
    balance_pen = np.sum((coords - 1.0) ** 2)
    
    # Combined score: we want to minimize this
    # Weight collision heavily since it directly breaks cap set property
    score = 2.0 * collision_pen + 0.3 * fourier_pen + 0.2 * balance_pen
    return score

def priority(vector, n):
    """
    Greedy cap set construction.
    Returns the size of the largest cap set found.
    """
    global _START_TIME
    start_time = _START_TIME
    
    # Generate all vectors in Z_3^n
    all_vecs = list(product(range(3), repeat=n))
    
    # Compute initial priorities (without existing set, just structure-based)
    # Sort by structure penalty to get a good starting order
    init_scores = []
    for v in all_vecs:
        s = _fourier_energy_penalty(v, n) + 0.5 * np.sum((np.array(v, dtype=float) - 1.0) ** 2)
        init_scores.append((s, v))
    
    init_scores.sort(key=lambda x: x[0])
    candidates = [v for _, v in init_scores]
    
    # Greedy construction
    cap_set = set()
    cap_list = []
    
    for i, vec in enumerate(candidates):
        # Time check
        if time.time() - start_time > _TIME_LIMIT - _SAFETY_MARGIN:
            break
        
        # Check if adding this vector maintains cap set property
        if not cap_set:
            cap_set.add(vec)
            cap_list.append(vec)
            continue
        
        # Quick check: does this vector form a 3-term AP with any two existing vectors?
        forms_ap = False
        for a in cap_list:
            for b in cap_list:
                if a == b:
                    continue
                # Check if vec, a, b form a 3-term AP
                # In Z_3^n: x + z = 2y mod 3 for some permutation
                # (vec, a, b): vec + b = 2a
                if all((vec[k] + b[k] - 2 * a[k]) % 3 == 0 for k in range(n)):
                    forms_ap = True
                    break
                # (a, vec, b): a + b = 2*vec
                if all((a[k] + b[k] - 2 * vec[k]) % 3 == 0 for k in range(n)):
                    forms_ap = True
                    break
                # (a, b, vec): a + vec = 2b
                if all((a[k] + vec[k] - 2 * b[k]) % 3 == 0 for k in range(n)):
                    forms_ap = True
                    break
            if forms_ap:
                break
        
        if not forms_ap:
            cap_set.add(vec)
            cap_list.append(vec)
    
    return float(len(cap_set))
