import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square using numpy convolution."""
    # Trim trailing zeros for efficiency but keep at least 1 element
    n = len(coefs)
    if n == 0:
        return 0
    # Use numpy convolve for speed
    arr = np.array(coefs, dtype=np.int64)
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _count_nz(coefs):
    return sum(1 for c in coefs if c != 0)

def _try_product_structure(best, rng):
    """
    Try P = A * B where A and B are sparse polynomials.
    If A has k terms and B has m terms, P has at most k*m terms.
    We want exactly 12 nonzero terms in P.
    The square P^2 = A^2 * B^2.
    
    Key insight: If A and B are chosen such that A^2 and B^2 have few terms,
    then P^2 = A^2 * B^2 may also have few terms due to structure.
    
    Specifically, try:
    - A = sum of a_i x^{e_i}, B = sum of b_j x^{f_j}
    - P = A * B has terms at degrees e_i + f_j
    - We need exactly 12 distinct nonzero terms in P
    """
    # Try small A and B
    # A with 2 terms, B with 6 terms -> P has at most 12 terms
    # A with 3 terms, B with 4 terms -> P has at most 12 terms
    # A with 4 terms, B with 3 terms -> P has at most 12 terms
    # A with 6 terms, B with 2 terms -> P has at most 12 terms
    
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _iter in range(20000):
        if time.time() > _DEADLINE:
            break
        
        # Choose factor sizes
        r = rng.random()
        if r < 0.4:
            k, m = 2, 6
        elif r < 0.7:
            k, m = 3, 4
        elif r < 0.9:
            k, m = 4, 3
        else:
            k, m = 6, 2
        
        # Generate A: k terms
        exponents_a = sorted(rng.sample(range(0, 25), k))
        coeffs_a = [rng.choice(val_pool) for _ in range(k)]
        
        # Generate B: m terms
        exponents_b = sorted(rng.sample(range(0, 25), m))
        coeffs_b = [rng.choice(val_pool) for _ in range(m)]
        
        # Compute P = A * B
        max_deg = max(exponents_a) + max(exponents_b)
        if max_deg > 40:
            continue
        
        P = [0] * 41
        for ea, ca in zip(exponents_a, coeffs_a):
            for eb, cb in zip(exponents_b, coeffs_b):
                d = ea + eb
                if d < 41:
                    P[d] += ca * cb
        
        # Check validity
        nz = _count_nz(P)
        if nz != 12 or P[40] == 0:
            # Try to fix: ensure P[40] != 0
            if nz == 12 and P[40] == 0:
                # Find a way to make P[40] nonzero
                # This is tricky, skip for now
                continue
            continue
        
        if not all(-64 <= c <= 64 for c in P):
            continue
        
        sc = _sq_nonzero_count(P)
        if sc < best[0]:
            best[0] = sc
            best[1] = P[:]
    
    return best

def _try_sum_structure(best, rng):
    """
    Try P = A + B where A and B are sparse.
    P^2 = A^2 + B^2 + 2AB.
    If A^2 and B^2 have few terms and 2AB has few terms, P^2 has few terms.
    
    Try A with 6 terms, B with 6 terms (but overlapping to get 12 total in P).
    Or A with 8 terms, B with 8 terms with 4 overlapping.
    """
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _iter in range(15000):
        if time.time() > _DEADLINE:
            break
        
        # Choose number of terms in A and B
        r = rng.random()
        if r < 0.5:
            # A and B each have 6 terms, no overlap -> P has 12 terms
            k_a, k_b = 6, 6
            overlap = 0
        else:
            # A and B each have 8 terms, 4 overlap -> P has 12 terms
            k_a, k_b = 8, 8
            overlap = 4
        
        # Generate support for A
        support_a = sorted(rng.sample(range(0, 35), k_a))
        coeffs_a = [rng.choice(val_pool) for _ in range(k_a)]
        
        # Generate support for B
        if overlap > 0:
            # Choose overlap degrees
            overlap_degs = rng.sample(support_a, overlap)
            remaining = [d for d in support_a if d not in overlap_degs]
            support_b_extra = sorted(rng.sample([d for d in range(0, 35) if d not in support_a], k_b - overlap))
            support_b = sorted(overlap_degs + support_b_extra)
        else:
            support_b = sorted(rng.sample([d for d in range(0, 35) if d not in support_a], k_b))
        
        coeffs_b = [rng.choice(val_pool) for _ in range(k_b)]
        
        # Compute P = A + B
        P = [0] * 41
        for d, c in zip(support_a, coeffs_a):
            if d < 41:
                P[d] += c
        for d, c in zip(support_b, coeffs_b):
            if d < 41:
                P[d] += c
        
        nz = _count_nz(P)
        if nz != 12 or P[40] == 0:
            continue
        
        if not all(-64 <= c <= 64 for c in P):
            continue
        
        sc = _sq_nonzero_count(P)
        if sc < best[0]:
            best[0] = sc
            best[1] = P[:]
    
    return best

def _local_search(best, rng):
    """
    Local search: start from a valid 12-term polynomial, try to reduce square terms.
    Use numpy for fast square computation.
    """
    if best[1] is None:
        return best
    
    P = best[1][:]
    best_score = best[0]
    
    # Get support
    support = [i for i in range(41) if P[i] != 0]
    n_sup = len(support)
    
    val_pool = [-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8]
    
    for _pass in range(50):
        if time.time() > _DEADLINE:
            break
        improved = False
        for idx in range(n_sup):
            if time.time() > _DEADLINE:
                break
            d = support[idx]
            old_val = P[d]
            
            # Try changing this coefficient
            for new_val in val_pool:
                if new_val == old_val:
                    continue
                P[d] = new_val
                if not _valid(P):
                    P[d] = old_val
                    continue
                sc = _sq_nonzero_count(P)
                if sc < best_score:
                    best_score = sc
                    best[0] = sc
                    best[1] = P[:]
                    improved = True
                else:
                    P[d] = old_val
            
            # Try adding a new term and removing an existing one
            if time.time() > _DEADLINE:
                break
            for d2 in range(41):
                if d2 in support:
                    continue
                # Remove term at d, add term at d2
                old_val2 = P[d2]
                P[d] = 0
                P[d2] = rng.choice(val_pool)
                if _valid(P):
                    sc = _sq_nonzero_count(P)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = P[:]
                        support = [i for i in range(41) if P[i] != 0]
                        n_sup = len(support)
                        improved = True
                        break
                P[d] = old_val
                P[d2] = old_val2
                if time.time() > _DEADLINE:
                    break
        if not improved:
            break
    
    return best

def _try_geometric_structure(best, rng):
    """
    Try polynomials with geometric or arithmetic progression structure.
    These often have structured squares.
    """
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    # Arithmetic progression supports
    for start in range(0, 10):
        for gap in range(2, 15):
            if time.time() > _DEADLINE:
                break
            sup = []
            d = start
            while d <= 40 and len(sup) < 12:
                sup.append(d)
                d += gap
            if len(sup) < 12:
                continue
            if 40 not in sup:
                sup = sup[:11] + [40]
                sup = sorted(set(sup))
                if len(sup) != 12:
                    continue
            
            # Try random coefficients
            for _ in range(200):
                if time.time() > _DEADLINE:
                    break
                P = [0] * 41
                for d in sup:
                    P[d] = rng.choice(val_pool)
                if P[40] == 0:
                    P[40] = 1
                if not _valid(P):
                    continue
                if not all(-64 <= c <= 64 for c in P):
                    continue
                sc = _sq_nonzero_count(P)
                if sc < best[0]:
                    best[0] = sc
                    best[1] = P[:]
                    # Do local search
                    best = _local_search(best, rng)
                    if time.time() > _DEADLINE:
                        return best
    
    return best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    rng = random.Random(42)
    best = [10**9, None]
    
    # Phase 1: Product structure search
    best = _try_product_structure(best, rng)
    
    # Phase 2: Sum structure search
    best = _try_sum_structure(best, rng)
    
    # Phase 3: Geometric/arithmetic structure search
    best = _try_geometric_structure(best, rng)
    
    # Phase 4: Local search from best found
    if best[1] is not None:
        best = _local_search(best, rng)
    
    # Fallback if nothing found
    if best[1] is None:
        # Try a simple 12-term polynomial
        P = [0] * 41
        for i in range(12):
            P[i] = 1
        best[0] = _sq_nonzero_count(P)
        best[1] = P
    
    return best[1]
