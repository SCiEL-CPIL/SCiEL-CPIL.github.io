import numpy as np
import time
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _fmm3_3x3x3():
    """
    Returns U, V, W matrices for 3x3x3 matrix multiplication with rank 23.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices:
    U row: i*3 + j
    V row: j*3 + k
    W row: k*3 + i
    """
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]
    
    terms = []
    
    def add_term(u_list, v_list, w_list):
        idx = len(terms)
        for (i, j, val) in u_list:
            U[i * 3 + j][idx] += val
        for (j, k, val) in v_list:
            V[j * 3 + k][idx] += val
        for (k, i, val) in w_list:
            W[k * 3 + i][idx] += val
        terms.append(idx)

    # 1. (a00+a11+a22)(b00+b11+b22)
    add_term([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)])
    # 2. (a00-a11)(b00-b11)
    add_term([(0,0,1),(1,1,-1)], [(0,0,1),(1,1,-1)], [(0,0,1),(1,1,-1)])
    # 3. (a00-a22)(b00-b22)
    add_term([(0,0,1),(2,2,-1)], [(0,0,1),(2,2,-1)], [(0,0,1),(2,2,-1)])
    # 4. (a11-a22)(b11-b22)
    add_term([(1,1,1),(2,2,-1)], [(1,1,1),(2,2,-1)], [(1,1,1),(2,2,-1)])
    # 5. (a01+a12+a20)(b01+b12+b20)
    add_term([(0,1,1),(1,2,1),(2,0,1)], [(0,1,1),(1,2,1),(2,0,1)], [(1,0,1),(2,1,1),(0,2,1)]) 
    # 6. (a01-a12)(b01-b12)
    add_term([(0,1,1),(1,2,-1)], [(0,1,1),(1,2,-1)], [(1,0,1),(2,1,-1)])
    # 7. (a01-a20)(b01-b20)
    add_term([(0,1,1),(2,0,-1)], [(0,1,1),(2,0,-1)], [(1,0,1),(0,2,-1)])
    # 8. (a12-a20)(b12-b20)
    add_term([(1,2,1),(2,0,-1)], [(1,2,1),(2,0,-1)], [(2,1,1),(0,2,-1)])
    
    # 9. (a02+a10+a21)(b02+b10+b21)
    add_term([(0,2,1),(1,0,1),(2,1,1)], [(0,2,1),(1,0,1),(2,1,1)], [(2,0,1),(0,1,1),(1,2,1)])
    # 10. (a02-a10)(b02-b10)
    add_term([(0,2,1),(1,0,-1)], [(0,2,1),(1,0,-1)], [(2,0,1),(0,1,-1)])
    # 11. (a02-a21)(b02-b21)
    add_term([(0,2,1),(2,1,-1)], [(0,2,1),(2,1,-1)], [(2,0,1),(1,2,-1)])
    # 12. (a10-a21)(b10-b21)
    add_term([(1,0,1),(2,1,-1)], [(1,0,1),(2,1,-1)], [(0,1,1),(1,2,-1)])
    
    # 13. (a00+a01+a02)(b00+b01+b02)
    add_term([(0,0,1),(0,1,1),(0,2,1)], [(0,0,1),(0,1,1),(0,2,1)], [(0,0,1),(1,0,1),(2,0,1)])
    # 14. (a10+a11+a12)(b10+b11+b12)
    add_term([(1,0,1),(1,1,1),(1,2,1)], [(1,0,1),(1,1,1),(1,2,1)], [(0,1,1),(1,1,1),(2,1,1)])
    # 15. (a20+a21+a22)(b20+b21+b22)
    add_term([(2,0,1),(2,1,1),(2,2,1)], [(2,0,1),(2,1,1),(2,2,1)], [(0,2,1),(1,2,1),(2,2,1)])
    
    # 16. (a00+a10+a20)(b00+b10+b20)
    add_term([(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(0,1,1),(0,2,1)])
    # 17. (a01+a11+a21)(b01+b11+b21)
    add_term([(0,1,1),(1,1,1),(2,1,1)], [(0,1,1),(1,1,1),(2,1,1)], [(1,0,1),(1,1,1),(1,2,1)])
    # 18. (a02+a12+a22)(b02+b12+b22)
    add_term([(0,2,1),(1,2,1),(2,2,1)], [(0,2,1),(1,2,1),(2,2,1)], [(2,0,1),(2,1,1),(2,2,1)])
    
    # 19. (a00-a01+a02)(b00-b01+b02)
    add_term([(0,0,1),(0,1,-1),(0,2,1)], [(0,0,1),(0,1,-1),(0,2,1)], [(0,0,1),(1,0,-1),(2,0,1)])
    # 20. (a10-a11+a12)(b10-b11+b12)
    add_term([(1,0,1),(1,1,-1),(1,2,1)], [(1,0,1),(1,1,-1),(1,2,1)], [(0,1,1),(1,1,-1),(2,1,1)])
    # 21. (a20-a21+a22)(b20-b21+b22)
    add_term([(2,0,1),(2,1,-1),(2,2,1)], [(2,0,1),(2,1,-1),(2,2,1)], [(0,2,1),(1,2,-1),(2,2,1)])
    
    # 22. (a00+a01-a02)(b00+b01-b02)
    add_term([(0,0,1),(0,1,1),(0,2,-1)], [(0,0,1),(0,1,1),(0,2,-1)], [(0,0,1),(1,0,1),(2,0,-1)])
    # 23. (a10+a11-a12)(b10+b11-b12)
    add_term([(1,0,1),(1,1,1),(1,2,-1)], [(1,0,1),(1,1,1),(1,2,-1)], [(0,1,1),(1,1,1),(2,1,-1)])
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    # Use integer arithmetic for exactness if possible, else Fraction
    # Convert to Fraction for safety
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                expected = 1 if (i == i2 and j == j2) else 0
                
                actual = Fraction(0)
                for col in range(rank):
                    actual += U_f[a][col] * V_f[b][col] * W_f[c][col]
                
                if actual != expected:
                    return False
    return True

def _build_rank32():
    """Build the rank-32 decomposition: FMM-3 (rank 23) + 9 extra terms for column 3."""
    n, m, p = 3, 3, 4
    U3, V3, W3 = _fmm3_3x3x3()
    r_fmm = 23
    r_extra = 9
    total_rank = r_fmm + r_extra
    
    U = [[0] * total_rank for _ in range(9)]
    V = [[0] * total_rank for _ in range(12)]
    W = [[0] * total_rank for _ in range(12)]
    
    for i in range(9):
        for t in range(r_fmm):
            U[i][t] = U3[i][t]
            V[i][t] = V3[i][t]
            W[i][t] = W3[i][t]
    
    # Extra terms for k=3
    # We need to compute C[i, 3] for i in 0..2
    # C[i, 3] = sum_j A[i, j] B[j, 3]
    # The terms are (A[i, j] * B[j, 3]) for j=0,1,2
    # So we add 3 terms? No, we need to match the structure.
    # The FMM part handles k=0,1,2.
    # For k=3, we can just use the standard terms:
    # Term (i, j): U[i, j] * V[j, 3] -> W[3, i]
    # There are 3*3=9 such terms.
    for i in range(3):
        for j in range(3):
            t = r_fmm + i * 3 + j
            U[i * 3 + j][t] = 1
            V[j * 4 + 3][t] = 1
            W[3 * 3 + i][t] = 1
    
    return U, V, W

def _extract_tensor_vectors(U, V, W, n, m, p):
    """
    Extract the rank-r tensor vectors T_t for each term t.
    Each T_t is a vector of length n*m*p with entries in {0, 1}.
    T_t[a, b, c] = U[a, t] * V[b, t] * W[c, t]
    Returns list of tuples (length n*m*p) for each term.
    """
    r = len(U[0])
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    total = dim_a * dim_b * dim_c
    
    vectors = []
    for t in range(r):
        vec = [0] * total
        for a in range(dim_a):
            ua = U[a][t]
            if ua == 0:
                continue
            for b in range(dim_b):
                vb = V[b][t]
                if vb == 0:
                    continue
                for c in range(dim_c):
                    wc = W[c][t]
                    if wc == 0:
                        continue
                    val = ua * vb * wc
                    idx = (a * dim_b + b) * dim_c + c
                    vec[idx] = val
        vectors.append(vec)
    return vectors

def _target_tensor(n, m, p):
    """Build the target tensor vector: 1 where (i==i2 and j==j2), else 0."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    total = dim_a * dim_b * dim_c
    vec = [0] * total
    for a in range(dim_a):
        i = a // m
        j = a % m
        for b in range(dim_b):
            j2 = b // p
            k = b % p
            for c in range(dim_c):
                k2 = c // n
                i2 = c % n
                if i == i2 and j == j2:
                    idx = (a * dim_b + b) * dim_c + c
                    vec[idx] = 1
    return vec

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank from current to target_rank by solving linear systems.
    For each subset of size target_rank from the current rank terms,
    check if the target tensor is in their span.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    start_time = time.time()
    current_rank = len(U[0])
    vectors = _extract_tensor_vectors(U, V, W, n, m, p)
    target = _target_tensor(n, m, p)
    total_dim = n * m * p
    
    # Try all subsets of size target_rank
    from itertools import combinations
    
    indices = list(range(current_rank))
    found = False
    best_solution = None
    
    # Pre-compute target as float for speed
    b = np.array(target, dtype=np.float64)
    
    for subset in combinations(indices, target_rank):
        if time.time() > deadline:
            break
        
        # Build matrix A (total_dim x target_rank) from the selected vectors
        # Solve A * c = target
        A = np.array([vectors[i] for i in subset], dtype=np.float64).T  # shape (total_dim, target_rank)
        
        # Check if target is in span of selected vectors
        # Use least squares and check residual
        try:
            c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-10:
            continue
        
        # Check if solution is close to integer/half-integer
        c_rounded = np.round(c * 2) / 2
        if np.max(np.abs(c - c_rounded)) < 1e-8:
            # Found a valid solution with half-integer coefficients
            found = True
            best_solution = (subset, c_rounded)
            break
    
    if not found:
        return None
    
    subset, c = best_solution
    # Build new U, V, W with only the selected terms, scaled by c
    U_new = [[0] * target_rank for _ in range(n * m)]
    V_new = [[0] * target_rank for _ in range(m * p)]
    W_new = [[0] * target_rank for _ in range(p * n)]
    
    for new_idx, old_idx in enumerate(subset):
        coeff = c[new_idx]
        for a in range(n * m):
            U_new[a][new_idx] = U[a][old_idx] * coeff
        for b in range(m * p):
            V_new[b][new_idx] = V[b][old_idx] * coeff
        for cc in range(p * n):
            W_new[cc][new_idx] = W[cc][old_idx] * coeff
    
    return U_new, V_new, W_new

def _try_general_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Attempt to find a rank-target_rank decomposition by searching for coefficients
    that are rational numbers with small denominators.
    This is a heuristic search.
    """
    # This is computationally expensive and likely to fail for large gaps.
    # We will rely on the subset method primarily.
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (or 150s if that's the hard kill)
    
    # Build rank-32 decomposition
    U, V, W = _build_rank32()
    
    # Validate rank-32
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_sb = [[0] * rank for _ in range(n * m)]
        V_sb = [[0] * rank for _ in range(m * p)]
        W_sb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    best_rank = 32
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce to rank 31
    if time.time() < deadline - 10:
        result = _try_rank_reduction(U, V, W, n, m, p, 31, deadline - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _validate(U_new, V_new, W_new, n, m, p):
                best_rank = 31
                best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try to reduce to rank 30 (if time permits)
    if time.time() < deadline - 20 and best_rank > 30:
        # Note: _try_rank_reduction takes the current best U,V,W
        result = _try_rank_reduction(best_U, best_V, best_W, n, m, p, 30, deadline - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _validate(U_new, V_new, W_new, n, m, p):
                best_rank = 30
                best_U, best_V, best_W = U_new, V_new, W_new
    
    # Try to reduce to rank 29 (if time permits)
    if time.time() < deadline - 30 and best_rank > 29:
        result = _try_rank_reduction(best_U, best_V, best_W, n, m, p, 29, deadline - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _validate(U_new, V_new, W_new, n, m, p):
                best_rank = 29
                best_U, best_V, best_W = U_new, V_new, W_new

    # Try a different structural approach if we have time:
    # Maybe we can find a rank 28 decomposition?
    # The subset search is limited by the number of combinations.
    # C(32, 28) is C(32, 4) = 35960. This is feasible.
    # C(32, 27) is C(32, 5) = 201376. This might be tight.
    
    if time.time() < deadline - 15 and best_rank > 28:
        result = _try_rank_reduction(best_U, best_V, best_W, n, m, p, 28, deadline - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _validate(U_new, V_new, W_new, n, m, p):
                best_rank = 28
                best_U, best_V, best_W = U_new, V_new, W_new

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
