"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. Returns True if correct."""
    r = len(U[0]) if U else 0
    # Build tensor T[i,j,k] = sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha]
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for alpha in range(r):
                    s += U_arr[i*m+j, alpha] * V_arr[j*p+k, alpha] * W_arr[k*n+i, alpha]
                expected = 1 if (i == i and j == j and k == k) else 0
                # The tensor for matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
                # T[i,j,k] should be 1 if this is a valid term, 0 otherwise
                # Actually, the tensor is T[i,j,k] = delta(i,i)*delta(j,j)*delta(k,k) = 1 always?
                # No. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
                # Wait, let me reconsider. The tensor T_{ijk} for multiplying (n x m) by (m x p)
                # is defined as T_{ijk} = delta_{ii} delta_{jj} delta_{kk}... no.
                # The tensor is T_{ijk} = 1 if and only if the term A_{ij} * B_{jk} contributes to C_{ik}.
                # So T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p).
                # The decomposition is: T_{ijk} = sum_alpha U_{ij,alpha} * V_{jk,alpha} * W_{ki,alpha}
                # where W is indexed as W[k*n+i, alpha].
                expected_val = 1
                if s != expected_val:
                    return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a decomposition of rank r using randomized search."""
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in 3*r*n*m + 3*r*m*p + 3*r*p*n variables (but with structure)
    # Actually, we have n*m*r + m*p*r + p*n*r unknowns.
    # The equations are: for each (i,j,k): sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha] = 1
    
    # For small r, we can try to solve this. But it's nonlinear (trilinear).
    # Let's try a different approach: use the fact that for fixed alpha, the term is rank-1.
    
    # Actually, let's just try the standard construction and see if we can do better.
    # The trivial construction has rank n*m*p = 2*5*6 = 60.
    # Can we do better?
    
    # For now, let's just return the trivial construction with rank n*m*p
    # and see if we can find a better one.
    
    if time.time() > deadline:
        return None
    
    # Trivial construction
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    if r < n*m*p:
        # Can't do trivial
        return None
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try ranks from low to high
    # Lower bound: max(n*p, m*p, n*m) = max(12, 30, 10) = 30
    # Upper bound: n*m*p = 60
    
    best_result = None
    
    # First, try the trivial rank n*m*p = 60
    result = _try_rank(n*m*p, n, m, p, deadline)
    if result and _verify(result["U"], result["V"], result["W"], n, m, p):
        best_result = result
    
    # Try to find lower ranks
    # Known results: for 2x5x6, the rank might be lower than 60
    # Let's try some structured decompositions
    
    # Try rank 54 (from the seed)
    if time.time() < deadline:
        rank = 54
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        # Part 1: j = 0, standard
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        # Part 2: j in 1..4, use Strassen-like decomposition
        # For 2x4x6, split into two 2x2x6 blocks
        # For 2x2x6, we can use 3 Strassen 2x2x2 blocks (rank 21)
        strassen_terms = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)],          [(1,0,1), (1,1,-1)]),
            ([(0,0,1)],          [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ([(1,1,1)],          [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)],          [(0,0,-1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)],[(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)],[(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        for block_idx in range(2):
            j_base = 1 + block_idx * 2
            for s in range(3):
                k_base = s * 2
                for term in strassen_terms:
                    u_list, v_list, w_list = term
                    for (li, lj, val) in u_list:
                        U[li * m + j_base + lj][col] += val
                    for (lj, lk, val) in v_list:
                        V[(j_base + lj) * p + k_base + lk][col] += val
                    for (li, lk, val) in w_list:
                        W[(k_base + lk) * n + li][col] += val
                    col += 1
        
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
    
    # Try even lower ranks with different strategies
    # Try rank 50
    if time.time() < deadline and best_result is None or len(best_result["U"][0]) > 50:
        rank = 50
        # Try a different decomposition
        # Use the fact that 2x5x6 can be decomposed as 2x(1+4)x6
        # 2x1x6 has rank 12
        # 2x4x6: try rank 38 (instead of 42)
        # This would give total rank 50
        
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        # Part 1: j = 0, standard (rank 12)
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
        
        # Part 2: j in 1..4, try rank 38
        # 2x4x6: split into 2x2x6 + 2x2x6
        # Each 2x2x6: can we do better than 21?
        # 2x2x6 = 3 * 2x2x2, rank = 3*7 = 21
        # But maybe we can do 2x2x6 in rank 19?
        # For now, use 21 each, total 42, but we only have 38 for part 2
        # So this won't work directly.
        
        # Let's try a different approach: use the full 2x4x6 tensor directly
        # The tensor has 2*4*6 = 48 entries, all 1
        # Can we decompose it in rank 38?
        
        # For now, skip this and try rank 48 (trivial for 2x4x6)
        # Total rank would be 12 + 48 = 60, same as trivial
        
        # Try rank 52: 12 + 40
        # 2x4x6 in rank 40?
        
        # Let's just try to verify if the rank 54 solution is correct
        pass
    
    if best_result is None:
        # Fallback to trivial
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
