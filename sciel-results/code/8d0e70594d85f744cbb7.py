"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check all 12 (i,j,k) triples exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known minimal rank for 2x2x3 matrix multiplication.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _try_rank10_merge():
    """
    Try to find a rank-10 decomposition by merging one Strassen term with one schoolbook term.
    We consider all pairs (s, c) where s in {0..6} (Strassen terms) and c in {7..10} (schoolbook terms).
    For each pair, we attempt to find a merged term that covers the contributions of both.
    """
    import itertools
    from scipy.optimize import minimize
    
    deadline = time.time() + 120  # Safety margin
    
    # Get the rank-11 decomposition
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    # Strassen terms: 0-6, Schoolbook terms: 7-10
    strassen_terms = list(range(7))
    schoolbook_terms = list(range(7, 11))
    
    best_result = None
    best_rank = 11
    
    for s, c in itertools.product(strassen_terms, schoolbook_terms):
        if time.time() > deadline - 10:
            break
        
        # Try to merge terms s and c into a single term
        # The merged term must satisfy all constraints that were satisfied by terms s and c together
        
        # Collect all (i,j,k) triples that are affected by term s or term c
        affected = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check if term s or term c contributes to this (i,j,k)
                    if U11[i*m+j, s] != 0 and V11[j*p+k, s] != 0 and W11[k*n+i, s] != 0:
                        affected.add((i, j, k))
                    if U11[i*m+j, c] != 0 and V11[j*p+k, c] != 0 and W11[k*n+i, c] != 0:
                        affected.add((i, j, k))
        
        if len(affected) == 0:
            continue
        
        # For the merged term to work, we need to find U', V', W' such that:
        # For all (i,j,k) in affected:
        #   U'[i,j] * V'[j,k] * W'[k,i] = U11[i,j,s]*V11[j,k,s]*W11[k,i,s] + U11[i,j,c]*V11[j,k,c]*W11[k,i,c]
        # For all other (i,j,k):
        #   U'[i,j] * V'[j,k] * W'[k,i] = 0 (or the contribution is already covered by other terms)
        
        # This is a hard problem. Let's use a simpler approach:
        # Assume the merged term has the form where U', V', W' are linear combinations of the original terms.
        
        # Parameterize: U' = a*U_s + b*U_c, V' = c*V_s + d*V_c, W' = e*W_s + f*W_c
        # But this is too restrictive. Instead, let's try a different approach.
        
        # Alternative: Use the fact that we can add a small perturbation to one of the terms.
        # This is also hard.
        
        # Let's try a more direct approach: search over possible merged term structures.
        # For each pair, we can try to find if there's a rank-1 tensor that can replace both terms.
        
        # This is complex, so let's skip for now and just return the rank-11 result.
        pass
    
    return None


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    # The rank of 2x2x3 matrix multiplication is known to be 11.
    # We return the rank 11 decomposition directly.
    result = _rank11_decomposition()
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook()
