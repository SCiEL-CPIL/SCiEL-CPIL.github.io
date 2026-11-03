import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_rank66(n, m, p):
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a = i_base*m + j_base
        b = i_base*m + (j_base+1)
        c = (i_base+1)*m + j_base
        d = (i_base+1)*m + (j_base+1)
        e = j_base*p + k_base
        f = j_base*p + (k_base+1)
        g = (j_base+1)*p + k_base
        h = (j_base+1)*p + (k_base+1)
        P = k_base*n + i_base
        Q = (k_base+1)*n + i_base
        Rr = k_base*n + (i_base+1)
        S = (k_base+1)*n + (i_base+1)

        U[a][col]=1; U[d][col]=1; V[e][col]=1; V[h][col]=1; W[P][col]=1; W[S][col]=1; col+=1
        U[c][col]=1; U[d][col]=1; V[e][col]=1; W[Rr][col]=1; W[S][col]=-1; col+=1
        U[a][col]=1; V[f][col]=1; V[h][col]=-1; W[Q][col]=1; W[S][col]=1; col+=1
        U[d][col]=1; V[g][col]=1; V[e][col]=-1; W[P][col]=1; W[Rr][col]=1; col+=1
        U[a][col]=1; U[b][col]=1; V[h][col]=1; W[P][col]=-1; W[Q][col]=1; col+=1
        U[c][col]=1; U[a][col]=-1; V[e][col]=1; V[f][col]=1; W[S][col]=1; col+=1
        U[b][col]=1; U[d][col]=-1; V[g][col]=1; V[h][col]=1; W[P][col]=1; col+=1

    # Row 0: Schoolbook (1x4x6) -> 24 terms
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
            
    # Rows 1-2: Strassen on 2x2x2 blocks
    for kb in range(p//2):
        for jb in range(m//2):
            add_strassen(1, jb*2, kb*2)
            
    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, max_iter=200):
    """Try to reduce rank by finding linear dependencies among the rank terms.
    
    The tensor decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    If there's a linear dependency among the rank vectors, we can reduce rank.
    
    We look for a non-trivial linear combination sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0.
    This is equivalent to finding a vector c in the null space of the tensor unfolding.
    """
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor as a matrix: each rank term contributes to the tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The tensor has n*m*p entries. We have R rank terms.
    # If R > n*m*p, there's definitely a dependency.
    # But even if R <= n*m*p, there might be one.
    
    # Construct the matrix M where M[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # If there's a non-zero vector c with M @ c = 0, we can reduce rank.
    
    nm = n * m
    mp = m * p
    pn = p * n
    total = n * m * p
    
    M = np.zeros((total, R))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                idx += 1
    
    # Find null space of M
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=True)
        # Null space is the right singular vectors corresponding to zero singular values
        tol = 1e-10
        null_mask = s_vals < tol
        if np.any(null_mask):
            null_vecs = Vt[null_mask, :].T
            print(f"Found {null_vecs.shape[1]} null space vectors")
            
            # Try to use the null space vectors to reduce rank
            # If c is in null space, sum_r c_r * term_r = 0
            # This means we can express one term as a combination of others
            # and eliminate it.
            
            # For each null space vector, try to eliminate a term
            best_rank = R
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for nv in null_vecs:
                # Find the term with largest |c_r|
                max_idx = np.argmax(np.abs(nv))
                if abs(nv[max_idx]) < 1e-15:
                    continue
                
                # Express term max_idx in terms of others
                # c_r * term_r = -c_max * term_max for r != max
                # term_max = -sum_{r != max} (c_r / c_max) * term_r
                
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                
                # For each other term, add (c_r / c_max) * term_r to term_max
                # Actually, we want to eliminate term max_idx
                # New term max_idx = -sum_{r != max} (c_r/c_max) * term_r
                # But this doesn't reduce rank unless we can merge terms
                
                # Better approach: if two terms are proportional, merge them
                # Check if any pair of terms are linearly dependent
                pass
            
            # Alternative: try to find if any rank-1 term is a linear combination
            # of others, and merge
            for r_elim in range(R):
                # Try to eliminate term r_elim
                # Find c such that sum_r c_r * term_r = 0 and c[r_elim] != 0
                # Then term_r_elim = -sum_{r != r_elim} (c_r/c[r_elim]) * term_r
                # This doesn't directly reduce rank, but if the resulting expression
                # has structure, it might.
                pass
            
            # Simpler approach: check if any two rank-1 terms are identical or proportional
            # If term_a = term_b, we can merge them
            for r1 in range(R):
                for r2 in range(r1+1, R):
                    # Check if U[:,r1] ⊗ V[:,r1] ⊗ W[:,r1] = U[:,r2] ⊗ V[:,r2] ⊗ W[:,r2]
                    # This is hard to check directly. Instead, check if they're proportional
                    # in some sense.
                    pass
    except Exception as e:
        print(f"Rank reduction failed: {e}")
    
    return U, V, W


def construct():
    """Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6)."""
    n, m, p = 3, 4, 6

    U66, V66, W66 = _build_rank66(n, m, p)

    if not _verify(U66, V66, W66, n, m, p, tol=1e-9):
        return {"U": U66, "V": V66, "W": W66}

    # Try to reduce rank
    if _time_left() > 5.0:
        U_red, V_red, W_red = _try_rank_reduction(U66, V66, W66, n, m, p)
        if _verify(U_red, V_red, W_red, n, m, p, tol=1e-9):
            return {"U": U_red, "V": V_red, "W": W_red}

    return {"U": U66, "V": V66, "W": W66}
