"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _build_tensor(n=3, m=3, p=6):
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _als_decompose(T, r, max_iter=200, tol=1e-10, deadline=None, seed=0):
    """
    ALS decomposition of tensor T into U, V, W of rank r.
    T[a,b,c] = sum_l U[a,l] * V[b,l] * W[c,l]
    """
    if deadline is None:
        deadline = time.time() + 100
    
    rng = np.random.RandomState(seed)
    nm, mp, pn = T.shape
    
    # Initialize
    U = rng.randn(nm, r) * 0.1
    V = rng.randn(mp, r) * 0.1
    W = rng.randn(pn, r) * 0.1
    
    prev_err = float('inf')
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        # Update U: for each a, solve min_Ua ||T[a,:,:] - Ua @ (V*W)^T||_F
        new_U = np.zeros_like(U)
        for a in range(nm):
            # V*W has shape (mp, pn, r) -> we need (mp*pn, r)
            VW = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, r)
            target = T[a].reshape(mp * pn)
            try:
                new_U[a] = np.linalg.lstsq(VW, target, rcond=None)[0]
            except:
                new_U[a] = U[a]
        U = new_U
        
        # Update V
        new_V = np.zeros_like(V)
        for b in range(mp):
            UW = (U[:, :, None] * W[None, :, :]).reshape(nm * pn, r)
            target = T[:, b, :].reshape(nm * pn)
            try:
                new_V[b] = np.linalg.lstsq(UW, target, rcond=None)[0]
            except:
                new_V[b] = V[b]
        V = new_V
        
        # Update W
        new_W = np.zeros_like(W)
        for c in range(pn):
            UV = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, r)
            target = T[:, :, c].reshape(nm * mp)
            try:
                new_W[c] = np.linalg.lstsq(UV, target, rcond=None)[0]
            except:
                new_W[c] = W[c]
        W = new_W
        
        # Compute error
        err = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U[a], V[b] * W[c])
                    err += (val - T[a, b, c]) ** 2
        err = np.sqrt(err)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err


def _try_rank(T, r, deadline, max_restarts=3):
    """Try to find exact rank-r decomposition using ALS with multiple restarts."""
    nm, mp, pn = T.shape
    best_result = None
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        U, V, W, err = _als_decompose(T, r, max_iter=150, tol=1e-12, deadline=deadline, seed=restart)
        
        if err < 1e-8:
            # Verify exactly
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best_result


def _strassen_3x3_init(n=3, m=3, p=6):
    """
    Use Strassen's 7-multiplication algorithm for 3x3 * 3x3 as a base,
    then extend to 3x6 by handling the extra 3 columns.
    
    Strassen gives rank 7 for 3x3*3x3 (actually 19 for 3x3, but let's use the structure).
    
    Actually, for 3x3 * 3x3, the tensor rank is 19 (Landsberg). 
    For 3x3 * 3x6, we can think of it as two 3x3 * 3x3 multiplications,
    but they share the A factor, so we can potentially save.
    
    C = A * B where B = [B1 | B2] (each 3x3)
    C = [A*B1 | A*B2]
    
    Using Strassen for each: 7 + 7 = 14 multiplications, but A is shared.
    The Strassen intermediates for A*B1 and A*B2 share some structure.
    
    Actually, let's just use the fact that:
    C = A * B = A * [B1 | B2]
    
    We can write B as a 3x6 matrix. The tensor for 3x3*3x6 has rank at most
    2 * rank(3x3*3x3) = 2*19 = 38 if we just do two separate multiplications.
    
    But we can do better by sharing. Let's use the block structure.
    """
    pass


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    schoolbook_rank = n * m * p  # 54
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # For 3x3*3x6, the rank is known to be at most 36 (by doubling 3x3*3x3 rank 18? No, 19*2=38)
    # Actually, let's try from 18 upward
    
    best_rank = schoolbook_rank
    best_result = result
    
    # Try to find better decompositions
    for r in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        
        max_restarts = max(1, int((deadline - time.time()) / 20))
        result_r = _try_rank(T, r, deadline, max_restarts=max_restarts)
        
        if result_r is not None:
            best_rank = r
            best_result = result_r
            break  # Found the lowest rank, stop
    
    # If we found something better, use it
    if best_rank < schoolbook_rank:
        result = best_result
    
    # Ensure we return before deadline
    if time.time() > deadline - 5:
        return result
    
    return result
# EVOLVE-BLOCK-END
