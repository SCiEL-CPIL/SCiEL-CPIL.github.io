"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import scipy
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16
SCHOOBOOK_RANK = N * M * P  # 32

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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
    return U, V, W

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    # Correct Strassen terms for M_{2,2,2}
    # u, v, w are vectors of length 4 (flattened 2x2 matrices: [00, 01, 10, 11])
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # 4 blocks of 2x2 in the 2x8 matrix
    # B is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    # C is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U
            # U index is i (0..3) corresponding to A_{i//2, i%2}
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V
            # v is indexed by local (j, k) where j in 0..1, k in 0..1
            # Global B index: b = j * p + k_global
            # k_global = 2 * block_idx + k_local
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            # Fill W
            # w is indexed by local (i, k) where i in 0..1, k in 0..1
            # Global C index: c = k_global * n + i
            # k_global = 2 * block_idx + k_local
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def optimize_rank_decomposition(rank, time_limit=10.0):
    """
    Attempt to find a rank-`rank` decomposition using numerical optimization.
    Returns (U, V, W) if a valid exact decomposition is found, else None.
    """
    import numpy as np
    import random
    
    n, m, p = N, M, P
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    best_error = float('inf')
    best_solution = None
    
    start_time = time.time()
    restarts = 0
    
    while time.time() - start_time < time_limit:
        restarts += 1
        # Random initialization
        # Scale factors to keep numbers reasonable
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        def objective(params):
            U_flat = params[:nm*rank].reshape(nm, rank)
            V_flat = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_flat = params[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Compute tensor sum
            # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Use einsum for speed
            T_calc = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            error = np.sum((T_calc - T)**2)
            return error
            
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead for small problems, but it's slow.
        # Let's use L-BFGS-B if we can compute gradients, but gradients are complex.
        # Given the small size, Nelder-Mead might work with few iterations.
        # Or just simple gradient descent with finite differences? Too slow.
        
        # Let's try a few steps of gradient descent with random restarts.
        # Actually, for this specific task, random search with rounding might be better if the solution is integer.
        # But the hint says convex optimization.
        
        # Let's try scipy.optimize.minimize with Nelder-Mead
        try:
            result = scipy.optimize.minimize(objective, params0, method='Nelder-Mead', 
                                             options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_error:
                best_error = result.fun
                if result.fun < 1e-9:
                    U_sol = result.x[:nm*rank].reshape(nm, rank)
                    V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Round to nearest half-integer
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Convert to list of lists of Fractions or floats for verify
                    U_list = [[float(x) for x in row] for row in U_int]
                    V_list = [[float(x) for x in row] for row in V_int]
                    W_list = [[float(x) for x in row] for row in W_int]
                    
                    if verify(U_list, V_list, W_list):
                        return U_list, V_list, W_list
                        
        except Exception:
            pass

    return None

def construct():
    """Entry point: return the best decomposition found."""
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    # Fallback solution
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best_U, best_V, best_W = U, V, W
        else:
            best_U, best_V, best_W = schoolbook()
    except Exception:
        best_U, best_V, best_W = schoolbook()

    # Try to find lower rank solutions
    # We have ~110 seconds left
    time_remaining = deadline - time.time()
    
    # Try rank 24, 25, 26, 27
    # Allocate time based on likelihood
    ranks_to_try = [24, 25, 26, 27]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5: # Safety margin
            break
        
        # Allocate 10s per rank
        time_limit = min(10.0, deadline - time.time() - 5)
        if time_limit <= 0:
            break
            
        try:
            result = optimize_rank_decomposition(r, time_limit)
            if result:
                U_new, V_new, W_new = result
                # Verify again to be safe
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    # If we found rank 24, we can stop or try 23?
                    # Let's stop if we hit 24, as it's a significant jump.
                    if r <= 24:
                        break
        except Exception:
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
