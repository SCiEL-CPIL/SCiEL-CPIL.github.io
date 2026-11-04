import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p, atol=1e-9):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    if U_np.shape[1] != V_np.shape[1] or U_np.shape[1] != W_np.shape[1]:
        return False
        
    rank = U_np.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Check active entries: should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                if not np.isclose(val, 1.0, atol=atol):
                    return False
                
    # Check inactive entries: should be 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid.add((u, v, w))
                
    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                if (u, v, w) not in valid:
                    val = np.sum(U_np[u, :] * V_np[v, :] * W_np[w, :])
                    if not np.isclose(val, 0.0, atol=atol):
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7) each.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22) * (B11 + B22)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22) * B11
    U[1, 1] = 1; U[2, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11 * (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[2, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22 * (B21 - B11)
    U[2, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11 + A12) * B22
    U[0, 4] = 1; U[1, 4] = 1
    V[2, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    U[0, 5] = -1; U[1, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron(U1, U2, V1, V2, W1, W2):
    """Kronecker product of two decompositions."""
    r1, r2 = U1.shape[1], U2.shape[1]
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    return U, V, W

def _try_prune(U, V, W, n, m, p, deadline):
    """Try to remove columns from the decomposition while maintaining validity."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in range(rank):
            if time.time() > deadline - 2:
                break
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if _verify(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                improved = True
                break
    
    return U, V, W

def _random_search(n, m, p, target_rank, deadline, n_attempts=50):
    """Try random initialization with rounding to find a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    
    for _ in range(n_attempts):
        if time.time() > deadline - 1:
            break
            
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _construct_rank21():
    """
    Construct a rank-21 decomposition for 3x3 * 3x4 matrix multiplication.
    Uses the Kronecker product of Strassen's 2x2x2 (rank 7) with a 3x3x2 structure.
    """
    n, m, p = 3, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 9, 12, 12
    
    # Get Strassen decomposition for 2x2x2
    S_U, S_V, S_W = _strassen_2x2()
    
    # We need a rank-3 decomposition for 3x3 * 3x2? No, that's not right.
    # M_{3,3,4} is not simply M_{2,2,2} x M_{3,3,2}.
    # Let's try a different construction.
    
    # Actually, the rank of M_{3,3,4} is known to be at most 21.
    # One construction uses the following idea:
    # Partition the 3x3 matrix A into blocks and use Strassen-like formulas.
    
    # For now, let's return None and rely on other methods.
    return None

def _search_with_constraints(n, m, p, deadline):
    """
    Search for a low-rank decomposition using a constraint-based approach.
    We fix the structure of U and V to have sparse rows and solve for W.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Define the tensor entries
    tensor = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor[u, v, w] = 1.0
    
    # We want to find rank r such that tensor = sum_{l=1}^r u_l v_l^T w_l^T
    # This is equivalent to finding r vectors u_l, v_l, w_l such that
    # for all (u,v,w): tensor[u,v,w] = sum_l U[u,l] V[v,l] W[w,l]
    
    # Let's try to find a decomposition with small integer entries.
    # We'll search over possible values for U and V, then solve for W.
    
    # For each rank r, try to find U, V, W.
    # Start with r = 21 and go down.
    
    best = None
    
    for r in range(21, 35):
        if time.time() > deadline - 10:
            break
        
        # Try a few random sparse initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize U and V with sparse patterns
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            
            for l in range(r):
                # Choose 2-3 non-zero entries in U[:, l]
                num_u = np.random.choice([2, 3])
                idx_u = np.random.choice(nm, num_u, replace=False)
                U[idx_u, l] = np.random.choice([-1, 1], num_u)
                
                # Choose 2-3 non-zero entries in V[:, l]
                num_v = np.random.choice([2, 3])
                idx_v = np.random.choice(mp, num_v, replace=False)
                V[idx_v, l] = np.random.choice([-1, 1], num_v)
            
            # Now solve for W
            # We have equations: for each (u,v,w), tensor[u,v,w] = sum_l U[u,l] V[v,l] W[w,l]
            # This is a linear system in W.
            
            # Build the linear system
            # Number of equations: nm * mp * pn
            # Number of unknowns: pn * r
            
            # We'll use least squares to find W
            A = np.zeros((nm * mp * pn, pn * r))
            b = np.zeros(nm * mp * pn)
            
            eq_idx = 0
            for u in range(nm):
                for v in range(mp):
                    for w in range(pn):
                        for l in range(r):
                            A[eq_idx, w * r + l] = U[u, l] * V[v, l]
                        b[eq_idx] = tensor[u, v, w]
                        eq_idx += 1
            
            # Solve A W = b
            W_flat, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat.reshape(pn, r)
            
            # Check if the solution is valid
            if _verify(U, V, W, n, m, p, atol=1e-6):
                best = (r, U, V, W)
                break
        
        if best is not None and best[0] == r:
            break
    
    if best is not None:
        return best[1], best[2], best[3]
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to prune the schoolbook solution
    U_pruned, V_pruned, W_pruned = _try_prune(U, V, W, n, m, p, deadline - 20)
    
    if U_pruned.shape[1] < U.shape[1]:
        return {"U": U_pruned.tolist(), "V": V_pruned.tolist(), "W": W_pruned.tolist()}
    
    # Try constraint-based search
    if time.time() < deadline - 30:
        result = _search_with_constraints(n, m, p, deadline - 10)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Try random search for lower ranks
    for r in range(35, 20, -1):
        if time.time() > deadline - 10:
            break
        result = _random_search(n, m, p, r, deadline - 5, n_attempts=50)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
