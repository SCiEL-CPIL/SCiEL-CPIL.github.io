import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_fast(U, V, W):
    """Fast vectorized verification using einsum."""
    n, m, p = 2, 4, 5
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(T, 1.0, atol=1e-8))

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_als_perturbation(U, V, W, target_rank, time_limit=30.0, seed=0, n_trials=20):
    """
    Perturb the decomposition and run ALS to try to drive one rank-1 term toward zero.
    Uses a smooth surrogate: minimize the sum of squared residuals while encouraging sparsity.
    """
    n, m, p = 2, 4, 5
    T = np.ones((n, m, p))
    start = time.time()
    deadline = start + time_limit
    rng = np.random.RandomState(seed)

    best = None
    best_rank = U.shape[1]
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Start from the current best and perturb
        U_p = U.copy() + 1e-3 * rng.randn(*U.shape)
        V_p = V.copy() + 1e-3 * rng.randn(*V.shape)
        W_p = W.copy() + 1e-3 * rng.randn(*W.shape)

        # Run ALS with a focus on driving one column to zero
        for it in range(300):
            if time.time() > deadline - 2:
                break
            
            # ALS update for U
            for i in range(n):
                for j in range(m):
                    num = np.zeros(U_p.shape[1])
                    den = np.zeros(U_p.shape[1])
                    for k in range(p):
                        vw = V_p[j, :] * W_p[k, i]
                        num += T[i, j, k] * vw
                        den += vw ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    U_p[i * m + j, :] = num / den
            
            # ALS update for V
            for j in range(m):
                for k in range(p):
                    num = np.zeros(V_p.shape[1])
                    den = np.zeros(V_p.shape[1])
                    for i in range(n):
                        uw = U_p[i * m + j, :] * W_p[k, i]
                        num += T[i, j, k] * uw
                        den += uw ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    V_p[j * p + k, :] = num / den
            
            # ALS update for W
            for k in range(p):
                for i in range(n):
                    num = np.zeros(W_p.shape[1])
                    den = np.zeros(W_p.shape[1])
                    for j in range(m):
                        uv = U_p[i * m + j, :] * V_p[j * p + k, :]
                        num += T[i, j, k] * uv
                        den += uv ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    W_p[k * n + i, :] = num / den

            # Check if any column is near zero
            norms = np.linalg.norm(np.vstack([U_p, V_p, W_p]), axis=0)
            min_norm = np.min(norms)
            
            if min_norm < 1e-3:
                idx = int(np.argmin(norms))
                mask = [r for r in range(U_p.shape[1]) if r != idx]
                U35 = U_p[:, mask]
                V35 = V_p[:, mask]
                W35 = W_p[:, mask]
                if _verify_fast(U35, V35, W35):
                    if U35.shape[1] < best_rank:
                        best = (U35, V35, W35)
                        best_rank = U35.shape[1]
                break

        if best is not None and best_rank <= 34:
            break

    return best

def _try_structural_merge(U, V, W, time_limit=20.0):
    """
    Try to find structural merges where two rank-1 terms can be combined into one.
    This is a heuristic search for linear dependencies in the tensor structure.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    start = time.time()
    deadline = start + time_limit
    
    best = None
    best_rank = rank
    
    # Try removing pairs of columns and checking if the remaining can still span
    for i in range(rank):
        if time.time() > deadline - 5:
            break
        for j in range(i+1, rank):
            if time.time() > deadline - 5:
                break
            
            mask = [k for k in range(rank) if k != i and k != j]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify_fast(U_new, V_new, W_new):
                if U_new.shape[1] < best_rank:
                    best = (U_new, V_new, W_new)
                    best_rank = U_new.shape[1]
                break
        if best is not None and best_rank <= 34:
            break
    
    return best

def _try_greedy_removal(U, V, W, time_limit=10.0):
    """
    Greedily remove columns one at a time if possible.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    start = time.time()
    deadline = start + time_limit
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    rank_curr = U_curr.shape[1]
    
    improved = True
    while improved:
        improved = False
        for col in range(rank_curr):
            if time.time() > deadline - 2:
                break
            mask = [i for i in range(rank_curr) if i != col]
            U_new = U_curr[:, mask]
            V_new = V_curr[:, mask]
            W_new = W_curr[:, mask]
            if _verify_fast(U_new, V_new, W_new):
                U_curr, V_curr, W_curr = U_new, V_new, W_new
                rank_curr = U_curr.shape[1]
                improved = True
                break
    
    if rank_curr < rank:
        return (U_curr, V_curr, W_curr)
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE

    U, V, W = _build_strassen_36()

    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]

    # Phase 1: Greedy single column removal
    if time.time() < deadline - 30:
        result = _try_greedy_removal(U, V, W, time_limit=min(10.0, deadline - time.time() - 20))
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                rank = U_r.shape[1]

    # Phase 2: Pair removal (structural merge)
    if time.time() < deadline - 20:
        result = _try_structural_merge(best_U, best_V, best_W, time_limit=min(15.0, deadline - time.time() - 10))
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                rank = U_r.shape[1]

    # Phase 3: ALS perturbation to find border-rank decompositions
    if time.time() < deadline - 10:
        result = _try_als_perturbation(best_U, best_V, best_W, best_U.shape[1], 
                                       time_limit=min(25.0, deadline - time.time() - 5), 
                                       seed=42, n_trials=15)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                rank = U_r.shape[1]

    # Final verification
    if not _verify_fast(best_U, best_V, best_W):
        # Fallback to the known rank-36
        best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
