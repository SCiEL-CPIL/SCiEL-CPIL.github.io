import time
import numpy as np
from scipy.optimize import least_squares

DEADLINE = 120.0
START = time.time()

def _time_left():
    return DEADLINE - (time.time() - START)

def _build_schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # P1 = (A11+A22)(B11+B22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (A21+A22)A11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # P3 = A11(B12-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # P4 = A22(B21-B11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # P5 = (A11+A12)B22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[3,4]=1
    # P6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7 = (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _embed(t_U, t_V, t_W, n_src, m_src, p_src, n_tgt, m_tgt, p_tgt, n_map, m_map, p_map):
    """Embed a small tensor into a larger space."""
    rank = t_U.shape[1]
    U = np.zeros((n_tgt * m_tgt, rank))
    V = np.zeros((m_tgt * p_tgt, rank))
    W = np.zeros((p_tgt * n_tgt, rank))
    for i in range(n_src):
        for j in range(m_src):
            U[n_map[i]*m_tgt + m_map[j], :] = t_U[i*m_src+j, :]
    for j in range(m_src):
        for k in range(p_src):
            V[m_map[j]*p_tgt + p_map[k], :] = t_V[j*p_src+k, :]
    for k in range(p_src):
        for i in range(n_src):
            W[p_map[k]*n_tgt + n_map[i], :] = t_W[k*n_src+i, :]
    return U, V, W

def _tensor_sum(U_list, V_list, W_list, n, m, p):
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def _build_50_rank():
    """Build the rank-50 decomposition (Strassen-based)."""
    S_U, S_V, S_W = _build_strassen_222()
    
    # Build 2x4x4: four 2x2x2 Strassen blocks
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        for p_off in [0, 2]:
            u, v, w = _embed(S_U, S_V, S_W, 2,2,2, 2,4,4,
                             [0,1], [m_off, m_off+1], [p_off, p_off+1])
            U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U244, V244, W244 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,4)
    
    # Build 2x4x3: two 2x2x3 blocks, each = Strassen(2x2x2) + schoolbook(2x2x1)
    sch_U, sch_V, sch_W = _build_schoolbook(2, 2, 1)
    # 2x2x3 = embed Strassen into p=[0,1] + embed schoolbook into p=[2]
    u_s, v_s, w_s = _embed(S_U, S_V, S_W, 2,2,2, 2,2,3, [0,1],[0,1],[0,1])
    u_sc, v_sc, w_sc = _embed(sch_U, sch_V, sch_W, 2,2,1, 2,2,3, [0,1],[0,1],[2])
    U223 = np.hstack([u_s, u_sc])
    V223 = np.hstack([v_s, v_sc])
    W223 = np.hstack([w_s, w_sc])
    
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        u, v, w = _embed(U223, V223, W223, 2,2,3, 2,4,3,
                         [0,1], [m_off, m_off+1], [0,1,2])
        U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U243, V243, W243 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,3)
    
    # 2x4x7 = 2x4x4 (p=[0..3]) + 2x4x3 (p=[4..6])
    u1, v1, w1 = _embed(U244, V244, W244, 2,4,4, 2,4,7,
                        [0,1],[0,1,2,3],[0,1,2,3])
    u2, v2, w2 = _embed(U243, V243, W243, 2,4,7,
                        [0,1],[0,1,2,3],[4,5,6])
    U = np.hstack([u1, u2])
    V = np.hstack([v1, v2])
    W = np.hstack([w1, w2])
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=7):
    """Check decomposition correctness on random inputs."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    C_calc = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_calc[i, k] = s
    return np.allclose(C_calc, C_true, atol=1e-8)

def _try_rank_lsq(r, n=2, m=4, p=7, seed=0):
    """
    Try to find a rank-r decomposition using Levenberg-Marquardt.
    Objective: minimize sum_{i,j,k} (sum_{r} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    """
    rng = np.random.RandomState(seed)
    
    # Initialize near the rank-50 solution if possible, otherwise random
    if r >= 50:
        U0, V0, W0 = _build_50_rank()
        U0 = U0[:, :r]
        V0 = V0[:, :r]
        W0 = W0[:, :r]
    else:
        # Random initialization with some structure
        U0 = rng.randn(n*m, r) * 0.5
        V0 = rng.randn(m*p, r) * 0.5
        W0 = rng.randn(p*n, r) * 0.5
    
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    # Precompute index triples for vectorization
    # We want to evaluate: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can reshape U, V, W to (n, m, r), (m, p, r), (p, n, r)
    # Then T[i,j,k] = np.einsum('ijr,jkr,kin->ik', U, V, W) -- wait, indices are tricky.
    # U is (n*m, r). Reshape to (n, m, r).
    # V is (m*p, r). Reshape to (m, p, r).
    # W is (p*n, r). Reshape to (p, n, r).
    
    def residuals(params):
        if _time_left() < 5:
            return np.zeros(n*m*p) # Early exit signal? No, just return current
        
        U_flat = params[:n*m*r].reshape(n, m, r)
        V_flat = params[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
        W_flat = params[n*m*r + m*p*r :].reshape(p, n, r)
        
        # Compute tensor product
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_flat, V_flat, W_flat)
        
        return (T - 1.0).flatten()
    
    def objective(params):
        res = residuals(params)
        return np.sum(res**2)
    
    # Initial parameters
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Run least_squares
    try:
        result = least_squares(
            residuals, 
            x0, 
            method='lm', 
            max_nfev=1000,
            xtol=1e-10,
            ftol=1e-10
        )
        
        # Extract best solution
        x_best = result.x
        U_best = x_best[:n*m*r].reshape(n*m, r)
        V_best = x_best[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_best = x_best[n*m*r + m*p*r :].reshape(p*n, r)
        
        # Check convergence
        if result.success or np.max(np.abs(residuals(x_best))) < 1e-6:
            return U_best, V_best, W_best
        else:
            # If not converged, return the best found anyway to check if it's close
            return U_best, V_best, W_best
            
    except Exception:
        return None, None, None

def construct():
    t0 = time.time()
    
    # First, get the known rank-50 solution
    U50, V50, W50 = _build_50_rank()
    if not _verify(U50, V50, W50):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(2, 4, 7)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U50, V50, W50
    best_rank = 50
    
    # Try to find rank 49 using LM
    if _time_left() > 30:
        for seed in range(5):
            if _time_left() < 25:
                break
            U49, V49, W49 = _try_rank_lsq(49, seed=seed)
            if U49 is not None and _verify(U49, V49, W49):
                best_U, best_V, best_W = U49, V49, W49
                best_rank = 49
                break
    
    # Try rank 48 if time permits
    if best_rank == 49 and _time_left() > 30:
        for seed in range(5):
            if _time_left() < 25:
                break
            U48, V48, W48 = _try_rank_lsq(48, seed=seed)
            if U48 is not None and _verify(U48, V48, W48):
                best_U, best_V, best_W = U48, V48, W48
                best_rank = 48
                break
                
    # Try rank 47 if time permits (long shot)
    if best_rank == 48 and _time_left() > 30:
        for seed in range(3):
            if _time_left() < 25:
                break
            U47, V47, W47 = _try_rank_lsq(47, seed=seed)
            if U47 is not None and _verify(U47, V47, W47):
                best_U, best_V, best_W = U47, V47, W47
                best_rank = 47
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
