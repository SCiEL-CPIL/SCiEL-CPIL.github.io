"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 140.0  # safety margin before 150s kill

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    """Schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2x8_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x8 multiplication.
    Each term is (U_dict, V_dict, W_dict) where indices are local:
    U: (i,j) for i,j in {0,1}
    V: (j,k) for j in {0,1}, k in 0..7
    W: (k,i) for k in 0..7, i in {0,1}
    """
    p = 8
    terms = []
    # M1 = (A00+A11)(B0+B1) -> C00 += M1, C11 += M1
    terms.append(
        ({(0,0): 1, (1,1): 1},
         {(0,k): 1 for k in range(p)} | {(1,k): 1 for k in range(p)},
         {(k,0): 1 for k in range(p)} | {(k,1): 1 for k in range(p)}))
    # M2 = (A10+A11)B0 -> C10 += M2
    terms.append(
        ({(1,0): 1, (1,1): 1},
         {(0,k): 1 for k in range(p)},
         {(k,1): 1 for k in range(p)}))
    # M3 = A00(B1-B0) -> C10 += M3
    terms.append(
        ({(0,0): 1},
         {(1,k): 1 for k in range(p)} | {(0,k): -1 for k in range(p)},
         {(k,1): 1 for k in range(p)}))
    # M4 = A11(B0-B1) -> C01 += M4
    terms.append(
        ({(1,1): 1},
         {(0,k): 1 for k in range(p)} | {(1,k): -1 for k in range(p)},
         {(k,0): 1 for k in range(p)}))
    # M5 = (A00+A01)B1 -> C01 += M5
    terms.append(
        ({(0,0): 1, (0,1): 1},
         {(1,k): 1 for k in range(p)},
         {(k,0): 1 for k in range(p)}))
    # M6 = (A10-A00)(B0+B1) -> C11 += M6
    terms.append(
        ({(1,0): 1, (0,0): -1},
         {(0,k): 1 for k in range(p)} | {(1,k): 1 for k in range(p)},
         {(k,1): 1 for k in range(p)}))
    # M7 = (A01-A11)(B1+B0) -> C00 += M7
    terms.append(
        ({(0,1): 1, (1,1): -1},
         {(1,k): 1 for k in range(p)} | {(0,k): 1 for k in range(p)},
         {(k,0): 1 for k in range(p)}))
    return terms

def _build_strassen_3x3x8():
    """
    Strassen-based decomposition for 3x3 * 3x8 with rank 12.
    
    Uses Strassen on A[0:2,0:2] * B[0:2,:] (7 terms) + 5 cross terms.
    """
    n, m, p = 3, 3, 8
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # Strassen 2x2 * 2x8 on A[0:2,0:2] * B[0:2,:]
    s_terms = _strassen_2x2x2x8_terms()
    for Ud, Vd, Wd in s_terms:
        U_new = {u_idx(i, j): val for (i, j), val in Ud.items()}
        V_new = {v_idx(j, k): val for (j, k), val in Vd.items()}
        W_new = {w_idx(k, i): val for (k, i), val in Wd.items()}
        terms.append((U_new, V_new, W_new))
    
    # Cross terms (scalar * vector, each is 1 rank-1 term covering all 8 k's)
    # A[0,2] * B[2,k] -> C[0,k]
    terms.append(({u_idx(0,2): 1},
                  {v_idx(2,k): 1 for k in range(p)},
                  {w_idx(k,0): 1 for k in range(p)}))
    # A[1,2] * B[2,k] -> C[1,k]
    terms.append(({u_idx(1,2): 1},
                  {v_idx(2,k): 1 for k in range(p)},
                  {w_idx(k,1): 1 for k in range(p)}))
    # A[2,0] * B[0,k] -> C[2,k]
    terms.append(({u_idx(2,0): 1},
                  {v_idx(0,k): 1 for k in range(p)},
                  {w_idx(k,2): 1 for k in range(p)}))
    # A[2,1] * B[1,k] -> C[2,k]
    terms.append(({u_idx(2,1): 1},
                  {v_idx(1,k): 1 for k in range(p)},
                  {w_idx(k,2): 1 for k in range(p)}))
    # A[2,2] * B[2,k] -> C[2,k]
    terms.append(({u_idx(2,2): 1},
                  {v_idx(2,k): 1 for k in range(p)},
                  {w_idx(k,2): 1 for k in range(p)}))
    
    r = len(terms)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c in range(r):
        Ud, Vd, Wd = terms[c]
        for idx, val in Ud.items():
            U[idx][c] = val
        for idx, val in Vd.items():
            V[idx][c] = val
        for idx, val in Wd.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}

def _build_two_strassen():
    """
    Try two overlapping Strassen blocks to reduce rank below 12.
    
    Strategy: Use Strassen on A[0:2,0:2]*B[0:2,:] AND A[1:3,1:3]*B[1:3,:].
    These overlap at A[1,1] and B[1,:]. Need to handle the overlap carefully.
    
    Block 1 (rows 0,1 of A; cols 0,1 of B): 7 terms
    Block 2 (rows 1,2 of A; cols 1,2 of B): 7 terms
    Overlap: A[1,1]*B[1,k] is in both blocks.
    
    In Block 1, A[1,1]*B[1,k] is handled by Strassen terms.
    In Block 2, A[1,1]*B[1,k] is also handled.
    
    If we simply add both blocks, we double-count A[1,1]*B[1,k].
    We need to subtract it once, which requires additional terms.
    
    Alternatively, use a partition approach:
    - A[0:2,0:2] * B[0:2,:] via Strassen: 7 terms
    - A[0:2,2] * B[2,:] : 2 terms
    - A[2,0:2] * B[0:2,:] : 2 terms  
    - A[2,2] * B[2,:] : 1 term
    Total: 12 (same as before)
    
    Try a different partition:
    - A[0:2,0:2] * B[0:2,:] via Strassen: 7 terms
    - A[0:2,2:3] * B[2:3,:] : 2 terms (A[0,2]*B[2,:], A[1,2]*B[2,:])
    - A[2:3,0:2] * B[0:2,:] : 2 terms (A[2,0]*B[0,:], A[2,1]*B[1,:])
    - A[2,2] * B[2,:] : 1 term
    Total: 12
    
    To go below 12, we need to find structure that allows sharing.
    """
    return None  # No improvement found with this approach

def _try_polynomial_encoding():
    """
    Try encoding the 8 columns as polynomial coefficients.
    
    Idea: Encode B[j,k] as coefficient of x^k in polynomial B_j(x) mod f(x)
    where f(x) is irreducible of degree 8. Then A*B becomes polynomial
    multiplication in GF(2)[x]/f(x) or similar.
    
    However, this requires the 3x3 matrix multiplication to work over
    polynomials, which doesn't directly reduce the tensor rank of the
    scalar multiplication problem.
    
    This approach is not straightforward for reducing rank below 12.
    """
    return None

def _als_decompose(target_tensor, max_rank, n_iter=100, seed=42):
    """
    Try ALS (Alternating Least Squares) to find a low-rank decomposition.
    target_tensor: 3D array of shape (n*m, m*p, p*n)
    """
    rng = np.random.RandomState(seed)
    n, m, p = DIMENSIONS
    U_dim, V_dim, W_dim = n*m, m*p, p*n
    
    for r in range(max_rank, 11, -1):
        U = rng.randn(U_dim, r)
        V = rng.randn(V_dim, r)
        W = rng.randn(W_dim, r)
        
        for it in range(n_iter):
            # Update U: for each (i,j), solve U[i*m+j,:] = target[i*m+j,:,:] * (V⊙W)^+
            # target[i*m+j, v, w] = sum_c U[i*m+j,c] * V[v,c] * W[w,c]
            # So U[i*m+j,:] should satisfy: sum_c U[i*m+j,c]*V[v,c]*W[w,c] = T[i*m+j,v,w]
            # This is: (V ⊙ W)^T * U[i*m+j,:]^T = T[i*m+j, :, :].T
            # where ⊙ is Khatri-Rao product
            
            for idx in range(U_dim):
                # Build Khatri-Rao product of V and W: (V_dim*W_dim, r)
                # Actually: for each c, the vector is V[:,c] ⊗ W[:,c]
                # We need: sum_c U[idx,c] * (V[:,c] ⊗ W[:,c]) = T[idx].flatten()
                # So: [V[:,0]⊗W[:,0], ..., V[:,r-1]⊗W[:,r-1]] @ U[idx,:] = T[idx].flatten()
                # This is a Khatri-Rao product
                
                if r <= 1:
                    U[idx, 0] = 1.0
                    continue
                
                # Build the Khatri-Rao matrix (V_dim * W_dim, r)
                # K[v*W_dim + w, c] = V[v,c] * W[w,c]
                K = np.einsum('vc,wc->vwc', V, W).reshape(V_dim * W_dim, r)
                target_flat = target_tensor[idx].flatten()
                # Least squares: K @ U[idx,:] = target_flat
                try:
                    U[idx, :] = np.linalg.lstsq(K, target_flat, rcond=None)[0]
                except:
                    U[idx, :] = rng.randn(r)
            
            # Similarly update V and W
            for idx in range(V_dim):
                K = np.einsum('uc,wc->uwc', U, W).reshape(U_dim * W_dim, r)
                target_flat = target_tensor[:, idx, :].transpose(1, 0).flatten()
                # target_tensor[u, idx, w] for all u, w
                target_flat = target_tensor[:, idx, :].reshape(-1)
                # Need: sum_c V[idx,c] * U[u,c] * W[w,c] = T[u, idx, w]
                # K[u*W_dim+w, c] = U[u,c] * W[w,c]
                K = np.einsum('uc,wc->uwc', U, W).reshape(U_dim * W_dim, r)
                target_flat = target_tensor[:, idx, :].reshape(-1)
                try:
                    V[idx, :] = np.linalg.lstsq(K, target_flat, rcond=None)[0]
                except:
                    V[idx, :] = rng.randn(r)
            
            for idx in range(W_dim):
                # W_dim = p*n, idx = k*n+i
                k = idx // n
                i = idx % n
                # Need: sum_c U[u,c] * V[v,c] * W[idx,c] = T[u, v, idx]
                # K[u*V_dim+v, c] = U[u,c] * V[v,c]
                K = np.einsum('uc,vc->uvc', U, V).reshape(U_dim * V_dim, r)
                target_flat = target_tensor[:, :, idx].reshape(-1)
                try:
                    W[idx, :] = np.linalg.lstsq(K, target_flat, rcond=None)[0]
                except:
                    W[idx, :] = rng.randn(r)
        
        # Check if this solution is valid
        U_list = [[float(U[i, c]) for c in range(r)] for i in range(U_dim)]
        V_list = [[float(V[i, c]) for c in range(r)] for i in range(V_dim)]
        W_list = [[float(W[i, c]) for c in range(r)] for i in range(W_dim)]
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _build_target_tensor():
    """Build the 3D target tensor for 3x3*3x8."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try Strassen-based decomposition (rank 12)
    if _time_left() > 5.0:
        cand = _build_strassen_3x3x8()
        if cand is not None:
            r = len(cand["U"][0])
            if r < best_rank and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
                best = cand
                best_rank = r
    
    # Try ALS to find rank < 12
    if _time_left() > 30.0:
        target = _build_target_tensor()
        # Try ranks 11 down to 9
        for target_rank in [11, 10, 9]:
            if _time_left() < 20.0:
                break
            for seed in [42, 123, 777, 2024]:
                if _time_left() < 15.0:
                    break
                cand = _als_decompose(target, target_rank, n_iter=50, seed=seed)
                if cand is not None:
                    r = len(cand["U"][0])
                    if r < best_rank:
                        best = cand
                        best_rank = r
                        break
            if best_rank <= target_rank:
                break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
