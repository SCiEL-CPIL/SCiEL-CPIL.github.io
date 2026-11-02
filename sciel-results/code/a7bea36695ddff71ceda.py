"""
Matrix multiplication 2x2 by 2x5: rank-minimal exact tensor decomposition.

Strategy:
1. Build the target tensor T of shape (4, 10, 10).
2. Try to find a rank-17 decomposition via:
   a. Structured initialization from the rank-18 Strassen+schoolbook base,
      removing one term and solving the resulting underdetermined system.
   b. ALS (Alternating Least Squares) with multiple random restarts.
   c. Gradient-based polish (L-BFGS-B) from ALS solutions.
   d. Semi-definite programming / convex relaxation approach.
3. If rank-17 fails, try rank-16 with similar methods.
4. Fall back to the known rank-18 decomposition.

The output uses the trace convention W[k*n+i, r].
"""
import numpy as np
import time
import sys
from itertools import combinations

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    return T

def _check_decomposition(U, V, W, T, atol=1e-7):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    R = U_a.shape[1]
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T, atol=atol)

def _compute_error(U, V, W, T, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    err = 0.0
    for r in range(R):
        prod = np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
        err += np.sum((prod - T)**2)
    return np.sqrt(err / T.size)

def _build_rank18(n, m, p):
    """Build the rank-18 decomposition: Strassen on first 2 cols + schoolbook on last 3."""
    rank = 18
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    def add_strassen_block(col0, col1, base):
        # 7 Strassen terms for multiplying 2x2 by 2x2 block
        # U indices: 0=i0j0, 1=i0j1, 2=i1j0, 3=i1j1
        # V indices: 0=j0k0, 1=j0k1, 2=j1k0, 3=j1k1 (within block)
        # W indices: 0=k0i0, 1=k0i1, 2=k1i0, 3=k1i1 (within block)
        
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[col0][r] = 1; V[m + col1][r] = 1
        W[col0*n][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[col1][r] = 1; V[m + col1][r] = -1
        W[col1*n][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[m + col0][r] = 1; V[col0][r] = -1
        W[col0*n][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[m + col1][r] = 1
        W[col0*n][r] = -1; W[col1*n][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[col0][r] = 1; V[col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[m + col0][r] = 1; V[m + col1][r] = 1
        W[col0*n][r] = 1
    
    # Strassen for columns 0,1
    add_strassen_block(0, 1, 0)
    # Strassen for columns 2,3
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    for i in range(n):
        for j in range(m):
            r = 14 + i*m + j
            U[i*m + j][r] = 1
            V[j*p + 4][r] = 1
            W[4*n + i][r] = 1
    
    return U, V, W

def _als_solve_U(T, V, W, R):
    """Solve for U given V and W."""
    nm = V.shape[0] * W.shape[0]  # not directly used
    nU = T.shape[0]
    nV = T.shape[1]
    nW = T.shape[2]
    
    # For each u, solve: sum_r U[u,r] * (V[:,r] outer W[:,r]) = T[u,:,:]
    # Flatten: U[u,:] @ (V*W)_flat = T[u,:,:]_flat
    # (V*W)_flat has shape (nV*nW, R)
    VW = np.einsum('vr,wr->vwr', V, W).reshape(nV*nW, R)
    
    U = np.zeros((nU, R), dtype=np.float64)
    for u in range(nU):
        b = T[u].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(VW, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, R):
    nU = T.shape[0]
    nV = T.shape[1]
    nW = T.shape[2]
    
    UW = np.einsum('ur,wr->uwr', U, W).reshape(nU*nW, R)
    
    V = np.zeros((nV, R), dtype=np.float64)
    for v in range(nV):
        b = T[:, v, :].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(UW, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, R):
    nU = T.shape[0]
    nV = T.shape[1]
    nW = T.shape[2]
    
    UV = np.einsum('ur,vr->uvr', U, V).reshape(nU*nV, R)
    
    W = np.zeros((nW, R), dtype=np.float64)
    for w in range(nW):
        b = T[:, :, w].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(UV, b, rcond=None)
        W[w] = sol
    return W

def _try_als(U_init, V_init, W_init, T, R, max_iters=200, tol=1e-12, deadline=None):
    """Run ALS from given initialization."""
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        U = _als_solve_U(T, V, W, R)
        V = _als_solve_V(T, U, W, R)
        W = _als_solve_W(T, U, V, R)
        
        err = _compute_error(U, V, W, T, R)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    if _check_decomposition(U, V, W, T):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _gradient_polish(U, V, W, T, R, deadline, max_iter=100):
    """Polish using L-BFGS-B."""
    from scipy.optimize import minimize
    
    nU = T.shape[0]
    nV = T.shape[1]
    nW = T.shape[2]
    
    def objective(params):
        U_p = params[:nU*R].reshape(nU, R)
        V_p = params[nU*R:nU*R+nV*R].reshape(nV, R)
        W_p = params[nU*R+nV*R:].reshape(nW, R)
        err = 0.0
        for r in range(R):
            prod = np.einsum('u,v,w->uvw', U_p[:,r], V_p[:,r], W_p[:,r])
            err += np.sum((prod - T)**2)
        return err
    
    def gradient(params):
        U_p = params[:nU*R].reshape(nU, R)
        V_p = params[nU*R:nU*R+nV*R].reshape(nV, R)
        W_p = params[nU*R+nV*R:].reshape(nW, R)
        
        grad_U = np.zeros_like(U_p)
        grad_V = np.zeros_like(V_p)
        grad_W = np.zeros_like(W_p)
        
        for r in range(R):
            prod = np.einsum('u,v,w->uvw', U_p[:,r], V_p[:,r], W_p[:,r])
            diff = 2.0 * (prod - T)
            # d/dU[u,r] = sum_{v,w} diff[u,v,w] * V[v,r] * W[w,r]
            grad_U[:,r] = np.einsum('uvw,vr,wr->u', diff, V_p, W_p)
            grad_V[:,r] = np.einsum('uvw,ur,wr->v', diff, U_p, W_p)
            grad_W[:,r] = np.einsum('uvw,ur,vr->w', diff, U_p, V_p)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': 1e-10, 'disp': False})
    except Exception:
        return None
    
    if res.fun < 1e-8:
        U_p = res.x[:nU*R].reshape(nU, R)
        V_p = res.x[nU*R:nU*R+nV*R].reshape(nV, R)
        W_p = res.x[nU*R+nV*R:].reshape(nW, R)
        
        if _check_decomposition(U_p, V_p, W_p, T):
            return U_p.tolist(), V_p.tolist(), W_p.tolist()
    return None

def _round_to_half_int(U, V, W, T):
    """Try to round to half-integers."""
    for denom in [2, 4, 8, 16, 32, 64]:
        U_s = np.round(np.array(U) * denom) / denom
        V_s = np.round(np.array(V) * denom) / denom
        W_s = np.round(np.array(W) * denom) / denom
        if _check_decomposition(U_s, V_s, W_s, T):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    return None

def _try_rank17(n, m, p, T, deadline):
    """Try to find rank-17 decomposition."""
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Strategy A: Remove each term from rank-18 and try ALS
    for term in range(18):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term]
        U_init = U18[:, idx].copy()
        V_init = V18[:, idx].copy()
        W_init = W18[:, idx].copy()
        
        # Try several perturbations
        for seed in range(8):
            if time.time() > deadline:
                break
            np.random.seed(term * 1000 + seed * 100)
            scale = 0.05 + 0.03 * seed
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_als(U_pert, V_pert, W_pert, T, 17, max_iters=150, deadline=deadline)
            if result is not None:
                return result
            
            # Gradient polish
            if time.time() < deadline - 10:
                result_g = _gradient_polish(U_pert, V_pert, W_pert, T, 17, deadline, max_iter=50)
                if result_g is not None:
                    return result_g
    
    # Strategy B: Random initializations
    for seed in range(60):
        if time.time() > deadline:
            break
        np.random.seed(seed * 31 + 777)
        scale = 0.3 + 0.1 * (seed % 5)
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
        
        result = _try_als(U_init, V_init, W_init, T, 17, max_iters=100, deadline=deadline)
        if result is not None:
            return result
    
    # Strategy C: Structured approach - fix U to have sparse structure
    # Try U with specific patterns
    for seed in range(20):
        if time.time() > deadline:
            break
        np.random.seed(seed * 17 + 42)
        # Create sparse U
        U_init = np.zeros((4, 17), dtype=np.float64)
        for r in range(17):
            # Each column of U has 2-3 nonzeros
            nnz = np.random.choice([2, 3], p=[0.5, 0.5])
            positions = np.random.choice(4, size=nnz, replace=False)
            for pos in positions:
                U_init[pos, r] = np.random.choice([-1, 1])
        
        V_init = np.random.randn(10, 17) * 0.5
        W_init = np.random.randn(10, 17) * 0.5
        
        result = _try_als(U_init, V_init, W_init, T, 17, max_iters=100, deadline=deadline)
        if result is not None:
            return result
    
    return None

def _try_rank16(n, m, p, T, deadline):
    """Try to find rank-16 decomposition."""
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Remove pairs of terms
    for t1 in range(18):
        for t2 in range(t1+1, 18):
            if time.time() > deadline:
                break
            idx = [i for i in range(18) if i != t1 and i != t2]
            U_init = U18[:, idx].copy()
            V_init = V18[:, idx].copy()
            W_init = W18[:, idx].copy()
            
            for seed in range(3):
                if time.time() > deadline:
                    break
                np.random.seed((t1*20 + t2) * 10 + seed)
                scale = 0.05 + 0.02 * seed
                U_pert = U_init + np.random.randn(4, 16) * scale
                V_pert = V_init + np.random.randn(10, 16) * scale
                W_pert = W_init + np.random.randn(10, 16) * scale
                
                result = _try_als(U_pert, V_pert, W_pert, T, 16, max_iters=80, deadline=deadline)
                if result is not None:
                    return result
    
    # Random inits for rank 16
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(seed * 23 + 999)
        scale = 0.3 + 0.1 * (seed % 4)
        U_init = np.random.randn(4, 16) * scale
        V_init = np.random.randn(10, 16) * scale
        W_init = np.random.randn(10, 16) * scale
        
        result = _try_als(U_init, V_init, W_init, T, 16, max_iters=80, deadline=deadline)
        if result is not None:
            return result
    
    return None

def _try_hadamard_structure(n, m, p, T, deadline):
    """
    Use Walsh-Hadamard structure on Z_2^2 to find a low-rank decomposition.
    
    The idea: the 2x2 matrix multiplication can be diagonalized using
    the Hadamard transform. For the 2x5 case, we exploit the group structure.
    """
    # Hadamard matrix for Z_2^2 (4x4)
    H = np.array([[1, 1, 1, 1],
                  [1, -1, 1, -1],
                  [1, 1, -1, -1],
                  [1, -1, -1, 1]], dtype=np.float64) / 2.0
    
    # Try different structured approaches
    # Approach: Use the spectral decomposition to guide the search
    
    # For 2x2 by 2x5, we can think of it as 5 independent 2x2 multiplications
    # but sharing the left matrix. Strassen gives 7 per 2x2 block.
    # For 5 blocks: naive Strassen gives 7*5 = 35 (worse than schoolbook 20)
    # Better: share Strassen factors across blocks.
    
    # The key insight: for 2x2 by 2x5, we can use:
    # - 7 Strassen terms for the first 2 columns
    # - 7 Strassen terms for the next 2 columns  
    # - 4 schoolbook terms for the last column
    # Total: 18 (our current best)
    
    # To get rank 17, we need to merge one schoolbook term into Strassen structure.
    # To get rank 16, we need to merge two.
    
    # Let's try a different structured approach:
    # Use the fact that the last column can be partially shared with Strassen terms.
    
    # Try: 7 Strassen + 7 Strassen + 3 modified terms = 17
    # The 3 modified terms handle the 5th column with some sharing.
    
    # Construct candidate with 17 terms where the last 3 terms are structured
    for variant in range(5):
        if time.time() > deadline:
            break
        
        U = np.zeros((4, 17), dtype=np.float64)
        V = np.zeros((10, 17), dtype=np.float64)
        W = np.zeros((10, 17), dtype=np.float64)
        
        # First 14 terms: two Strassen blocks
        U[:4, :7] = U18_local[:4, :7]
        V[:10, :7] = V18_local[:10, :7]
        W[:10, :7] = W18_local[:10, :7]
        U[:4, 7:14] = U18_local[:4, 7:14]
        V[:10, 7:14] = V18_local[:10, 7:14]
        W[:10, 7:14] = W18_local[:10, 7:14]
        
        # Last 3 terms: handle column 4 with sharing
        # Term 14: U[0,14]=1, V[4,14]=1, W[4,14]=1 (i=0,j=0,k=4)
        # Term 15: U[1,14]=1, V[9,14]=1, W[4,14]=1 (i=0,j=1,k=4) - shares W with 14
        # Term 16: U[2,15]=1, V[4,15]=1, W[9,15]=1 (i=1,j=0,k=4)
        # Term 17: U[3,15]=1, V[9,15]=1, W[9,15]=1 (i=1,j=1,k=4) - shares W with 16
        
        # Modified: try to merge terms
        if variant == 0:
            # Merge terms 14 and 15 partially
            U[0, 14] = 1; U[1, 14] = 1
            V[4, 14] = 1; V[9, 14] = 1
            W[4, 14] = 1
            # But this doesn't work directly...
            pass
        
        # Use ALS to fill in the last 3 terms
        U_init = U.copy()
        V_init = V.copy()
        W_init = W.copy()
        
        # Initialize last 3 terms with schoolbook values
        U_init[0, 14] = 1; V_init[4, 14] = 1; W_init[4, 14] = 1
        U_init[1, 15] = 1; V_init[9, 15] = 1; W_init[4, 15] = 1
        U_init[2, 16] = 1; V_init[4, 16] = 1; W_init[9, 16] = 1
        # We need one more for U[3], V[9], W[9]
        # But we only have 17 terms... 
        # This means we can't do pure schoolbook for col 4 with 3 terms.
        # We need to modify the Strassen terms to help.
        
        # Perturb and run ALS
        np.random.seed(variant * 100 + 1)
        scale = 0.1
        U_pert = U_init + np.random.randn(4, 17) * scale
        V_pert = V_init + np.random.randn(10, 17) * scale
        W_pert = W_init + np.random.randn(10, 17) * scale
        
        result = _try_als(U_pert, V_pert, W_pert, T, 17, max_iters=100, deadline=deadline)
        if result is not None:
            return result
    
    return None

# Store reference to rank-18 for use in hadamard function
U18_local, V18_local, W18_local = None, None, None

def construct():
    global U18_local, V18_local, W18_local
    
    n, m, p = DIMENSIONS
    deadline = time.time() + 125  # Safety margin before 150s kill
    
    T = _build_target()
    
    # Build rank-18 baseline
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_local, V18_local, W18_local = U18, V18, W18
    
    # Verify rank-18 works
    if not _check_decomposition(U18, V18, W18, T):
        # Fallback: pure schoolbook
        U_s = np.zeros((4, 20), dtype=np.float64)
        V_s = np.zeros((10, 20), dtype=np.float64)
        W_s = np.zeros((10, 20), dtype=np.float64)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, r] = 1
                    V_s[j*p+k, r] = 1
                    W_s[k*n+i, r] = 1
                    r += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try rank-17
    result17 = _try_rank17(n, m, p, T, deadline)
    if result17 is not None:
        U, V, W = result17
        if _check_decomposition(U, V, W, T):
            # Try to round
            rounded = _round_to_half_int(U, V, W, T)
            if rounded is not None:
                return {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
            return {"U": U, "V": V, "W": W}
    
    # Try Hadamard-structured approach
    if time.time() < deadline - 20:
        result_h = _try_hadamard_structure(n, m, p, T, deadline)
        if result_h is not None:
            U, V, W = result_h
            if _check_decomposition(U, V, W, T):
                rounded = _round_to_half_int(U, V, W, T)
                if rounded is not None:
                    return {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
                return {"U": U, "V": V, "W": W}
    
    # Try rank-16
    if time.time() < deadline - 15:
        result16 = _try_rank16(n, m, p, T, deadline)
        if result16 is not None:
            U, V, W = result16
            if _check_decomposition(U, V, W, T):
                rounded = _round_to_half_int(U, V, W, T)
                if rounded is not None:
                    return {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
                return {"U": U, "V": V, "W": W}
    
    # Return rank-18
    return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
