"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _quantize_half_integer(mat):
    """Quantize matrix to nearest half-integers."""
    return np.round(np.array(mat, dtype=np.float64) * 2) / 2.0

def _check_half_integer(mat):
    """Check if all entries are half-integers."""
    scaled = np.array(mat, dtype=np.float64) * 2
    return np.max(np.abs(scaled - np.round(scaled))) < 1e-9

def _try_als_decomposition(U_init, V_init, W_init, n, m, p, T_flat, max_iters=10, tol=1e-8):
    """
    Perform ALS iterations to refine a decomposition.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    T_flat: (pn, nm*mp) - flattened target tensor
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = U_init.shape[1]
    
    U = np.array(U_init, dtype=np.float64).copy()
    V = np.array(V_init, dtype=np.float64).copy()
    W = np.array(W_init, dtype=np.float64).copy()
    
    for it in range(max_iters):
        # Fix U, V; solve for W
        # T_flat[k, i, j] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For fixed U, V: T_flat = W @ (V^T * U) reshaped
        # B_flat[r, (i,j)] = U[i,r] * V[j,r]
        B_flat = np.zeros((rank, nm * mp))
        for r in range(rank):
            B_flat[r] = np.outer(U[:, r], V[:, r]).flatten()
        
        try:
            W_sol, _, _, _ = np.linalg.lstsq(B_flat.T, T_flat.T, rcond=1e-12)
            W_new = W_sol.T
        except:
            break
        
        # Fix U, W; solve for V
        # T_flat[k, i, j] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For fixed U, W: reshape T_flat to (pn, nm, mp)
        T_reshaped = T_flat.reshape(pn, nm, mp)
        # For each j: sum_r U[i,r] * W[k,r] * V[j,r] = T[k,i,j]
        # Let C[i,k,r] = U[i,r] * W[k,r], then sum_r C[i,k,r] * V[j,r] = T[k,i,j]
        # This is a linear system for V
        # Vectorize: for each (k,i), we have: V @ C_vec = T[k,i,:]
        # Better: build design matrix for V
        # D[j, (k,i,r)] ... this is complex. Let's do it differently.
        
        # Actually, let's use the same approach: fix W, solve for V given U
        # T_flat[k, i, j] = sum_r (U[i,r] * W[k,r]) * V[j,r]
        # Let A[i,k,r] = U[i,r] * W[k,r]
        # Then for each j: sum_r A[i,k,r] * V[j,r] = T[k,i,j]
        # Reshape: for fixed j, we have a linear system in V[j,:]
        # sum_r (A[i,k,r] * V[j,r]) = T[k,i,j]
        # This means: for each (k,i), V[j,:] dot A[:,k,i,:] = T[k,i,j]
        # So: V[j,:] @ A_reshaped.T = T_reshaped.T[:, j] where A_reshaped has shape (nm*pn, rank)
        
        A = np.zeros((nm, pn, rank))
        for r in range(rank):
            A[:, :, r] = np.outer(U[:, r], W[:, r])
        A_flat = A.reshape(nm * pn, rank)
        
        try:
            V_sol, _, _, _ = np.linalg.lstsq(A_flat.T, T_reshaped.T.reshape(mp, -1).T, rcond=1e-12)
            V_new = V_sol.T
        except:
            break
        
        # Fix V, W; solve for U
        # T_flat[k, i, j] = sum_r U[i,r] * (V[j,r] * W[k,r])
        # Let B2[j,k,r] = V[j,r] * W[k,r]
        # For each i: sum_r U[i,r] * B2[j,k,r] = T[k,i,j]
        # Reshape: for fixed i, U[i,:] @ B2_flat.T = T[:, i, :]
        
        B2 = np.zeros((mp, pn, rank))
        for r in range(rank):
            B2[:, :, r] = np.outer(V[:, r], W[:, r])
        B2_flat = B2.reshape(mp * pn, rank)
        
        try:
            U_sol, _, _, _ = np.linalg.lstsq(B2_flat.T, T_reshaped.T.reshape(pn, nm, mp).T.reshape(nm, -1).T, rcond=1e-12)
            U_new = U_sol.T
        except:
            break
        
        # Check convergence
        change = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        
        if change < tol:
            break
    
    return U, V, W

def _try_rank(target_rank, n, m, p, T_flat, deadline, max_inits=20):
    """Try to find a valid decomposition of given rank using ALS with quantization."""
    if time.time() > deadline - 15:
        return None
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Try multiple random initializations
    for init in range(max_inits):
        if time.time() > deadline - 10:
            break
        
        # Random half-integer initialization with small values
        # Use a mix of strategies
        strategy = init % 4
        
        if strategy == 0:
            # Sparse random: mostly zeros with some ±1, ±0.5
            U = np.random.choice([0, 0, 0, 1, -1, 0.5, -0.5], size=(nm, target_rank))
            V = np.random.choice([0, 0, 0, 1, -1, 0.5, -0.5], size=(mp, target_rank))
            W = np.random.choice([0, 0, 0, 1, -1, 0.5, -0.5], size=(pn, target_rank))
        elif strategy == 1:
            # Dense small values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
        elif strategy == 2:
            # Structured: use block patterns
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            for r in range(target_rank):
                # Randomly assign non-zero entries
                u_idx = np.random.choice(nm, size=min(3, nm), replace=False)
                v_idx = np.random.choice(mp, size=min(3, mp), replace=False)
                w_idx = np.random.choice(pn, size=min(3, pn), replace=False)
                U[u_idx, r] = np.random.choice([-1, 1])
                V[v_idx, r] = np.random.choice([-1, 1])
                W[w_idx, r] = np.random.choice([-1, 1])
        else:
            # Gaussian-like but quantized
            U = np.round(np.random.randn(nm, target_rank) * 0.7) / 2
            V = np.round(np.random.randn(mp, target_rank) * 0.7) / 2
            W = np.round(np.random.randn(pn, target_rank) * 0.7) / 2
        
        # Run ALS refinement
        U_ref, V_ref, W_ref = _try_als_decomposition(U, V, W, n, m, p, T_flat, max_iters=8)
        
        # Quantize to half-integers
        U_q = _quantize_half_integer(U_ref)
        V_q = _quantize_half_integer(V_ref)
        W_q = _quantize_half_integer(W_ref)
        
        # Check if quantized solution works
        if _verify(U_q.tolist(), V_q.tolist(), W_q.tolist(), n, m, p, target_rank):
            return U_q.tolist(), V_q.tolist(), W_q.tolist()
        
        # Also try without ALS, just quantize the initial
        if _verify(_quantize_half_integer(U).tolist(), _quantize_half_integer(V).tolist(), _quantize_half_integer(W).tolist(), n, m, p, target_rank):
            return _quantize_half_integer(U).tolist(), _quantize_half_integer(V).tolist(), _quantize_half_integer(W).tolist()
    
    return None

def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try ranks from 119 down to 100 first (more likely to succeed)
    for target_rank in range(119, 99, -1):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(target_rank, n, m, p, T_flat, deadline, max_inits=15)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            # Continue trying lower ranks
            break
    
    # If we found a rank, try to go lower
    if best_rank < 119:
        for target_rank in range(best_rank - 1, 99, -1):
            if time.time() > deadline - 10:
                break
            result = _try_rank(target_rank, n, m, p, T_flat, deadline, max_inits=10)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                print(f"Found valid rank {best_rank}")
                break
    
    return best_result
# EVOLVE-BLOCK-END
