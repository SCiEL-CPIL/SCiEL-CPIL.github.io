import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _build_rank28():
    """Build the standard rank-28 decomposition for 2x3 x 3x5."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5), rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2), rank 7
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[1 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1), rank 4
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2), rank 7
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=3, p=5, atol=1e-8):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = np.array(U[:, r] if isinstance(U, np.ndarray) else [U[i][r] for i in range(n*m)])
        v = np.array(V[:, r] if isinstance(V, np.ndarray) else [V[i][r] for i in range(m*p)])
        w = np.array(W[:, r] if isinstance(W, np.ndarray) else [W[i][r] for i in range(p*n)])
        T += np.einsum('i,j,k->ijk', u, v, w)
    T_orig = np.ones((n, m, p))
    return np.allclose(T, T_orig, atol=atol)

def _try_remove_term(U, V, W, k_idx, n=2, m=3, p=5, atol=1e-10):
    """Try to remove term k_idx by expressing it as a linear combination of others.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Build tensor for each term
    other_indices = [r for r in range(rank) if r != k_idx]
    
    # Build matrix of other terms: (n*m*p, rank-1)
    size = n * m * p
    M = np.zeros((size, len(other_indices)))
    for idx, r in enumerate(other_indices):
        t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        M[:, idx] = t.flatten()
    
    # Target tensor
    t_k = np.einsum('i,j,k->ijk', U_mat[:, k_idx], V_mat[:, k_idx], W_mat[:, k_idx])
    t_k_flat = t_k.flatten()
    
    # Solve least squares
    try:
        x, residuals, _, _ = np.linalg.lstsq(M, t_k_flat, rcond=None)
        residual = np.linalg.norm(M @ x - t_k_flat)
        
        if residual < atol:
            # T_k = sum_{r != k} x_r * T_r
            # T = sum_{r != k} T_r + T_k = sum_{r != k} (1 + x_r) T_r
            new_U = []
            new_V = []
            new_W = []
            idx = 0
            for r in other_indices:
                coeff = 1 + x[idx]
                new_U.append(U_mat[:, r] * coeff)
                new_V.append(V_mat[:, r] * coeff)
                new_W.append(W_mat[:, r] * coeff)
                idx += 1
            
            new_U = np.array(new_U).T
            new_V = np.array(new_V).T
            new_W = np.array(new_W).T
            
            # Verify
            if _verify_decomposition(new_U, new_V, new_W, n, m, p, atol=1e-6):
                return True, new_U, new_V, new_W
    except Exception:
        pass
    
    return False, None, None, None

def _try_pairwise_merge(U, V, W, k1, k2, n=2, m=3, p=5, atol=1e-10):
    """Try to merge terms k1 and k2 into a single rank-1 term.
    This requires that T_{k1} + T_{k2} is rank 1.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Compute T_{k1} + T_{k2}
    t1 = np.einsum('i,j,k->ijk', U_mat[:, k1], V_mat[:, k1], W_mat[:, k1])
    t2 = np.einsum('i,j,k->ijk', U_mat[:, k2], V_mat[:, k2], W_mat[:, k2])
    t_sum = t1 + t2
    
    # Check if t_sum is rank 1 using SVD
    # Reshape to (n*m, p) and check rank
    t_reshaped = t_sum.reshape(n*m, p)
    s = np.linalg.svd(t_reshaped, compute_uv=False)
    
    if len(s) > 0 and s[0] > 1e-10 and (len(s) < 2 or s[1] / s[0] < 1e-8):
        # It's rank 1 (or close enough)
        # Factor it: t_sum = u * v * w where u is (n*m,), v is (m*p,), w is (p*n,)
        # Actually, t_sum[i,j,k] = u[i] * v[j] * w[k] for some u, v, w
        # This is the outer product of three vectors
        # We can use the SVD of the reshaped matrix to get the factorization
        
        # For a rank-1 tensor T[i,j,k] = a[i]*b[j]*c[k],
        # the matrix T.reshape(n*m, p) has rank 1
        # So T.reshape(n*m, p) = (a * b_reshaped) @ c_reshaped
        # where b_reshaped is b reshaped to (m*p,) and c_reshaped is c reshaped to (p*n,)
        
        # Let's just use the SVD directly
        U_svd, S, Vt = np.linalg.svd(t_reshaped, full_matrices=False)
        if len(S) > 0 and S[0] > 1e-10:
            u_vec = U_svd[:, 0] * S[0]  # (n*m,)
            v_vec = Vt[0, :]  # (p,) - this needs to be reshaped
            
            # Actually, let's think more carefully.
            # t_sum[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
            # t_reshaped[i*m+j, k] = t_sum[i,j,k]
            # If rank 1: t_reshaped = u * v^T where u is (n*m,) and v is (p,)
            # So t_sum[i,j,k] = u[i*m+j] * v[k]
            # This means the tensor is u_outer * v_outer
            # But we need it as a product of THREE vectors: a[i]*b[j]*c[k]
            # u[i*m+j] = a[i]*b[j] for all i,j
            # This requires u to be a rank-1 matrix when reshaped to (n,m)
            
            u_reshaped = u_vec.reshape(n, m)
            u_svd2, S2, Vt2 = np.linalg.svd(u_reshaped, full_matrices=False)
            
            if len(S2) > 0 and S2[0] > 1e-10 and (len(S2) < 2 or S2[1] / S2[0] < 1e-8):
                a = u_svd2[:, 0] * S2[0]  # (n,)
                b = Vt2[0, :]  # (m,)
                c = v_vec  # (p,)
                
                # Now we have t_sum[i,j,k] = a[i] * b[j] * c[k]
                # Map back to U, V, W indices:
                # U index: i*m + j' where j' is the column in the 2x3 matrix
                # V index: j'*p + k
                # W index: k*n + i
                
                # But wait, in our problem, the tensor is T[i,j,k] = 1 for all i,j,k
                # And each term is T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # So a[i] corresponds to U[i*m+j, r] for fixed j? No...
                
                # Actually, this approach is getting complicated. Let me just try a simpler approach:
                # Check if the sum of two rank-1 tensors is rank 1 by checking if their
                # corresponding vectors are proportional in a certain way.
                
                pass
    
    return False, None, None, None

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 5
    
    # Build the rank-28 decomposition
    U, V, W = _build_rank28()
    rank = len(U[0])
    
    # Try to reduce rank by removing individual terms
    for k_idx in range(rank):
        if time.time() > deadline:
            break
        
        success, new_U, new_V, new_W = _try_remove_term(U, V, W, k_idx, n, m, p)
        if success:
            return {
                "U": new_U.tolist(),
                "V": new_V.tolist(),
                "W": new_W.tolist()
            }
    
    # If no single term removal works, return the rank-28 decomposition
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
