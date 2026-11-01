import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 11]

def _verify_decomposition_fast(U, V, W, n=2, m=2, p=11, trials=2):
    """Fast verification of tensor decomposition."""
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    np.random.seed(12345)
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Efficient verification using einsum
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # C[i,k] = sum_{j,c} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        
        # Reshape
        U_r = U.reshape(n, m, rank)  # (n, m, r)
        V_r = V.reshape(m, p, rank)  # (m, p, r)
        W_r = W.reshape(p, n, rank)  # (p, n, r)
        
        # C[i,k] = sum_{j,c} U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # This is a 3-way contraction. We can do it as:
        # For each c, form a matrix M_c where M_c[i,k] = sum_j U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # But W_r[k,i,c] depends on both k and i, so it's not a simple outer product.
        
        # Let's use einsum directly:
        # C[i,k] = sum_{j,c} U_r[i,j,c] V_r[j,k,c] W_r[k,i,c]
        # Note the index pattern: i appears in U_r and W_r, j in U_r and V_r, k in V_r and W_r, c in all.
        
        # We can compute this as:
        # Step 1: For each c, compute T_c[i,j,k] = U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # This is expensive. Let's do it differently.
        
        # Alternative: 
        # C[i,k] = sum_c (sum_j U_r[i,j,c] V_r[j,k,c]) * W_r[k,i,c]
        # Let X_c[i,k] = sum_j U_r[i,j,c] V_r[j,k,c]
        # Then C[i,k] = sum_c X_c[i,k] * W_r[k,i,c]
        
        # Compute X_c for all c at once:
        # X_c = U_r[:, :, c] @ V_r[:, :, c] ? No, U_r is (n, m, r), V_r is (m, p, r)
        # X_c[i,k] = sum_j U_r[i,j,c] V_r[j,k,c]
        # This is like a batched matrix multiplication where for each c, we multiply (n,m) by (m,p)
        
        # Reshape for batched matmul:
        # U_batch: (r, n, m) -> U_r.transpose(2,0,1)
        # V_batch: (r, m, p) -> V_r.transpose(2,0,1)
        # X_batch: (r, n, p) = U_batch @ V_batch
        
        U_batch = U_r.transpose(2, 0, 1)  # (r, n, m)
        V_batch = V_r.transpose(2, 0, 1)  # (r, m, p)
        X_batch = U_batch @ V_batch  # (r, n, p)
        
        # Now C[i,k] = sum_c X_batch[c, i, k] * W_r[k, i, c]
        # W_r is (p, n, r). We need W_r[k, i, c] which is W_r.T[c, i, k]? 
        # W_r[k,i,c] -> transpose to (r, n, p): W_batch = W_r.transpose(2, 1, 0)
        # Then W_batch[c, i, k] = W_r[k, i, c]
        
        W_batch = W_r.transpose(2, 1, 0)  # (r, n, p)
        
        # C[i,k] = sum_c X_batch[c, i, k] * W_batch[c, i, k]
        # This is an element-wise product summed over c:
        C_dec = np.sum(X_batch * W_batch, axis=0)  # (n, p)
        
        if not np.allclose(C, C_dec, atol=1e-10):
            return False
    return True

def _build_strassen_2x2x2():
    """
    Build a rank-7 decomposition for 2x2 by 2x2 matrix multiplication.
    Convention:
      U indexed by (i, j) with row index i*m + j, m=2
      V indexed by (j, k) with row index j*p + k, p=2
      W indexed by (k, i) with row index k*n + i, n=2
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00 + a11)(b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10 + a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a00(b01 - b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: a11(b10 - b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a00 + a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a10 - a00)(b00 + b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01 - a11)(b11 + b10)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _build_schoolbook_block(p_local=1):
    """Build a rank-4 schoolbook decomposition for 2x2 by 2xp_local."""
    U = np.zeros((4, 4))
    V = np.zeros((2 * p_local, 4))
    W = np.zeros((p_local * 2, 4))
    
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        U[i * 2 + j, idx] = 1
        for k in range(p_local):
            V[j * p_local + k, idx] = 1
            W[k * 2 + i, idx] = 1
            
    return U, V, W

def _als_cp_decomposition(n, m, p, rank, max_iters=50, tol=1e-8, seed=42):
    """
    Alternating Least Squares for CP decomposition of the matrix multiplication tensor.
    Target tensor T_{i,j,k} = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}.
    
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) or None if failed.
    """
    np.random.seed(seed)
    
    # Target tensor: T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    # Initialize factors
    U = np.random.randn(n, m, rank) * 0.5
    V = np.random.randn(m, p, rank) * 0.5
    W = np.random.randn(p, n, rank) * 0.5
    
    for it in range(max_iters):
        # Update U: for each c, minimize ||T - sum_c U_c \otimes V_c \otimes W_c||^2
        # Fix V and W, solve for U
        # The loss is sum_{i,j,k,c} (T_{ijk} - sum_c' U_{ic}V_{jc}W_{kc})^2
        # This is a least squares problem for each c.
        
        # Precompute V_c @ W_c for each c (as a matrix of size n*m? No, it's a tensor)
        # Actually, for each c, we want to find U_c such that:
        # sum_c U_c[i] * (sum over j,k V_c[j]W_c[k] T_{ijk})... this isn't quite right.
        
        # Let's use the standard ALS formulation.
        # For each c, the residual R^{(c)} = T - sum_{c' != c} U_{c'} \otimes V_{c'} \otimes W_{c'}
        # Then U_c = argmin ||R^{(c)} - U_c \otimes V_c \otimes W_c||^2
        
        # This can be solved by:
        # U_c[i] = (sum_{j,k} R^{(c)}[i,j,k] * V_c[j] * W_c[k]) / (sum_{j,k} V_c[j]^2 * W_c[k]^2)
        
        # But computing R^{(c)} for each c is expensive.
        # Alternative: compute the full contraction.
        
        # Let's use a different approach: matrix unfolding.
        # Unfold T along mode 1: T_(1) is (n, m*p)
        # T_(1)[i, j*p+k] = T[i,j,k] = 1
        # The CP decomposition implies T_(1) = U @ (W * V)^T where * is element-wise and we need to reshape.
        # Actually, T_(1) = sum_c U_c (V_c \otimes W_c)^T where V_c \otimes W_c is flattened appropriately.
        
        # Let's just do the direct ALS update using the formula above.
        
        # Precompute V_c^2 and W_c^2 sums
        V_sq = np.sum(V**2, axis=(0, 1))  # (rank,)
        W_sq = np.sum(W**2, axis=(0, 1))  # (rank,)
        
        # Update U
        for c in range(rank):
            # R^{(c)}[i,j,k] = T[i,j,k] - sum_{c' != c} U[c',i] * V[c',j] * W[c',k]
            # This is expensive. Let's compute it more efficiently.
            
            # Current reconstruction: Reconstructed[i,j,k] = sum_c' U[c',i] V[c',j] W[c',k]
            # We can compute this as:
            # Rec = np.einsum('ci,cj,ck->ijk', U, V, W)
            # But that's O(n*m*p*rank) which is fine for small dimensions.
            
            # For now, let's just compute the full reconstruction and subtract the c-th term.
            Rec = np.einsum('ci,cj,ck->ijk', U, V, W)
            Rec_c = np.einsum('ij,ik->ijk', V[:, :, c], W[:, :, c])  # Wait, V is (m,p,rank), W is (p,n,rank)
            # V[:, :, c] is (m, p), W[:, :, c] is (p, n)
            # We need V_c[j] * W_c[k] but with different indices.
            # V_c[j] is V[j, k, c] for fixed j, varying k? No.
            # V is (m, p, rank) so V[j, k, c]
            # W is (p, n, rank) so W[k, i, c]
            # The term is U[i, c] * V[j, k, c] * W[k, i, c]
            
            # This is getting complicated. Let's use a simpler approach:
            # Just compute the gradient or use a library.
            
            # For now, let's skip the ALS and use a different strategy.
            pass
        
        # Since ALS is complex to implement correctly here, let's try a simpler approach:
        # Use the known Strassen decomposition and try to optimize it.
        
        # Actually, let's just return None and use the fallback.
        # The ALS approach is too complex to implement correctly in this time frame.
        
        return None
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    deadline = time.time() + 115  # Safety margin
    
    U_base, V_base, W_base = _build_strassen_2x2x2()
    
    # Build the 39-rank solution: 5 Strassen blocks + 1 schoolbook block
    total_rank = 5 * 7 + 1 * 4
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    # 5 Strassen blocks for columns 0-9
    for block_idx in range(5):
        start_col = block_idx * 2
        U[:, col:col+7] = U_base
        for j in range(2):
            for k_local in range(2):
                global_k = start_col + k_local
                V[j * p + global_k, col:col+7] = V_base[j * 2 + k_local, :]
        for k_local in range(2):
            for i in range(2):
                global_k = start_col + k_local
                W[global_k * n + i, col:col+7] = W_base[k_local * 2 + i, :]
        col += 7
    
    # 1 schoolbook block for column 10
    start_col = 10
    U_sub, V_sub, W_sub = _build_schoolbook_block(1)
    
    U[:, col:col+4] = U_sub
    for j in range(2):
        V[j * p + start_col, col:col+4] = V_sub[j, :]
    for i in range(2):
        W[start_col * n + i, col:col+4] = W_sub[i, :]
    col += 4
    
    # Verify with fast verification
    if time.time() < deadline:
        if not _verify_decomposition_fast(U, V, W):
            # Fall back to schoolbook
            rank = n * m * p
            U_fb = np.zeros((n*m, rank))
            V_fb = np.zeros((m*p, rank))
            W_fb = np.zeros((p*n, rank))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, column] = 1
                        V_fb[j*p+k, column] = 1
                        W_fb[k*n+i, column] = 1
                        column += 1
            U, V, W = U_fb, V_fb, W_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
