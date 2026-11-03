"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product as iproduct

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 120.0

    def elapsed():
        return time.time() - start_time

    def remaining():
        return max(0.0, deadline - elapsed())

    # ---------- Tensor verification ----------
    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Check if U,V,W correctly decompose the matmul tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        """
        r = U_mat.shape[1]
        # Build the tensor from factors
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_fast(U_mat, V_mat, W_mat, n, m, p, tol=1e-8):
        """Vectorized verification using einsum-like approach."""
        r = U_mat.shape[1]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_r = U_mat.reshape(n, m, r)
        V_r = V_mat.reshape(m, p, r)
        W_r = W_mat.reshape(p, n, r)
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use np.einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)

    # ---------- Build initial Strassen-based decomposition ----------
    def build_strassen_decomposition():
        """Build the 3x5x6 matmul tensor using Strassen on 2x2x2 blocks."""
        # Partition:
        # n=3: [2, 1]
        # m=5: [2, 2, 1]
        # p=6: [2, 2, 2]
        
        n_blocks = [(0, 2), (2, 3)]
        m_blocks = [(0, 2), (2, 4), (4, 5)]
        p_blocks = [(0, 2), (2, 4), (4, 6)]
        
        terms = []  # Each term: (list of (U_row, coeff), list of (V_row, coeff), list of (W_row, coeff))
        
        def idx_U(i, j): return i * m + j
        def idx_V(j, k): return j * p + k
        def idx_W(k, i): return k * n + i
        
        def add_strassen(n_s, n_e, m_s, m_e, p_s, p_e):
            """Add 7 Strassen terms for a 2x2x2 sub-block."""
            # Local indices
            n0, n1 = n_s, n_s + 1
            m0, m1 = m_s, m_s + 1
            p0, p1 = p_s, p_s + 1
            
            # M1 = (A[n0,m0]+A[n1,m1]) * (B[m0,p0]+B[m1,p1])
            terms.append((
                [(idx_U(n0, m0), 1), (idx_U(n1, m1), 1)],
                [(idx_V(m0, p0), 1), (idx_V(m1, p1), 1)],
                [(idx_W(p0, n0), 1), (idx_W(p1, n1), 1)]
            ))
            
            # M2 = (A[n1,m0]+A[n1,m1]) * B[m0,p0]
            terms.append((
                [(idx_U(n1, m0), 1), (idx_U(n1, m1), 1)],
                [(idx_V(m0, p0), 1)],
                [(idx_W(p0, n1), 1), (idx_W(p1, n1), -1)]
            ))
            
            # M3 = A[n0,m0] * (B[m0,p1]-B[m1,p1])
            terms.append((
                [(idx_U(n0, m0), 1)],
                [(idx_V(m0, p1), 1), (idx_V(m1, p1), -1)],
                [(idx_W(p1, n0), 1), (idx_W(p1, n1), 1)]
            ))
            
            # M4 = A[n1,m1] * (B[m1,p0]-B[m0,p0])
            terms.append((
                [(idx_U(n1, m1), 1)],
                [(idx_V(m1, p0), 1), (idx_V(m0, p0), -1)],
                [(idx_W(p0, n0), 1), (idx_W(p0, n1), 1)]
            ))
            
            # M5 = (A[n0,m0]+A[n0,m1]) * B[m1,p1]
            terms.append((
                [(idx_U(n0, m0), 1), (idx_U(n0, m1), 1)],
                [(idx_V(m1, p1), 1)],
                [(idx_W(p1, n0), -1), (idx_W(p1, n1), 1)]
            ))
            
            # M6 = (A[n1,m0]-A[n0,m0]) * (B[m0,p0]+B[m0,p1])
            terms.append((
                [(idx_U(n1, m0), 1), (idx_U(n0, m0), -1)],
                [(idx_V(m0, p0), 1), (idx_V(m0, p1), 1)],
                [(idx_W(p1, n1), 1)]
            ))
            
            # M7 = (A[n0,m1]-A[n1,m1]) * (B[m1,p0]+B[m1,p1])
            terms.append((
                [(idx_U(n0, m1), 1), (idx_U(n1, m1), -1)],
                [(idx_V(m1, p0), 1), (idx_V(m1, p1), 1)],
                [(idx_W(p0, n0), 1)]
            ))
        
        def add_schoolbook(n_s, n_e, m_s, m_e, p_s, p_e):
            """Add schoolbook terms for a block."""
            for i in range(n_s, n_e):
                for j in range(m_s, m_e):
                    for k in range(p_s, p_e):
                        terms.append((
                            [(idx_U(i, j), 1)],
                            [(idx_V(j, k), 1)],
                            [(idx_W(k, i), 1)]
                        ))
        
        for (ns, ne) in n_blocks:
            for (ms, me) in m_blocks:
                for (ps, pe) in p_blocks:
                    nsz = ne - ns
                    msz = me - ms
                    psz = pe - ps
                    if nsz == 2 and msz == 2 and psz == 2:
                        add_strassen(ns, ne, ms, me, ps, pe)
                    else:
                        add_schoolbook(ns, ne, ms, me, ps, pe)
        
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
            for row, coeff in u_entries:
                U[row, ci] += coeff
            for row, coeff in v_entries:
                V[row, ci] += coeff
            for row, coeff in w_entries:
                W[row, ci] += coeff
        
        return U, V, W, rank

    # ---------- Rank reduction via column elimination ----------
    def try_eliminate_columns(U, V, W, n, m, p, max_time=100.0):
        """Try to eliminate linearly dependent or redundant columns."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start = time.time()
        
        improved = True
        while improved:
            improved = False
            for col in range(rank):
                if time.time() - start > max_time:
                    return U, V, W
                if rank <= 1:
                    break
                U_test = np.delete(U, col, axis=1)
                V_test = np.delete(V, col, axis=1)
                W_test = np.delete(W, col, axis=1)
                if verify_fast(U_test, V_test, W_test, n, m, p, tol=1e-8):
                    U, V, W = U_test, V_test, W_test
                    rank -= 1
                    improved = True
                    break
        return U, V, W

    # ---------- Smooth optimization for rank reduction ----------
    def smooth_rank_reduce(U, V, W, n, m, p, max_time=80.0):
        """Use smooth optimization to find a lower-rank decomposition.
        
        We parameterize a potentially lower-rank decomposition and minimize
        the tensor reconstruction error. If we can achieve zero error with
        fewer factors, we've found a lower-rank decomposition.
        """
        U0, V0, W0 = U.copy(), V.copy(), W.copy()
        rank0 = U0.shape[1]
        start = time.time()
        
        # Try reducing by 1, 2, 3... columns
        for target_rank in range(rank0 - 1, max(20, rank0 - 10), -1):
            if time.time() - start > max_time:
                break
            if remaining() < 15.0:
                break
            
            # Initialize with a subset of columns
            # Try different subsets
            n_attempts = 5
            for attempt in range(n_attempts):
                if time.time() - start > max_time or remaining() < 10.0:
                    break
                
                # Select target_rank columns from the original
                if attempt == 0:
                    cols = list(range(target_rank))
                else:
                    # Random subset
                    cols = np.random.choice(rank0, target_rank, replace=False).tolist()
                
                U_t = U0[:, cols].copy()
                V_t = V0[:, cols].copy()
                W_t = W0[:, cols].copy()
                
                # Smooth optimization: minimize ||T(U,V,W) - T_true||^2
                # using gradient descent on the factor matrices
                lr = 0.01
                n_iter = 200
                best_loss = float('inf')
                
                for it in range(n_iter):
                    if time.time() - start > max_time or remaining() < 5.0:
                        break
                    
                    # Compute tensor and gradient
                    U_r = U_t.reshape(n, m, target_rank)
                    V_r = V_t.reshape(m, p, target_rank)
                    W_r = W_t.reshape(p, n, target_rank)
                    
                    # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                    loss = np.sum((T - 1.0) ** 2)
                    
                    if loss < best_loss:
                        best_loss = loss
                        if best_loss < 1e-12:
                            return U_t, V_t, W_t
                    
                    # Gradients (approximate)
                    # dL/dU[i,j,c] = 2*(T[i,j,k]-1) * V[j,k,c] * W[k,i,c] summed over k
                    dU_r = np.einsum('ijk,jkc,kic->ijc', T - 1.0, V_r, W_r) * 2.0
                    dV_r = np.einsum('ijk,ijc,kic->jkc', T - 1.0, U_r, W_r) * 2.0
                    dW_r = np.einsum('ijk,ijc,jkc->kic', T - 1.0, U_r, V_r) * 2.0
                    
                    U_t -= lr * dU_r.reshape(n*m, target_rank)
                    V_t -= lr * dV_r.reshape(m*p, target_rank)
                    W_t -= lr * dW_r.reshape(p*n, target_rank)
                
                # Check if we achieved a valid decomposition
                if best_loss < 1e-10:
                    return U_t, V_t, W_t
        
        return U0, V0, W0

    # ---------- Main construction ----------
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify initial decomposition
    if not verify_fast(U, V, W, n, m, p, tol=1e-8):
        # Fallback: pure schoolbook
        U_s = np.zeros((n*m, n*m*p))
        V_s = np.zeros((m*p, n*m*p))
        W_s = np.zeros((p*n, n*m*p))
        for ci, (i, j, k) in enumerate(iproduct(range(n), range(m), range(p))):
            U_s[i*m+j, ci] = 1
            V_s[j*p+k, ci] = 1
            W_s[k*n+i, ci] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try column elimination
    U, V, W = try_eliminate_columns(U, V, W, n, m, p, max_time=30.0)
    rank = U.shape[1]
    
    # Try smooth optimization for further reduction
    if remaining() > 20.0:
        U2, V2, W2 = smooth_rank_reduce(U, V, W, n, m, p, max_time=min(60.0, remaining() - 10.0))
        if U2.shape[1] < U.shape[1] and verify_fast(U2, V2, W2, n, m, p, tol=1e-9):
            U, V, W = U2, V2, W2
    
    # Clean up: round to nearest representable value
    for mat in [U, V, W]:
        np.round(mat, decimals=10, out=mat)
        # Zero out very small values
        mat[np.abs(mat) < 1e-10] = 0.0
    
    # Final verification
    if not verify_fast(U, V, W, n, m, p, tol=1e-9):
        # Revert to the pre-optimization version
        U, V, W, _ = build_strassen_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
