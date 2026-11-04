"""Strassen-based 4x5x6 matrix multiplication decomposition (rank 108)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS
    
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i0, j0, k0, r_start):
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        
        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        
        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        return r_start + 1

    r_offset = 0
    
    # 12 Strassen blocks for the 4x4x4 core
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                r_offset = add_strassen_block(i0, j0, k0, r_offset)
    
    # Strip: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6], rank 24
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1
            
    # Verify correctness
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the tensor T from U, V, W: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # But the convention is W[k*n+i, r], so T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        A_r = U_np[:, r].reshape(n, m)
        B_r = V_np[:, r].reshape(m, p)
        C_r = W_np[:, r].reshape(p, n)
        # C[i,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
        # Actually: C[i,k] = sum_r (sum_j A[i,j]*A_r[i,j]) * (sum_k B[j,k]*B_r[j,k]) * C_r[k,i]
        # Wait, the tensor decomposition is: C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Let me re-derive. The multiplication tensor T satisfies:
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # The tensor T has components T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]
        # No wait. The standard tensor decomposition for matrix multiplication:
        # The tensor T of matrix mult has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C.
        # So T[i,j,k] = 1 for all i,j,k (it's the identity in the j dimension).
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # And W is indexed as W[k*n+i, r]
        
        # So: sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_k' B[j,k'] V[j*p+k', r]) * W[k*n+i, r]
        # Wait, that's not right either. Let me think again.
        
        # The correct formula: C[i,k] = sum_r U_col_r @ A @ V_col_r * W[k*n+i, r]
        # where U_col_r = U[:, r] reshaped to (n, m), V_col_r = V[:, r] reshaped to (m, p)
        # No: C[i,k] = sum_r (U[:,r].reshape(n,m) * A).sum() ... no.
        
        # Let's just compute it directly:
        # C[i,k] = sum_r U[i*m+j, r] ... no, sum over j is implicit in the reshape.
        
        # Actually: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_{j'} B[j',k] * V[j'*p+k, r]) * W[k*n+i, r]
        # But that would be sum_r (A @ U_r^T)[i] * (V_r @ B^T)[k] * W[k*n+i, r]
        # Hmm, let me just compute numerically.
        
        U_r = U_np[:, r].reshape(n, m)
        V_r = V_np[:, r].reshape(m, p)
        W_r = W_np[:, r].reshape(p, n)
        
        # The formula should be:
        # C[i,k] = sum_r sum_j A[i,j] * U_r[i,j] * sum_{j'} B[j',k] * V_r[j',k] * W_r[k,i]
        # Wait no. Let me look at the Strassen construction.
        
        # In Strassen: M1 = (A00+A11)(B00+B11)
        # U has +1 at (0,0) and (1,1) positions
        # V has +1 at (0,0) and (1,1) positions  
        # W has +1 at (0,0) and (1,1) positions (for C00 and C11)
        
        # So for a 2x2x2 block: C[i,k] = sum_r (sum_j A[i,j]*U_r[i,j]) * (sum_{j'} B[j',k]*V_r[j',k]) * W_r[k,i]
        # For M1: sum_j A[i,j]*U_r[i,j] = A[0,0]+A[1,1] for i=0 or 1
        # sum_{j'} B[j',k]*V_r[j',k] = B[0,0]+B[1,1] for k=0 or 1
        # W_r[k,i] = 1 for (k,i) in {(0,0),(1,1)}
        # So C[0,0] gets contribution (A00+A11)(B00+B11)*1
        # C[1,1] gets contribution (A00+A11)(B00+B11)*1
        # That matches!
        
        a_part = np.sum(A * U_r)  # scalar: sum_ij A[i,j]*U_r[i,j]
        b_part = np.sum(B * V_r)  # scalar: sum_jk B[j,k]*V_r[j,k]
        C_decomp += a_part * b_part * W_r.T  # W_r is (p,n), W_r[k,i], so C[i,k] += a*b*W_r[k,i]
    
    error = np.max(np.abs(C_decomp - C_expected))
    if error > 1e-9:
        raise ValueError(f"Decomposition error: {error}")
    
    return {"U": U, "V": V, "W": W}
