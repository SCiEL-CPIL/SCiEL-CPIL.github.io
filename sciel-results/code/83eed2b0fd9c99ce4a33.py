import time
import numpy as np

def construct():
    n, m, p = 5, 5, 5
    
    # Partition 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    # offsets = [0, 2, 4, 5]
    
    # Strassen factors for 2x2x2
    # Each factor: (U_vec, V_vec, W_vec) where each vec has length 4
    # U indices: [a11, a12, a21, a22]
    # V indices: [b11, b12, b21, b22]
    # W indices: [c11, c12, c21, c22]
    strassen_factors = [
        # M1 = (a11+a22)(b11+b22) -> c11 += M1, c12 -= M1, c21 -= M1, c22 += M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a21+a22)b11 -> c21 += M2, c22 -= M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = a11(b12-b22) -> c12 += M3, c22 += M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = a22(b21-b11) -> c11 += M4, c21 += M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (a11+a12)b22 -> c11 -= M5, c12 += M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (a21-a11)(b11+b12) -> c22 += M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a12-a22)(b21+b22) -> c11 += M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Compute total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Initialize factor matrices
    U = [[0.0] * total_rank for _ in range(n * m)]  # 25 x rank
    V = [[0.0] * total_rank for _ in range(m * p)]  # 25 x rank
    W = [[0.0] * total_rank for _ in range(p * n)]  # 25 x rank
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen for this 2x2x2 block
                    for (uf, vf, wf) in strassen_factors:
                        # Map local 2x2 indices to global indices
                        # U: row block i (size 2), col block j (size 2)
                        # Local U indices: [0,1,2,3] = [(i,j),(i,j+1),(i+1,j),(i+1,j+1)]
                        local_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        # V: row block j (size 2), col block k (size 2)
                        local_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        # W: row block k (size 2), col block i (size 2)
                        # W convention: W[k*n+i, r] where k is output row, i is output col
                        local_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify the decomposition is correct
    # C = U @ diag(multipliers) @ V^T should equal the matrix product
    # Actually, the decomposition is: sum_r U[:,r] * V[:,r] * W[:,r]^T = matmul tensor
    # Let's verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from the decomposition
    # C[k, i] = sum_r W[k*n+i, r] * (sum_s U[s, r] * A[...]) * (sum_t V[t, r] * B[...])
    # More precisely: for each r, let a_r = U[:,r] as a matrix (n x m), b_r = V[:,r] as (m x p)
    # Then C += W[:,r] as (p x n) outer product... 
    
    # The tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So C[k,i] = sum_j sum_r A[i,j] * B[j,k] * ... 
    
    # Let me verify differently:
    # For each r, define:
    #   A_r[i,j] = U[i*m+j, r]
    #   B_r[j,k] = V[j*p+k, r]  
    #   C_r[k,i] = W[k*n+i, r]
    # Then sum_r (A_r * A) ⊙ (B_r * B) ... no, that's not right.
    
    # The correct interpretation:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # = sum_r sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # This is a bilinear form, so:
    # Let x_r = sum_j U[i*m+j, r] * A[i,j] for each i -> but this depends on i
    # Actually, for fixed r:
    #   u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    #   The contribution to C is: for each (i,j,k), T[i,j,k] += u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    #   So C[k,i] = sum_j sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] * A[i,j] * B[j,k]
    #   Wait, that's not how tensor decomposition works for matmul.
    
    # The matmul tensor T[i,j,k] = 1 if we're computing C[k,i] from A[i,j]*B[j,k]
    # Actually T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that C[k,i] = sum_j A[i,j]*B[j,k])
    # No, T[i,j,k] is the coefficient such that C[k,i] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # The rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    
    # Let's verify this
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape: U_np is (n*m, rank), V_np is (m*p, rank), W_np is (p*n, rank)
    # T[i,j,k] = sum_r U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # Check a few entries
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(total_rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    assert max_err < 1e-10, f"Decomposition error too large: {max_err}"
    
    return {"U": U, "V": V, "W": W}
