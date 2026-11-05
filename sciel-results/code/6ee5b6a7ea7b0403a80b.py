import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    def verify_decomp(U_mat, V_mat, W_mat, tol=1e-8):
        """Verify tensor decomposition correctness using numpy."""
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        rank = U_np.shape[1]
        # Build tensor_mat: (125, rank)
        tensor_mat = np.zeros((n*m*p, rank))
        for r in range(rank):
            u_reshaped = U_np[:, r].reshape(n, m)
            v_reshaped = V_np[:, r].reshape(m, p)
            w_reshaped = W_np[:, r].reshape(p, n)
            t = np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped)
            tensor_mat[:, r] = t.flatten()
        # Check that sum over r gives the identity tensor
        result = tensor_mat.sum(axis=1)
        # Identity tensor: T[i,j,k] = 1 if i==k else 0, flat: i*m*p + j*p + k
        expected = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i == k:
                        expected[i*m*p + j*p + k] = 1.0
        if not np.allclose(result, expected, atol=tol):
            return False
        return True
    
    # Block sizes for partitioning 5 = 1 + 2 + 2
    block_sizes = [1, 2, 2]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen decomposition for 2x2x2
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors
    
    strassen_factors = get_strassen_factors()
    
    # Calculate total rank
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
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    # Return best effort if we haven't finished
                    # Since construction is fast, this is unlikely to trigger
                    pass
                
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen for 2x2x2 blocks
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for other blocks
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
    
    # Verify the decomposition
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook if verification fails
        U_s = [[0.0] * (n*m*p) for _ in range(n*m)]
        V_s = [[0.0] * (n*m*p) for _ in range(m*p)]
        W_s = [[0.0] * (n*m*p) for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_s[i*m+j][r] = 1.0
                    V_s[j*p+k][r] = 1.0
                    W_s[k*n+i][r] = 1.0
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Attempt rank reduction via null-space analysis
    if time.time() < deadline - 5.0:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build tensor matrix: (125, total_rank)
        tensor_mat = np.zeros((n*m*p, total_rank))
        for r in range(total_rank):
            u_reshaped = U_np[:, r].reshape(n, m)
            v_reshaped = V_np[:, r].reshape(m, p)
            w_reshaped = W_np[:, r].reshape(p, n)
            t = np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped)
            tensor_mat[:, r] = t.flatten()
        
        # Check rank of tensor_mat
        tensor_rank = np.linalg.matrix_rank(tensor_mat, tol=1e-10)
        
        if tensor_rank < total_rank:
            # Find null space and attempt rank reduction
            _, s, vh = np.linalg.svd(tensor_mat, full_matrices=True)
            null_vecs = vh[tensor_rank:, :].T  # shape (total_rank, d) where d = total_rank - tensor_rank
            
            # For each null vector c: sum_r c_r * T_r = 0
            # We can eliminate one rank vector by expressing it in terms of others
            # Pick the rank vector with largest |c_r| as the one to eliminate
            # Then: T_r* = -(1/c_r*) * sum_{r != r*} c_r * T_r
            # This means in the factor matrices:
            # U_r* = -(1/c_r*) * sum_{r != r*} c_r * U_r  (element-wise)
            # V_r* = -(1/c_r*) * sum_{r != r*} c_r * V_r
            # W_r* = -(1/c_r*) * sum_{r != r*} c_r * W_r
            # But this is WRONG because the tensor is a product, not a sum of products
            # sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0 does NOT imply
            # (U_r* ⊗ V_r* ⊗ W_r*) = sum_{r≠r*} (c_r/c_r*) * (U_r ⊗ V_r ⊗ W_r)
            # The LHS is a rank-1 tensor, RHS is a sum of rank-1 tensors
            # They can only be equal if the RHS is also rank-1, which is generally false
            
            # So null-space reduction of the tensor doesn't directly give rank reduction
            # of the factorization. We'd need a more sophisticated approach.
            pass
    
    return {"U": U, "V": V, "W": W}
