"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10). Rank 55."""
import time
import numpy as np

def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    col = 0

    # For each pair of columns (k0, k1), use rank-11 2x3x2 decomposition.
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Term 1: (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
        U[0][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 2: (A10+A11)*B0k0 -> C1k0, C1k1
        U[3][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        # Term 3: A00*(B0k1-B1k1) -> C0k1, C1k1
        U[0][col] += 1
        V[0*10+k1][col] += 1; V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 4: A11*(B1k0-B0k0) -> C0k0, C1k0
        U[4][col] += 1
        V[1*10+k0][col] += 1; V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        # Term 5: (A00+A01)*B1k1 -> C0k0(-1), C0k1
        U[0][col] += 1; U[1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        # Term 6: (A10-A00)*(B0k0+B0k1) -> C1k1
        U[3][col] += 1; U[0][col] -= 1
        V[0*10+k0][col] += 1; V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        # Term 7: (A01-A11)*(B1k0+B1k1) -> C0k0
        U[1][col] += 1; U[4][col] -= 1
        V[1*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        # Terms 8-11: direct terms for row 2 of B
        U[2][col] += 1; V[2*10+k0][col] += 1; W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k1][col] += 1; W[k1*2+0][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k1][col] += 1; W[k1*2+1][col] += 1
        col += 1

    # Try to find a linear dependency to reduce rank
    if time.time() - _t0 < 140:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Check linear independence of the 55 rank-1 tensors
            # T_j = U[:,j] ⊗ V[:,j] ⊗ W[:,j]
            # Flatten each into a vector of length 6*30*20 = 3600
            # Then check if they're linearly independent
            
            # Build the 3600 x 55 matrix
            flat_tensors = np.zeros((3600, 55))
            for j in range(55):
                flat_tensors[:, j] = np.kron(np.kron(U_np[:, j], V_np[:, j]), W_np[:, j])
            
            # Check rank
            r = np.linalg.matrix_rank(flat_tensors, tol=1e-10)
            if r < 55:
                # There's a linear dependency. Find it.
                # Get null space
                _, s, vh = np.linalg.svd(flat_tensors, full_matrices=True)
                # Null space vectors are rows of vh corresponding to zero singular values
                null_mask = s < 1e-10
                if np.any(null_mask):
                    # Take the first null vector
                    null_vec = vh[-1, :]  # last row of vh
                    # Find the largest coefficient
                    idx_max = np.argmax(np.abs(null_vec))
                    # We can eliminate column idx_max by adjusting others
                    # new_U = U - null_vec[idx_max] * (U / null_vec[idx_max]) ... this is tricky
                    # Actually, if sum_j c_j * T_j = 0, then we can remove one term
                    # by expressing it in terms of others
                    c = null_vec
                    c_idx = idx_max
                    if abs(c[c_idx]) > 1e-10:
                        # T_c_idx = -sum_{j != c_idx} (c_j / c_c_idx) * T_j
                        # So we can eliminate column c_idx
                        new_rank = 54
                        new_U = [U[i][:c_idx] + U[i][c_idx+1:] for i in range(6)]
                        new_V = [V[i][:c_idx] + V[i][c_idx+1:] for i in range(30)]
                        new_W = [W[i][:c_idx] + W[i][c_idx+1:] for i in range(20)]
                        # Verify: the tensor sum should be the same
                        # sum_j c_j T_j = 0 means sum_j T_j = T_c_idx - sum_{j!=c_idx} T_j ... 
                        # Actually, the original sum is sum_j T_j. If sum_j c_j T_j = 0,
                        # then sum_j T_j = sum_j T_j - sum_j c_j T_j ... this doesn't directly help.
                        # Let me reconsider. The decomposition is: M = sum_j U_j ⊗ V_j ⊗ W_j
                        # If there's a dependency sum_j c_j (U_j ⊗ V_j ⊗ W_j) = 0,
                        # then we can't simply remove one term. We need to find a subset
                        # that sums to zero.
                        # This is getting complicated. Let's just return rank 55.
                        pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
