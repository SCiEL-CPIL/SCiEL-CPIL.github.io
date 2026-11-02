import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        r = U_mat.shape[1]
        # Reconstruct the tensor
        # T[i, j, k, jp, kp] should be delta[j, jp] * delta[k, kp] ? 
        # No, the tensor for mat mult (A * B) = C is:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # The tensor is T_{i, j, k, j', k'} = delta_{j, j'} delta_{k, k'} ?
        # Wait, the standard tensor for multiplication is T_{i, j, k} where i is row of A, j is col of A/row of B, k is col of B.
        # The task specifies: "output factor follows the official trace convention W[k*n+i,r]"
        # This implies the tensor indices are (i, j, k) for A_{ij}, B_{jk}, C_{ik}.
        # The decomposition is C_{ik} = sum_c U_{i,j,c} V_{j',k,c} W_{k',i,c} ?
        # No, the standard decomposition is:
        # C_{ik} = sum_{j} A_{ij} B_{jk}
        # We approximate this by sum_c U_{i,j,c} V_{j',k,c} W_{k',i,c} ?
        # The verify function in the prompt checks:
        # s = sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
        # expected = 1.0 if j == jp else 0.0
        # This checks the identity:
        # sum_j A_{ij} B_{jk} = sum_c (sum_j U_{ijc} A_{ij}) (sum_k V_{jkc} B_{jk}) ... ?
        # Actually, it checks the tensor T_{i, j, k, j', k'} = delta_{j, j'} delta_{k, k'}?
        # No, the prompt's verify loop:
        # for i, k, j, jp:
        #   s = sum_c U[i*m+j, c] V[jp*p+k, c] W[k*n+i, c]
        #   expected = 1 if j==jp else 0
        # This implies the tensor being decomposed is the "identity" tensor in the middle dimension?
        # Wait, if j != jp, expected is 0.
        # This corresponds to the matrix multiplication tensor where the "contracted" index is j.
        # The tensor is T_{i, j, k, j', k'} = \delta_{j, j'} \delta_{k, k'}?
        # No, if k != kp, it doesn't check k.
        # Let's re-read the verify function carefully.
        # It iterates i, k, j, jp.
        # It sums over c.
        # It compares to 1 if j==jp else 0.
        # This means the tensor is non-zero only when j==jp.
        # This is the tensor for multiplying a 2x4 matrix by a 4x7 matrix?
        # The result C is 2x7.
        # C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # The tensor T has indices i, j, k.
        # The decomposition is T_{i, j, k} = sum_c U_{i, j, c} V_{j, k, c} W_{?, ?, c}?
        # The prompt's W index is k*n+i. So W depends on k and i.
        # This is a "slice" or "trace" convention.
        # The verify function effectively checks if the decomposition correctly computes the identity map on the middle dimension?
        # No, it checks the full tensor.
        
        # Vectorized verification:
        # We want to check if sum_c U[:, :, c] * V[:, :, c] * W[:, :, c] == Target_Tensor
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        # The target tensor T has shape (n, m, p, m, p)? No.
        # The verify loop indices: i (0..n), k (0..p), j (0..m), jp (0..m).
        # The sum is over c.
        # Term: U[i, j, c] * V[jp, k, c] * W[k, i, c]
        # This looks like a tensor product contraction.
        
        # Let's just use the loop for correctness first, it's small (2*4*7*4*7*50 ~ 60k ops).
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U_mat[i*m + j, c] * V_mat[jp*p + k, c] * W_mat[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def build_rank50():
        """Build the current rank-50 Strassen+schoolbook decomposition."""
        rank = 50
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            w00 = k0 * n + i0
            w01 = (k0 + 1) * n + i0
            w10 = k0 * n + (i0 + 1)
            w11 = (k0 + 1) * n + (i0 + 1)
            
            U[a, col] += 1; U[d, col] += 1
            V[e, col] += 1; V[h, col] += 1
            W[w00, col] += 1; W[w11, col] += 1
            col += 1
            
            U[c, col] += 1; U[d, col] += 1
            V[e, col] += 1
            W[w10, col] += 1; W[w11, col] -= 1
            col += 1
            
            U[a, col] += 1
            V[f, col] += 1; V[h, col] -= 1
            W[w01, col] += 1; W[w11, col] += 1
            col += 1
            
            U[d, col] += 1
            V[g, col] += 1; V[e, col] -= 1
            W[w00, col] += 1; W[w10, col] += 1
            col += 1
            
            U[a, col] += 1; U[b, col] += 1
            V[h, col] += 1
            W[w00, col] -= 1; W[w01, col] += 1
            col += 1
            
            U[c, col] += 1; U[a, col] -= 1
            V[e, col] += 1; V[f, col] += 1
            W[w11, col] += 1
            col += 1
            
            U[b, col] += 1; U[d, col] -= 1
            V[g, col] += 1; V[h, col] += 1
            W[w00, col] += 1
            col += 1

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            w0 = k0 * n + i0
            w1 = k0 * n + (i0 + 1)
            
            U[a, col] += 1; V[e, col] += 1; W[w0, col] += 1
            col += 1
            U[b, col] += 1; V[g, col] += 1; W[w0, col] += 1
            col += 1
            U[c, col] += 1; V[e, col] += 1; W[w1, col] += 1
            col += 1
            U[d, col] += 1; V[g, col] += 1; W[w1, col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_222(0, j_start, k_start)
            add_schoolbook_221(0, j_start, 6)
        
        assert col == rank
        return U, V, W
    
    # Build initial rank-50 decomposition
    U, V, W = build_rank50()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
