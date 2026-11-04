import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS

    def verify_numpy(U, V, W, n, m, p):
        """
        Verify tensor decomposition using NumPy.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        
        Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        We want T[i,j,k] = 1 for all i,j,k.
        
        We can check this by constructing the full tensor T and comparing to ones.
        Or more efficiently, for each i, k, check if sum_j T[i,j,k] matches AB?
        No, the tensor represents the bilinear form.
        T is of shape (n, m, p).
        T[i, j, k] = 1.
        
        Let's compute T from U, V, W.
        T = sum_r U_r[:, None, None] * V_r[None, :, None] * W_r[None, None, :] 
        where U_r is reshaped to (n, m), V_r to (m, p), W_r to (p, n)?
        No, W index is k*n + i. So W_r is a vector of length p*n.
        Reshape W_r to (p, n).
        Then term_r[i, j, k] = U_r[i, j] * V_r[j, k] * W_r[k, i].
        
        Let's use einsum or explicit loops if small.
        n=2, m=3, p=10. Total 60 elements.
        Rank is ~55.
        60 * 55 ops is trivial.
        """
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=np.float64)
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            # Loop over i, j, k for clarity since dimensions are tiny
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        # Check if T is all ones
        if np.allclose(T, 1.0, atol=1e-9):
            return True
        return False

    # Rank 55: Strassen for 2x2x10 part + Schoolbook for 2x1x10 part
    def make_rank55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        
        # Strassen for each 2x2x2 block (5 blocks, each using 7 terms)
        # A_left = A[:, 0:2] (2x2)
        # B_top = B[0:2, :] (2x10)
        # Output C_left = C[:, 0:10] (2x10)
        
        # Standard Strassen for 2x2 matrices A, B:
        # m1 = (a00 + a11) * (b00 + b11)
        # m2 = (a10 + a11) * b00
        # m3 = a00 * (b01 - b11)
        # m4 = a11 * (b10 - b00)
        # m5 = (a00 + a01) * b11
        # m6 = (a10 - a00) * (b00 + b01)
        # m7 = (a01 - a11) * (b10 + b11)
        
        # C00 = m1 + m4 + m5 + m6
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m7
        
        # We decompose B_top into 5 blocks of 2x2: B_k for k=0..4
        # B_k has cols 2k, 2k+1.
        # For each block k, we compute C_block_k (2x2) using Strassen.
        
        # Patterns for U, V, W for the 7 Strassen terms.
        # Local indices:
        # A: [[a00, a01], [a10, a11]] -> indices (0,0), (0,1), (1,0), (1,1)
        # B: [[b00, b01], [b10, b11]] -> indices (0,0), (0,1), (1,0), (1,1) relative to block
        # C: [[c00, c01], [c10, c11]] -> indices (0,0), (0,1), (1,0), (1,1) relative to block
        
        # U, V, W are global matrices.
        # U[i, j] corresponds to A[i, j].
        # V[j, k] corresponds to B[j, k].
        # W[k, i] corresponds to C[i, k] (Note: W is indexed k*n + i).
        
        # Term 1: m1 = (a00 + a11) * (b00 + b11)
        # U: a00+1, a11+1
        # V: b00+1, b11+1
        # C contribution: C00 += 1, C11 += 1
        # W: C[0,0] is W[0*n+0]=W[0], C[1,1] is W[1*n+1]=W[3] (if n=2)
        
        # Let's define the patterns explicitly for a 2x2x2 block.
        # Indices in U (2x2): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Indices in V (2x2): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Indices in W (2x2, k*2+i): 
        #   C[0,0] (i=0,k=0) -> W[0]
        #   C[0,1] (i=0,k=1) -> W[1]
        #   C[1,0] (i=1,k=0) -> W[2]
        #   C[1,1] (i=1,k=1) -> W[3]
        
        strassen_blocks = [
            # (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
            # m1
            ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
            # m2
            ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0)], [1]),
            # m3
            ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1)], [1]),
            # m4
            ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(1,0)], [1]),
            # m5
            ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [1, 1]),
            # m6
            ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(0,0)], [1]),
            # m7
            ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(1,1)], [1]),
        ]
        
        for block_idx in range(5):
            k1 = block_idx * 2
            # For each Strassen term
            for u_inds, u_vals, v_inds, v_vals, w_inds, w_vals in strassen_blocks:
                # Update U
                for (ri, ci), val in zip(u_inds, u_vals):
                    U[ri * m + ci, col] += val
                # Update V
                # B global row is same as local row (0,1).
                # B global col is k1 + local_col
                for (ri, ci), val in zip(v_inds, v_vals):
                    V[ri * p + (k1 + ci), col] += val
                # Update W
                # C global row i, col k.
                # W index is k * n + i.
                # Local C indices (i, k) map to global (i, k1+k)
                for (ci, ki), val in zip(w_inds, w_vals):
                    # Note: w_inds are (k, i) in my previous comment? 
                    # Let's stick to (i, k) for clarity in the tuple, then map.
                    # My patterns above used (i, k) for W indices?
                    # m1 W: [(0,0), (1,1)] -> C[0,0] and C[1,1].
                    # So (0,0) is i=0, k=0. (1,1) is i=1, k=1.
                    # Global k = k1 + local_k.
                    global_k = k1 + ki
                    W[global_k * n + ci, col] += val
                
                col += 1
        
        # Schoolbook for A_right * B_bottom
        # A_right is A[:, 2] (2x1). B_bottom is B[2, :] (1x10).
        # C += A_right * B_bottom.
        # C[i, k] += A[i, 2] * B[2, k].
        for i in range(n):
            for k in range(p):
                U[i * m + 2, col] = 1
                V[2 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        
        # Verify
        if verify_numpy(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try rank 55 first
    result = make_rank55()
    if result:
        return result
    
    # Fallback: Schoolbook (Rank 60)
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return make_schoolbook()
