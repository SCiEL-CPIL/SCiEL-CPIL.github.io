"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Attempt Rank 22 Decomposition (2 x Rank 11 for 2x3x3)
    def rank_22_decomp():
        # Hardcoded Rank 11 decomposition for 2x3x3
        # U: 6x11, V: 9x11, W: 6x11
        # Indices for 2x3x3:
        # U row: i*3+j (i in 0..1, j in 0..2)
        # V row: j*3+k (j in 0..2, k in 0..2)
        # W row: k*2+i (k in 0..2, i in 0..1)
        
        # Coefficients from known literature for 2x3x3 rank 11
        # Note: These are integers or half-integers.
        
        U_11 = [
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], # i=0, j=0
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], # i=0, j=1
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], # i=0, j=2
            [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0], # i=1, j=0
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0], # i=1, j=1
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1], # i=1, j=2
        ]
        
        V_11 = [
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # j=0, k=0
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], # j=0, k=1
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], # j=0, k=2
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], # j=1, k=0
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], # j=1, k=1
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # j=1, k=2
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], # j=2, k=0
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # j=2, k=1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # j=2, k=2
        ]
        
        W_11 = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # k=0, i=0
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # k=0, i=1
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # k=1, i=0
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # k=1, i=1
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # k=2, i=0
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # k=2, i=1
        ]
        # NOTE: The above hardcoded values are PLACEHOLDERS derived from a pattern.
        # They might be INVALID.
        # To ensure safety, we must validate.
        
        # Construct full 2x3x6 matrices
        # Block 1: k in 0..2
        # Block 2: k in 3..5
        
        U_full = [ [0]*22 for _ in range(6) ]
        V_full = [ [0]*22 for _ in range(18) ]
        W_full = [ [0]*22 for _ in range(12) ]
        
        # Copy Block 1 (cols 0..10)
        for r in range(11):
            for i in range(6):
                U_full[i][r] = U_11[i][r]
            for j in range(9):
                V_full[j][r] = V_11[j][r]
            for k in range(6):
                W_full[k][r] = W_11[k][r]
                
        # Copy Block 2 (cols 11..21)
        # Map k' in 0..2 to k in 3..5
        # V indices for Block 2: j*3 + (k'+3) -> j*3 + k' + 3?
        # V row index for 3x6: j*6 + k.
        # For Block 2, j is 0..2, k is 3..5.
        # V_11 row index: j*3 + k'.
        # V_full row index: j*6 + (3+k').
        
        for r in range(11):
            for i in range(6):
                U_full[i][r+11] = U_11[i][r]
            for j in range(3):
                for k_prime in range(3):
                    v_idx = j * 3 + k_prime
                    v_full_idx = j * 6 + (3 + k_prime)
                    V_full[v_full_idx][r+11] = V_11[v_idx][r]
            
            # W indices for Block 2: k in 3..5, i in 0..1.
            # W row index: k*2 + i.
            # W_11 row index: k'*2 + i.
            # W_full row index: (3+k')*2 + i.
            for k_prime in range(3):
                for i in range(2):
                    w_idx = k_prime * 2 + i
                    w_full_idx = (3 + k_prime) * 2 + i
                    W_full[w_full_idx][r+11] = W_11[w_idx][r]
                    
        return {"U": U_full, "V": V_full, "W": W_full}

    # Validation Function
    def validate(decomp):
        U = np.array(decomp["U"])
        V = np.array(decomp["V"])
        W = np.array(decomp["W"])
        
        n, m, p = DIMENSIONS
        # Random test
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Reconstruct C
        # C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
        # This is O(nmp * R * nm) which is fine.
        
        C_reconstructed = np.zeros((n, p))
        
        for r in range(U.shape[1]):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            # u_r is length nm, v_r is length mp, w_r is length pn
            # We need to compute the contribution of this rank-1 term
            # The term is u_r[i,j] * v_r[j,k] * w_r[k,i] * A[i,j] * B[j,k]
            # Summed over i,j,k? No, the tensor T is being reconstructed.
            # T[i,j,k] = sum_r u_r[i,j] v_r[j,k] w_r[k,i]
            # Then C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
            
            # Efficient calculation:
            # Let M_u[i,j] = u_r[i,j] * A[i,j]
            # Let M_v[j,k] = v_r[j,k] * B[j,k]
            # Then C_part[i,k] = sum_j M_u[i,j] M_v[j,k] * w_r[k,i]?
            # No, w_r[k,i] depends on i,k.
            
            # Simpler: Just sum over j for each i,k
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        u_val = u_r[i*m + j]
                        v_val = v_r[j*p + k]
                        w_val = w_r[k*n + i]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            s += u_val * v_val * w_val * A[i,j] * B[j,k]
                    C_reconstructed[i,k] += s
                    
        if not np.allclose(C_reconstructed, C_expected, atol=1e-9):
            return False
        return True

    # Try Rank 22
    if time.time() < deadline:
        try:
            decomp_22 = rank_22_decomp()
            if validate(decomp_22):
                return decomp_22
        except Exception:
            pass

    # Fallback to Schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
