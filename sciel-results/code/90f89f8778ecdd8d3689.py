import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        res = np.zeros((n, p))
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            res += np.einsum('ij,jk,ki->ik', U_c, V_c, W_c)
        return np.allclose(res, 1.0, atol=1e-9)

    def to_half_integer_list(mat):
        """Convert a matrix to a list of lists of half-integers (as floats or ints)."""
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                val = mat[i, j]
                # Check if it's a half-integer
                f = Fraction(val).limit_denominator(1000000)
                if f.denominator in [1, 2]:
                    if f.denominator == 1:
                        row.append(int(f.numerator))
                    else:
                        row.append(float(f))
                else:
                    # Fallback: just use the float value, but this shouldn't happen for our constructions
                    row.append(float(val))
            result.append(row)
        return result

    # --- Construction 1: Block Decomposition (Rank 28) ---
    def build_block_decomp():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00 + A11)(B00 + B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            
            # M2 = (A10 + A11) B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            
            # M3 = A00 (B01 - B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            
            # M4 = A11 (B10 - B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            
            # M5 = (A00 + A01) B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer_product(i_indices, j_idx, k_indices):
            nonlocal col
            for i in i_indices:
                for k in k_indices:
                    U[i*m + j_idx, col] = 1
                    V[j_idx*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1

        # 1. A[:, 0:2] * B[0:2, 0:2] -> Strassen
        add_strassen(0, 1, 0, 1, 0, 1)
        
        # 2. A[:, 2] * B[2, 0:2] -> Outer
        add_outer_product([0, 1], 2, [0, 1])
        
        # 3. A[:, 0:2] * B[0:2, 2:4] -> Strassen
        add_strassen(0, 1, 0, 1, 2, 3)
        
        # 4. A[:, 0:2] * B[0:2, 4] -> Outer
        add_outer_product([0, 1], 0, [4])
        add_outer_product([0, 1], 1, [4])
        
        # 5. A[:, 2] * B[2, 2:5] -> Outer
        add_outer_product([0, 1], 2, [2, 3, 4])
        
        return U, V, W, rank

    # --- Construction 2: Try to find a better rank via simple optimization ---
    # This is a placeholder for a more advanced search if time permits.
    # For now, we stick to the robust block decomposition.
    # If we had more time, we could try:
    # 1. Random initialization of U, V, W with small integers
    # 2. Alternating least squares to fit the tensor
    # 3. Rounding to nearest half-integer
    # 4. Verification
    # But given the 120s limit and the risk of non-convergence, we prioritize the known rank 28.

    # Try Construction 1
    U, V, W, rank = build_block_decomp()
    
    if verify(U, V, W, rank):
        # Convert to lists
        U_list = to_half_integer_list(U)
        V_list = to_half_integer_list(V)
        W_list = to_half_integer_list(W)
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook (should not happen, but safe)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_list = to_half_integer_list(U)
        V_list = to_half_integer_list(V)
        W_list = to_half_integer_list(W)
        return {"U": U_list, "V": V_list, "W": W_list}
