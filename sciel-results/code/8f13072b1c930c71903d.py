"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a bilinear decomposition for (2,5,5) matrix multiplication
    using block decomposition and Strassen's algorithm for sub-blocks.
    """
    import numpy as np

    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zero_matrix(rows, cols):
        return [[0] * cols for _ in range(rows)]

    # Helper to combine decompositions
    # This is complex to do generically with lists, so we'll do it step-by-step in construct
    # Or we can use numpy for internal construction and convert to lists at the end.
    
    # We'll use a recursive function that returns numpy arrays for U, V, W
    def construct_np(n, m, p):
        # Base case: m == 1
        if m == 1:
            rank = n * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            idx = 0
            for i in range(n):
                for k in range(p):
                    U[i * m + 0, idx] = 1.0
                    V[0 * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
            return U, V, W
        
        # Base case: n == 1 and p == 1
        if n == 1 and p == 1:
            rank = m
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for j in range(m):
                U[0 * m + j, j] = 1.0
                V[j * p + 0, j] = 1.0
                W[0 * n + 0, j] = 1.0
            return U, V, W

        # Base case: n == 2 and m == 2 and p == 2 (Strassen)
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # M1 = (A00 + A11) * (B00 + B11)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # C00, C01, C10, C11 all get M1? No.
            
            # Let's use the standard Strassen coefficients
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # W matrix rows are indexed by k*n + i.
            # k=0, i=0 -> 0 (C00)
            # k=0, i=1 -> 1 (C10)
            # k=1, i=0 -> 2 (C01)
            # k=1, i=1 -> 3 (C11)
            
            # M1: C00(1), C11(1)
            W[0, 0] = 1
            W[3, 0] = 1
            
            # M2 = (A10 + A11) * B00
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            # C10(1), C11(-1)
            W[1, 1] = 1
            W[3, 1] = -1
            
            # M3 = A00 * (B01 - B11)
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            # C01(1), C11(1)
            W[2, 2] = 1
            W[3, 2] = 1
            
            # M4 = A11 * (B10 - B00)
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            # C00(1), C10(1)
            W[0, 3] = 1
            W[1, 3] = 1
            
            # M5 = (A00 + A01) * B11
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            # C00(-1), C01(1)
            W[0, 4] = -1
            W[2, 4] = 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            # C11(1)
            W[3, 5] = 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            # C00(1)
            W[0, 6] = 1
            
            return U, V, W

        # Recursive case: Split m
        # Split m into m1 and m2.
        # To use Strassen effectively, we prefer splitting into powers of 2 or small numbers.
        # For m=5, split into 2 and 3.
        # For m=3, split into 2 and 1.
        
        if m > 2:
            # Choose split point
            if m == 3:
                m1 = 2
                m2 = 1
            elif m == 5:
                m1 = 2
                m2 = 3
            else:
                # Fallback: split in half
                m1 = m // 2
                m2 = m - m1
            
            # Sub-problem 1: n x m1 x p
            U1, V1, W1 = construct_np(n, m1, p)
            # Sub-problem 2: n x m2 x p
            U2, V2, W2 = construct_np(n, m2, p)
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            R = r1 + r2
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Copy U1
            # U1 rows correspond to A indices i in [0,n), j in [0,m1)
            # Global U rows are i in [0,n), j in [0,m)
            # So for each i, j1 in U1, map to j1 in U.
            for i in range(n):
                for j1 in range(m1):
                    U[i * m + j1, :r1] = U1[i * m1 + j1, :]
            
            # Copy U2
            # U2 rows correspond to A indices i in [0,n), j in [0,m2)
            # Global U rows for j2 are j1 + j2
            for i in range(n):
                for j2 in range(m2):
                    U[i * m + m1 + j2, r1:] = U2[i * m2 + j2, :]
            
            # Copy V1
            # V1 rows correspond to B indices j in [0,m1), k in [0,p)
            # Global V rows are j in [0,m), k in [0,p)
            for j1 in range(m1):
                for k in range(p):
                    V[j1 * p + k, :r1] = V1[j1 * p + k, :]
            
            # Copy V2
            # V2 rows correspond to B indices j in [0,m2), k in [0,p)
            # Global V rows for j2 are m1 + j2
            for j2 in range(m2):
                for k in range(p):
                    V[(m1 + j2) * p + k, r1:] = V2[j2 * p + k, :]
            
            # Copy W
            # W is simply concatenated columns
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            return U, V, W
        
        # If m is small but not base case (e.g. m=2, n!=2 or p!=2)
        # Fallback to schoolbook if not handled
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    U, V, W = construct_np(n, m, p)
    
    # Convert to lists for JSON serialization
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
