"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import scipy.linalg

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    # Schoolbook decomposition as fallback
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        # Returns U, V, W for multiplying 2x2 by 2x2
        # C[0,0] = M1 - M2 + M4 + M5
        # C[0,1] = M3 + M5 - M7 - M8
        # C[1,0] = M1 + M2
        # C[1,1] = M4 - M3 + M6 + M7
        # where:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # M8 = (A[0,0]+A[1,1])*(B[0,0]-B[1,1]) -- wait, let me use standard Strassen
        
        # Standard Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # M8 = (a11+a22)(b11-b22) -- actually this is sometimes used, but standard uses 7
        
        # Actually standard Strassen uses 7 multiplications:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)(b11)
        # M3 = (a11)(b21-b22)
        # M4 = (a22)(b12-b11)
        # M5 = (a11+a12)(b22)
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Indexing: A is 2x2, rows 0,1; cols 0,1 -> indices 0,1,2,3 for [0,0],[0,1],[1,0],[1,1]
        # B is 2x2, rows 0,1; cols 0,1 -> indices 0,1,2,3 for [0,0],[0,1],[1,0],[1,1]
        # C is 2x2, rows 0,1; cols 0,1 -> indices 0,1,2,3 for [0,0],[0,1],[1,0],[1,1]
        
        # U[i*m+j, col] where i is row of A, j is col of A
        # V[j*p+k, col] where j is row of B, k is col of B
        # W[k*n+i, col] where k is col of C, i is row of C
        
        # M1: (a11+a22)(b11+b22)
        # A part: a11 (idx 0) + a22 (idx 3)
        # B part: b11 (idx 0) + b22 (idx 3)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C11 += M1, C22 += M1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (a21+a22)(b11)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C21 += M2, C22 -= M2
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3: a11(b21-b22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # C12 += M3, C22 += M3
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: a22(b12-b11)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        # C11 -= M4, C21 += M4
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5: (a11+a12)(b22)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 += M5, C12 += M5
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6: (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C22 += M6
        W[3, 5] = 1
        
        # M7: (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 += M7
        W[0, 6] = 1
        
        return U, V, W, rank

    # Build a decomposition for n x m x p using block decomposition with Strassen where possible
    def build_decomposition(n, m, p):
        """
        Build a rank-minimal decomposition for n x m x p matrix multiplication.
        Uses block decomposition and Strassen for 2x2x2 blocks.
        """
        # If m or p is 1, use schoolbook (no Strassen possible)
        # Strategy: try to decompose into blocks of size 2 where possible
        
        # Simple approach: if m >= 2 and p >= 2 and n >= 2, we can use Strassen on 2x2x2 blocks
        # But this is complex for arbitrary n, m, p.
        
        # For now, use a simpler approach:
        # If all dimensions are even, we can use Strassen recursively.
        # Otherwise, fall back to schoolbook or partial Strassen.
        
        # Let's try: decompose m into m1 + m2, p into p1 + p2
        # C = A1 * B1 + A2 * B2 (if we split along m)
        # Or C = A * B1 + A * B2 (if we split along p)
        
        # For 3x4x6:
        # Split m=4 into 2+2: C = A_left(3x2) * B_top(2x6) + A_right(3x2) * B_bottom(2x6)
        # Each term is 3x2x6.
        # For 3x2x6: split p=6 into 2+2+2: three 3x2x2 terms.
        # For 3x2x2: split n=3 into 2+1: 
        #   (2x2x2 part) + (1x2x2 part)
        #   2x2x2 uses Strassen (rank 7)
        #   1x2x2 uses schoolbook (rank 4)
        #   Total for 3x2x2: 11
        # Total for 3x2x6: 3 * 11 = 33
        # Total for 3x4x6: 2 * 33 = 66
        
        # Let's implement this specific construction for 3x4x6
        if n == 3 and m == 4 and p == 6:
            # Build 3x4x6 as 2 * (3x2x6)
            # Each 3x2x6 as 3 * (3x2x2)
            # Each 3x2x2 as (2x2x2 Strassen) + (1x2x2 schoolbook)
            
            U_list = []
            V_list = []
            W_list = []
            
            # For each of the 2 m-blocks (left, right)
            for m_block in range(2):
                # For each of the 3 p-blocks
                for p_block in range(3):
                    # 3x2x2 decomposition
                    
                    # Part 1: 2x2x2 Strassen for rows 0,1 of A
                    U_s, V_s, W_s, rank_s = strassen_2x2x2()
                    
                    # Remap indices:
                    # A has 3 rows, 2 cols. For this 3x2x2 block:
                    #   A rows 0,1 are the 2x2 part. Row 2 is the 1x2 part.
                    #   B has 2 rows, 2 cols.
                    #   C has 3 rows, 2 cols.
                    
                    # U_s is 4x7 for 2x2 A part (rows 0,1 of A, cols 0,1)
                    # We need U to be (3*2)=6 rows for the full 3x2 A
                    # V is (2*2)=4 rows for B (2x2)
                    # W is (2*3)=6 rows for C (2 cols, 3 rows) -> W[k*n+i, col]
                    
                    U_part = np.zeros((6, rank_s))
                    V_part = np.zeros((4, rank_s))
                    W_part = np.zeros((6, rank_s))
                    
                    # Copy Strassen U into rows 0,1 of U_part (A rows 0,1)
                    # U_s[i*2+j, col] -> U_part[i*2+j, col] for i=0,1
                    U_part[0:4, :] = U_s
                    V_part[:, :] = V_s
                    # W_s[k*2+i, col] -> W_part[k*3+i, col] for i=0,1
                    # k is col of C (0,1), i is row of C (0,1)
                    for k in range(2):
                        for i in range(2):
                            W_part[k*3 + i, :] = W_s[k*2 + i, :]
                    
                    U_list.append(U_part)
                    V_list.append(V_part)
                    W_list.append(W_part)
                    
                    # Part 2: 1x2x2 schoolbook for row 2 of A
                    # A row 2, cols 0,1. B 2x2. C row 2, cols 0,1.
                    # C[2,k] = sum_j A[2,j] * B[j,k]
                    # 4 terms: (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1)
                    rank_sb = 4
                    U_sb = np.zeros((6, rank_sb))
                    V_sb = np.zeros((4, rank_sb))
                    W_sb = np.zeros((6, rank_sb))
                    
                    col = 0
                    for j in range(2):
                        for k in range(2):
                            # A[2,j]: row 2, col j -> index 2*2 + j = 4+j
                            U_sb[4 + j, col] = 1
                            # B[j,k]: row j, col k -> index j*2 + k
                            V_sb[j * 2 + k, col] = 1
                            # C[2,k]: row 2, col k -> index k*3 + 2
                            W_sb[k * 3 + 2, col] = 1
                            col += 1
                    
                    U_list.append(U_sb)
                    V_list.append(V_sb)
                    W_list.append(W_sb)
            
            U = np.hstack(U_list)
            V = np.hstack(V_list)
            W = np.hstack(W_list)
            rank = U.shape[1]
            return U, V, W, rank
        
        # Fallback: schoolbook
        U, V, W, rank = schoolbook(n, m, p)
        return U, V, W, rank

    # Try the specific construction
    U, V, W, rank = build_decomposition(n, m, p)
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        # Build the multiplication tensor
        # T[i*m+j, j*p+k, k*n+i] = 1
        # For each term c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Check a few random entries
        rng = np.random.RandomState(42)
        for _ in range(100):
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            expected = 1.0
            actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
            if abs(actual - expected) > tol:
                return False, (i, j, k, actual)
        
        # Also check that non-entries are zero
        # T[i*m+j, j'*p+k, k'*n+i'] = 0 if (j != j') or (k != k')
        for _ in range(100):
            i = rng.randint(n)
            j = rng.randint(m)
            j2 = rng.randint(m)
            k = rng.randint(p)
            i2 = rng.randint(n)
            if j == j2 and k == k:
                continue  # This is a valid entry
            expected = 0.0
            actual = sum(U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i2, c] for c in range(rank))
            if abs(actual - expected) > tol:
                return False, (i, j, j2, k, i2, actual)
        
        return True, None

    is_valid, err = verify(U, V, W, n, m, p)
    
    if not is_valid:
        # Fall back to schoolbook
        U, V, W, rank = schoolbook(n, m, p)
        is_valid, err = verify(U, V, W, n, m, p)
    
    # Convert to list of lists for the expected format
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
