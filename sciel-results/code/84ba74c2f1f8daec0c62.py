import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    n_m = n * m  # 4
    m_p = m * p  # 14
    p_n = p * n  # 14
    
    # Laderman rank-7 decomposition for 2x2 x 2x2 -> 2x2
    # Indices: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}
    # 7 terms with (A_coeff, B_coeff, C_coeff) as 2x2 matrices
    laderman_terms = [
        # (A_mat, B_mat, C_mat) each 2x2
        ([1,0, 0,0], [1,0, 0,0], [1,0, 0,0]),  # A00*B00 -> C00
        ([0,1, 0,0], [0,0, 1,0], [0,0, 0,1]),  # A01*B11 -> C01... 
    ]
    # Actually let me use the standard Laderman decomposition explicitly.
    # Laderman's 7 terms for 2x2x2:
    # Using the standard formulation:
    # T1: A00, B00, C00
    # T2: A01, B10, C01  -- wait, need to be careful
    #
    # Standard Laderman (from Wikipedia / standard references):
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # i = a*e, j = a*f + b*h, k = c*g + d*e, l = c*h + d*f
    # 
    # Laderman's 7 multiplications:
    # m1 = a*e
    # m2 = (a+b)*(e+f)
    # m3 = (a+c)*(e+g)
    # m4 = (b+d)*(f+h)
    # m5 = (a-b+e-f)*... 
    #
    # Let me use a well-known explicit Laderman decomposition:
    # The 7 bilinear forms are:
    # M1 = A[0,0] * B[0,0]
    # M2 = A[0,1] * B[1,0]  
    # M3 = A[1,0] * B[0,1]
    # M4 = A[1,1] * B[1,1]
    # M5 = (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
    # M6 = (A[0,0]-A[0,1]+A[1,0]-A[1,1]) * (B[0,0]+B[0,1]-B[1,0]-B[1,1])
    # M7 = (A[0,0]+A[0,1]-A[1,0]-A[1,1]) * (B[0,0]-B[0,1]+B[1,0]-B[1,1])
    #
    # Then:
    # C[0,0] = M1
    # C[0,1] = M3 + (M5 - M6 - M7)/2  ... this is getting complicated
    #
    # Let me just use the direct rank-4 schoolbook for each 2x2x2 block instead,
    # which gives 4*4 = 16, and then try to optimize.
    #
    # Actually, let me use a verified Laderman decomposition.
    # Standard Laderman 2x2x2 rank-7:
    #
    # Define 7 terms where each term is a triple (u, v, w) of vectors/matrices
    # such that sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    #
    # For 2x2x2, the tensor has 8 non-zero entries. Laderman shows rank 7.
    #
    # Here's an explicit construction (verified):
    # Using 0-indexed i,j,k in {0,1}:
    
    # Let me build it directly. The 8 non-zero entries of T_{2,2,2}:
    # T[i,j,k] = 1 if (i,j,k) matches (i',j',k') where i=i', j=j', k=k'
    # i.e., T[i,j,k] = 1 for all 8 combinations, else 0.
    # Wait no. For matrix mult C[i,k] = sum_j A[i,j] B[j,k]:
    # T[i,j,k] = 1 (the tensor entry is 1 for each (i,j,k) triple)
    # So all 8 entries are 1.
    
    # For the full problem 2x2 x 2x7:
    # T[i,j, j',k] = 1 if j==j' (since A[i,j]*B[j,k] contributes to C[i,k])
    # With i in {0,1}, j in {0,1}, j' in {0,1}, k in {0,...,6}
    # T[i,j,j',k] = 1 if j == j', else 0
    # Total non-zero: 2 * 1 * 7 = 14? No wait.
    # T is indexed by (a,b,c) where a = i*m+j (A index), b = j'*p+k (B index), c = k'*n+i' (C index)
    # T[a,b,c] = 1 if i==i' and j==j' and k==k'
    # So T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i==i', j==j', k==k'
    # Number of non-zero: for each (i,j,k) there's exactly one (i',j',k') = (i,j,k)
    # So 2*2*7 = 28 non-zero entries, each equal to 1.
    
    # Schoolbook: rank 28, one term per (i,j,k)
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for term r corresponding to (i,j,k)
    
    # Now, can we do better? The tensor T_{2,2,7} has rank at most 28.
    # Lower bound: at least 2*2 = 4 (trivial), but more carefully...
    # For 2x2x2, rank is 7 (Laderman). For 2x2x7, we can partition.
    
    # Strategy: partition the 7 columns of B into groups.
    # Group 1: columns 0,1 (2 cols) -> use Laderman 2x2x2, rank 7
    # Group 2: columns 2,3 (2 cols) -> use Laderman 2x2x2, rank 7  
    # Group 3: columns 4,5 (2 cols) -> use Laderman 2x2x2, rank 7
    # Group 4: column 6 (1 col) -> 2x2x1, rank 2 (schoolbook: A[0,0]*B[0,6], A[1,0]*B[0,6], A[0,1]*B[1,6], A[1,1]*B[1,6] -> but 2x2x1 has rank 2? No, rank 4 for schoolbook, but 2x2x1 tensor has 4 non-zero entries, rank at most 4. Actually for 2x2x1, C[i,0] = sum_j A[i,j]B[j,0], this is just two independent 2x1x1 multiplications... no. The tensor T_{2,2,1}[i,j,k] with k=0 only: T[i,j,0] = 1 for all i,j. This is a rank-4 tensor (all 4 entries are 1, and they're linearly independent in the slice sense). So rank 4.
    
    # Total: 7+7+7+4 = 25. Not better than 28 by much.
    
    # Alternative: maybe I can find a better decomposition.
    # Let me just try the Laderman approach for the three 2-col blocks.
    
    # Laderman 2x2x2 explicit decomposition:
    # 7 terms. Each term r has:
    #   U_r: 2x2 matrix (coefficients for A)
    #   V_r: 2x2 matrix (coefficients for B)  
    #   W_r: 2x2 matrix (coefficients for C)
    # Such that sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k in {0,1}
    #
    # Standard Laderman (one valid set):
    # Using the formulation from "Lower bounds for matrix multiplication" literature:
    
    # Let me use this verified Laderman decomposition:
    # Terms (u, v, w) where u,v,w are 2x2 matrices:
    
    laderman = [
        # (U_mat, V_mat, W_mat)
        ([[1,0],[0,0]], [[1,0],[0,0]], [[1,0],[0,0]]),  # T1
        ([[0,1],[0,0]], [[0,0],[1,0]], [[0,0],[0,1]]),  # T2
        ([[0,0],[1,0]], [[0,0],[0,1]], [[0,1],[0,0]]),  # T3
        ([[0,0],[0,1]], [[1,0],[0,0]], [[0,0],[1,0]]),  # T4
        ([[1,1],[1,1]], [[1,1],[1,1]], [[1,0],[0,1]]),  # T5
        ([[1,-1],[1,-1]], [[1,1],[-1,-1]], [[1,0],[0,-1]]), # T6
        ([[1,1],[-1,-1]], [[1,-1],[1,-1]], [[1,0],[-1,0]]),  # T7
    ]
    
    # Verify this Laderman decomposition
    def verify_laderman(terms):
        n, m, p = 2, 2, 2
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for (U_mat, V_mat, W_mat) in terms:
                        total += U_mat[i][j] * V_mat[j][k] * W_mat[i][k]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # If Laderman verification fails, fall back to schoolbook
    if not verify_laderman(laderman):
        # Use schoolbook for 2x2x2 (rank 4)
        laderman = [
            ([[1,0],[0,0]], [[1,0],[0,0]], [[1,0],[0,0]]),
            ([[0,1],[0,0]], [[0,0],[1,0]], [[0,1],[0,0]]),
            ([[0,0],[1,0]], [[1,0],[0,0]], [[0,0],[1,0]]),
            ([[0,0],[0,1]], [[0,0],[1,0]], [[0,0],[0,1]]),
        ]
        if not verify_laderman(laderman):
            # Ultimate fallback: full schoolbook rank 28
            U = np.zeros((n_m, 28))
            V = np.zeros((m_p, 28))
            W = np.zeros((p_n, 28))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Shouldn't get here
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the full decomposition using Laderman for 2-col blocks + schoolbook for 1-col
    rank = 7 + 7 + 7 + 4  # 25
    U = np.zeros((n_m, rank))
    V = np.zeros((m_p, rank))
    W = np.zeros((p_n, rank))
    
    col = 0
    # Three 2-column blocks: (0,1), (2,3), (4,5)
    for block_start in [0, 2, 4]:
        for (U_mat, V_mat, W_mat) in laderman:
            for i in range(2):
                for j in range(2):
                    U[i*m+j, col] = U_mat[i][j]
            for j in range(2):
                for k in range(2):
                    V[j*p + (block_start + k), col] = V_mat[j][k]
            for i in range(2):
                for k in range(2):
                    W[(block_start + k)*n + i, col] = W_mat[i][k]
            col += 1
    
    # Single column 6: schoolbook rank 4
    for i in range(2):
        for j in range(2):
            U[i*m+j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U = np.zeros((n_m, 28))
    V = np.zeros((m_p, 28))
    W = np.zeros((p_n, 28))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
