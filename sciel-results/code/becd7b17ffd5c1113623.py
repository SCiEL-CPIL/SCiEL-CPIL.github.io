"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    def schoolbook(n_, m_, p_):
        """Return (U, V, W) as numpy arrays for schoolbook decomposition of n_ x m_ x p_."""
        rank = n_ * m_ * p_
        U = np.zeros((n_ * m_, rank))
        V = np.zeros((m_ * p_, rank))
        W = np.zeros((p_ * n_, rank))
        c = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i * m_ + j, c] = 1.0
                    V[j * p_ + k, c] = 1.0
                    W[k * n_ + i, c] = 1.0
                    c += 1
        return U, V, W
    
    def strassen_222():
        """Return (U, V, W) as numpy arrays for Strassen 2x2x2 decomposition."""
        # Strassen's algorithm: 7 multiplications
        # U indices: (i, j) where i,j in {0,1} -> row index i*2+j
        # V indices: (j, k) where j,k in {0,1} -> row index j*2+k
        # W indices: (k, i) where k,i in {0,1} -> row index k*2+i
        
        # The 7 terms are:
        # m1 = (A11 + A22)(B11 + B22)
        # m2 = (A21 + A22)B11
        # m3 = A11(B21 - B22)
        # m4 = A22(B12 - B11)
        # m5 = (A11 + A12)B22
        # m6 = (A21 - A11)(B11 + B12)
        # m7 = (A12 - A22)(B21 + B22)
        
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # m1: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1  # C11+=, C12-=, C21+=, C22-=
        
        # m2: (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1  # A21, A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1; W[3, 1] = -1  # C21+=, C22-=
        
        # m3: A11(B21-B22)
        U[0, 2] = 1  # A11
        V[2, 2] = 1; V[3, 2] = -1  # B21, B22
        W[1, 2] = 1; W[3, 2] = 1  # C12+=, C22+=
        
        # m4: A22(B12-B11)
        U[3, 3] = 1  # A22
        V[1, 3] = 1; V[0, 3] = -1  # B12, B11
        W[0, 3] = 1; W[2, 3] = 1  # C11+=, C21+=
        
        # m5: (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1  # A11, A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1; W[1, 4] = 1  # C11-=, C12+=
        
        # m6: (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1  # A21, A11
        V[0, 5] = 1; V[1, 5] = 1  # B11, B12
        W[3, 5] = 1  # C22+=
        
        # m7: (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1  # A12, A22
        V[2, 6] = 1; V[3, 6] = 1  # B21, B22
        W[0, 6] = 1  # C11+=
        
        return U, V, W
    
    def strassen_333():
        """
        Return (U, V, W) for a 3x3x3 decomposition.
        We decompose 3x3 into:
          - One 2x2x2 Strassen block (top-left 2x2)
          - Remaining terms handled by schoolbook for the cross terms
        
        Actually, let's use a simpler approach:
        Split 3 = 2 + 1.
        The 3x3 matrix multiplication can be viewed as:
        [A11 A12 A13]   [B11 B12 B13]
        [A21 A22 A23] * [B21 B22 B23]
        [A31 A32 A33]   [B31 B32 B33]
        
        We can group as:
        [A22 A23] [B22 B23]  -> 2x2 block using Strassen
        [A32 A33] [B32 B33]  -> cross terms
        [A11 A12] [B11 B12]  -> cross terms
        [A13]     [B13]      -> 1x1
        etc.
        
        Let's use a cleaner decomposition:
        Partition indices {0,1,2} into {0,1} and {2}.
        
        The full 3x3x3 tensor has 27 terms.
        We can decompose it as:
        - Strassen on the (0,1)x(0,1) subblock: rank 7
        - Schoolbook for all other terms: 27 - 4 = 23 terms
        
        Total rank: 7 + 23 = 30
        
        But wait, we need to be more careful. The Strassen decomposition handles the 4 output entries C00, C01, C10, C11 
        using the 4x4 submatrices of A and B. The remaining 5 output entries (C02, C12, C20, C21, C22) 
        plus the cross terms that feed into C00-C11 from the 3rd row/col need schoolbook.
        
        Actually, the cleanest way:
        - Use Strassen for the 2x2x2 block on indices {0,1}x{0,1}x{0,1}: rank 7
        - Use schoolbook for all other (i,j,k) triples: 27 - 8 = 19 terms
        
        Total: 7 + 19 = 26
        """
        rank_strassen = 7
        rank_school = 19
        total_rank = rank_strassen + rank_school
        
        U = np.zeros((9, total_rank))
        V = np.zeros((9, total_rank))
        W = np.zeros((9, total_rank))
        
        # First, Strassen part for i,j,k in {0,1}
        S_U, S_V, S_W = strassen_222()
        # Map 2x2 indices to 3x3: index 0->0, 1->1
        for r in range(4):
            U[r, :rank_strassen] = S_U[r, :]
            V[r, :rank_strassen] = S_V[r, :]
            W[r, :rank_strassen] = S_W[r, :]
        
        # Schoolbook for remaining 19 terms
        c = rank_strassen
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i < 2 and j < 2 and k < 2:
                        continue  # Already handled by Strassen
                    U[i * 3 + j, c] = 1.0
                    V[j * 3 + k, c] = 1.0
                    W[k * 3 + i, c] = 1.0
                    c += 1
        
        return U, V, W
    
    def combine_blocks(n_blocks, m_blocks, p_blocks, block_decomps):
        """
        Combine block decompositions.
        n_blocks: list of n-sizes for each block
        m_blocks: list of m-sizes for each block  
        p_blocks: list of p-sizes for each block
        block_decomps: list of (U, V, W) for each block
        
        Actually, this is getting complex. Let me use a simpler approach.
        """
        pass
    
    # Strategy: Decompose 3x5x5 using recursive splitting
    # 3 = 2 + 1, 5 = 2 + 3
    
    # Let's try: R(3,5,5) = R(2,5,5) + R(1,5,5)
    # R(1,5,5) = 5 (schoolbook for 1x5x5)
    # R(2,5,5) = R(2,2,5) + R(2,3,5)
    # R(2,2,5) = R(2,2,2) + R(2,2,3) = 7 + R(2,2,3)
    # R(2,2,3) = R(2,2,1) + R(2,2,2) = 4 + 7 = 11 (schoolbook for 2x2x1 is 4)
    # So R(2,2,5) = 7 + 11 = 18
    # R(2,3,5) = R(2,1,5) + R(2,2,5) = 10 + 18 = 28 (schoolbook for 2x1x5 is 10)
    # R(2,5,5) = 18 + 28 = 46
    # R(3,5,5) = 46 + 5 = 51
    
    # Build this decomposition step by step
    
    def build_decomp(n, m, p):
        """
        Build a decomposition for n x m x p using recursive splitting.
        Returns (U, V, W) as numpy arrays.
        """
        # Base case: if any dimension is 1, use schoolbook
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
        
        # If all dimensions are 2, use Strassen
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
        
        # Otherwise, split the largest dimension
        # Prefer splitting p first, then m, then n
        # Split strategy: find a split point for the largest dim
        
        # Let's split p into p1 + p2 where p1 = min(2, p//2)
        # Actually, let's be more systematic
        
        # Split p: p = p1 + p2, p1 = min(2, p - 1)
        p1 = min(2, p - 1)
        p2 = p - p1
        
        # R(n, m, p) = R(n, m, p1) + R(n, m, p2)
        U1, V1, W1 = build_decomp(n, m, p1)
        U2, V2, W2 = build_decomp(n, m, p2)
        
        # Concatenate: the V and W matrices need to be partitioned by the p index
        # V has shape (m*p, rank). For V1, it corresponds to rows j*p1+k where j in [0,m), k in [0,p1)
        # For V2, it corresponds to rows j*p2+k where j in [0,m), k in [0,p2)
        # We need to embed these into the full (m*p, total_rank) space
        
        total_rank = U1.shape[1] + U2.shape[1]
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        # U: same row structure for both, just concatenate columns
        U_full[:, :U1.shape[1]] = U1
        U_full[:, U1.shape[1]:] = U2
        
        # V: need to scatter rows
        # V1 has rows indexed by (j, k1) where j in [0,m), k1 in [0,p1)
        # In the full V, these correspond to rows (j, k1) where the full k index is k1
        # V2 has rows indexed by (j, k2) where j in [0,m), k2 in [0,p2)
        # In the full V, these correspond to rows (j, k1+k2)
        
        for j in range(m):
            for k1 in range(p1):
                V_full[j * p + k1, :U1.shape[1]] = V1[j * p1 + k1, :]
            for k2 in range(p2):
                V_full[j * p + p1 + k2, U1.shape[1]:] = V2[j * p2 + k2, :]
        
        # W: W has shape (p*n, rank). Index is (k, i) where k in [0,p), i in [0,n)
        # W1 corresponds to k in [0, p1), W2 to k in [p1, p)
        for k1 in range(p1):
            for i in range(n):
                W_full[k1 * n + i, :U1.shape[1]] = W1[k1 * n + i, :]
        for k2 in range(p2):
            for i in range(n):
                W_full[(p1 + k2) * n + i, U1.shape[1]:] = W2[k2 * n + i, :]
        
        return U_full, V_full, W_full
    
    # Try building the decomposition
    try:
        U, V, W = build_decomp(n, m, p)
        
        # Verify the decomposition
        # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-9:
            # Convert to lists of integers/half-integers
            U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(2*x)/2 for x in row] for row in U.tolist()]
            V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(2*x)/2 for x in row] for row in V.tolist()]
            W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(2*x)/2 for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # If verification fails, fall back to schoolbook
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = schoolbook(n, m, p)
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(2*x)/2 for x in row] for row in U.tolist()]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(2*x)/2 for x in row] for row in V.tolist()]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(2*x)/2 for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
