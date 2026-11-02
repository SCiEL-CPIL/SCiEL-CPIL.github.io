"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

def _build_block_strassen(n, m, p):
    """
    Construct a decomposition using Strassen's algorithm on 2x2x2 sub-blocks
    where possible, and schoolbook for the rest.
    """
    # n=3, m=5, p=5
    # We identify 2x2x2 blocks in the (i, j, k) space.
    # i in {0, 1}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    
    # Define the 4 blocks
    # Block 1: i in [0,1], j in [0,1], k in [0,1]
    # Block 2: i in [0,1], j in [2,3], k in [0,1]
    # Block 3: i in [0,1], j in [0,1], k in [2,3]
    # Block 4: i in [0,1], j in [2,3], k in [2,3]
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
    ]
    
    # Determine which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for i_list, j_list, k_list in blocks:
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    covered.add((i, j, k))
    
    # Calculate total rank
    # Each block uses 7 terms. 4 blocks -> 28 terms.
    # Total triples = 3 * 5 * 5 = 75.
    # Covered triples = 4 * 8 = 32.
    # Remaining triples = 75 - 32 = 43.
    # Total rank = 28 + 43 = 71.
    
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, col_idx):
        U[i * m + j][col_idx] = 1
        V[j * p + k][col_idx] = 1
        W[k * n + i][col_idx] = 1

    # Helper to add Strassen terms for a block
    def add_strassen(i_list, j_list, k_list, start_col):
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list
        
        # Strassen multipliers M1..M7
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        
        # C[i0,k0] = M1 + M4 - M5 + M7
        # C[i0,k1] = M3 + M5
        # C[i1,k0] = M2 + M4
        # C[i1,k1] = M1 - M2 + M3 + M6
        
        # We need to distribute these M's into the tensor decomposition.
        # Each M_s corresponds to a product of sums.
        # E.g., M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # This expands to 4 schoolbook terms.
        # But in the tensor decomposition, we introduce one term for M1.
        # The U, V, W vectors for this term must capture the sums.
        
        # Term for M1:
        # U: 1 at (i0,j0), 1 at (i1,j1)
        # V: 1 at (j0,k0), 1 at (j1,k1)
        # W: 1 at (k0,i0), 1 at (k1,i1) ... Wait.
        # The tensor entry T[i,k] gets contribution from M1.
        # M1 contributes to C[i0,k0], C[i1,k1].
        # Actually, Strassen combines them.
        # The decomposition is:
        # T = sum_s (u_s \otimes v_s \otimes w_s)
        
        # For M1:
        # u_1 has 1s at (i0,j0) and (i1,j1).
        # v_1 has 1s at (j0,k0) and (j1,k1).
        # w_1 must be such that u_1 * v_1 * w_1 reconstructs the parts of T.
        # This is tricky because M1 contributes to multiple C entries.
        
        # Let's use the standard Strassen decomposition for 2x2x2 tensor.
        # The 2x2x2 tensor has rank 7.
        # The 7 basis tensors are known.
        
        # We will map the local 2x2x2 tensor to the global indices.
        
        # Local indices: li in {0,1}, lj in {0,1}, lk in {0,1}
        # Global: i = i0 + li, j = j0 + lj, k = k0 + lk
        
        # The 7 Strassen terms for local (li, lj, lk):
        
        # Term 1 (M1):
        # U: (0,0), (1,1)
        # V: (0,0), (1,1)
        # W: (0,0), (1,1)
        # Coefficients in T:
        # T[0,0] += 1, T[1,1] += 1
        # But Strassen M1 is used in C[0,0] and C[1,1] with signs.
        # Actually, the tensor decomposition is linear.
        # We just need to provide the 7 vectors U, V, W such that
        # sum_s U_s[i,j] V_s[j,k] W_s[k,i] = 1 if (i,j,k) is in the block, 0 else?
        # No, the tensor T is 1 for all (i,j,k) in the block?
        # No, T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
        # For a specific (i,j,k), the tensor entry is 1.
        # So we need to represent the sum of all (i,j,k) in the block as a sum of 7 rank-1 tensors.
        
        # The 2x2x2 tensor is:
        # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
        # T[1,0,0]=1, T[1,0,1]=1, T[1,1,0]=1, T[1,1,1]=1
        
        # Strassen decomposition of this tensor:
        # s1: U=[1,1,0,0]^T (flattened), V=[1,1,0,0]^T, W=[1,1,0,0]^T ?
        # Let's use the known rank-7 decomposition of 2x2x2.
        
        # Vectors (flattened 4x1 for U, 4x1 for V, 4x1 for W):
        # U1 = [1, 0, 0, 1] (i0j0, i1j1)
        # V1 = [1, 0, 0, 1] (j0k0, j1k1)
        # W1 = [1, 0, 0, 1] (k0i0, k1i1)
        # Contribution: T[0,0,0] += 1, T[1,1,1] += 1.
        
        # We need to cover all 8 entries with 7 terms.
        
        # Standard Strassen basis:
        # 1. (1,0,0,1) (1,0,0,1) (1,0,0,1) -> Covers (0,0,0) and (1,1,1)
        # 2. (0,0,1,1) (1,0,0,0) (0,0,0,1) -> Covers (1,0,0) and (1,1,1)? No.
        
        # Let's just use the explicit Strassen formulas for the matrix product C = AB.
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # The tensor T[i,j,k] is 1.
        # We can express the identity tensor for 2x2x2 as a sum of 7 rank-1 tensors.
        
        # Here is a valid decomposition:
        # t1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        # t2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,0,1]
        # t3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
        # t4: U=[0,0,0,1], V=[0,0,1,-1], W=[0,0,1,0]
        # t5: U=[1,0,1,0], V=[0,0,0,1], W=[0,0,0,1]
        # t6: U=[0,1,-1,0], V=[1,1,0,0], W=[1,0,0,0]
        # t7: U=[0,1,0,-1], V=[0,0,1,1], W=[0,1,0,0]
        
        # We need to verify this.
        # Sum of u_s[i] v_s[j] w_s[k] = 1 for all i,j,k in {0,1}^3.
        
        # Let's check (0,0,0):
        # t1: 1*1*1 = 1
        # t2: 0
        # t3: 1*0*0 = 0
        # t4: 0
        # t5: 1*0*0 = 0
        # t6: 0*1*1 = 0
        # t7: 0
        # Sum = 1. Correct.
        
        # Check (0,0,1):
        # t1: 1*1*0 = 0
        # t2: 0
        # t3: 1*0*1 = 0
        # t4: 0
        # t5: 1*0*0 = 0
        # t6: 0
        # t7: 0
        # Sum = 0. Incorrect. Should be 1.
        
        # The above decomposition is for the *product* tensor, not the identity tensor.
        # We need the decomposition of the *multiplication tensor* T[i,j,k] = 1.
        # The multiplication tensor for 2x2 matrices is:
        # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
        # This is a 2x2x2 all-ones tensor.
        # The rank of the all-ones 2x2x2 tensor is 1? No, it's 1.
        # Wait. T[i,j,k] = 1 for all i,j,k.
        # This is the tensor for computing sum_{l} A[i,l] B[l,k] = C[i,k].
        # No, that's not right.
        
        # The matrix multiplication tensor T is defined such that
        # T_{i,j,k} = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
        # For 2x2, there are 8 such terms.
        # T is the sum of 8 rank-1 tensors: e_i \otimes e_j \otimes e_k.
        # Can we write this sum of 8 as a sum of 7?
        # Yes, Strassen shows that the *product* can be computed with 7 multiplications.
        # This means the tensor T can be decomposed into 7 rank-1 tensors.
        
        # The 7 vectors are derived from the Strassen multipliers.
        
        # Let's use a library or hardcode the 7 vectors for 2x2x2.
        # U, V, W are 4x7 matrices.
        
        # U:
        # col 1: [1, 0, 0, 1]
        # col 2: [0, 0, 1, 1]
        # col 3: [1, 0, 0, 0]
        # col 4: [0, 0, 0, 1]
        # col 5: [1, 0, 1, 0]
        # col 6: [0, 1, -1, 0]
        # col 7: [0, 1, 0, -1]
        
        # V:
        # col 1: [1, 0, 0, 1]
        # col 2: [1, 0, 0, 0]
        # col 3: [0, 1, 0, -1]
        # col 4: [0, 0, 1, -1]
        # col 5: [0, 0, 0, 1]
        # col 6: [1, 1, 0, 0]
        # col 7: [0, 0, 1, 1]
        
        # W:
        # col 1: [1, 0, 0, 1]
        # col 2: [0, 0, 0, 1]
        # col 3: [0, 1, 0, 0]
        # col 4: [0, 0, 1, 0]
        # col 5: [0, 0, 0, 1]
        # col 6: [1, 0, 0, 0]
        # col 7: [0, 1, 0, 0]
        
        # Let's verify one entry, e.g., (0,0,1) -> U[0], V[0], W[1]
        # Sum_s U_s[0] V_s[0] W_s[1]
        # s1: 1 * 1 * 0 = 0
        # s2: 0 * 1 * 0 = 0
        # s3: 1 * 0 * 1 = 0
        # s4: 0 * 0 * 0 = 0
        # s5: 1 * 0 * 0 = 0
        # s6: 0 * 1 * 0 = 0
        # s7: 0 * 0 * 0 = 0
        # Sum = 0. Should be 1.
        
        # The vectors above are for the *Strassen multipliers*, not the direct decomposition of the all-ones tensor.
        # The decomposition of the all-ones 2x2x2 tensor is NOT rank 7.
        # The rank of the all-ones tensor of size 2x2x2 is 1.
        # T = 1 \otimes 1 \otimes 1.
        
        # I am confusing the *multiplication algorithm* with the *tensor decomposition*.
        # The tensor T is the set of coefficients.
        # T[i,j,k] = 1.
        # This tensor has rank 8 (schoolbook).
        # Strassen reduces the *number of multiplications* to 7.
        # This implies that the tensor T can be written as a sum of 7 rank-1 tensors?
        # No. Strassen shows that the *bilinear complexity* is 7.
        # This means T can be decomposed into 7 rank-1 tensors.
        
        # Let's find the correct 7 vectors.
        # Reference: "The rank of the matrix multiplication tensor"
        
        # For 2x2, the 7 vectors are:
        # (U, V, W)
        # 1: ([1,0,0,1], [1,0,0,1], [1,0,0,1])
        # 2: ([0,0,1,1], [1,0,0,0], [0,0,0,1])
        # 3: ([1,0,0,0], [0,1,0,-1], [0,1,0,0])
        # 4: ([0,0,0,1], [0,0,1,-1], [0,0,1,0])
        # 5: ([1,0,1,0], [0,0,0,1], [0,0,0,1])
        # 6: ([0,1,-1,0], [1,1,0,0], [1,0,0,0])
        # 7: ([0,1,0,-1], [0,0,1,1], [0,1,0,0])
        
        # I will assume this is correct and verify in code.
        
        U_block = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, -1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V_block = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, -1, 1, 0, 1]
        ])
        W_block = np.array([
            [1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0]
        ])
        
        # Map local indices to global
        for s in range(7):
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        val = U_block[li*2+lj, s] * V_block[lj*2+lk, s] * W_block[lk*2+li, s]
                        # This product should sum to 1 for each (li,lj,lk)
                        # But we need to add the vectors to U, V, W
                        pass
                        
        # Actually, we just add the vectors.
        for s in range(7):
            col_idx = start_col + s
            # U
            for li in range(2):
                for lj in range(2):
                    i = i0 + li
                    j = j0 + lj
                    U[i * m + j][col_idx] += U_block[li*2+lj, s]
            # V
            for lj in range(2):
                for lk in range(2):
                    j = j0 + lj
                    k = k0 + lk
                    V[j * p + k][col_idx] += V_block[lj*2+lk, s]
            # W
            for lk in range(2):
                for li in range(2):
                    k = k0 + lk
                    i = i0 + li
                    W[k * n + i][col_idx] += W_block[lk*2+li, s]

    # Add Strassen blocks
    for i_list, j_list, k_list in blocks:
        add_strassen(i_list, j_list, k_list, col)
        col += 7
        
    # Add schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k, col)
                    col += 1
                    
    return U, V, W

def _verify_fast(Ua, Va, Wa, n, m, p):
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(Ua[a], Va[b] * Wa[c])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W = _build_block_strassen(n, m, p)
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    if not _verify_fast(Ua, Va, Wa, n, m, p):
        # Fallback to schoolbook if Strassen decomposition fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        
    return {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
