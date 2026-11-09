"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We want to find a rank-minimal decomposition for 2x4 by 4x8 matrix multiplication.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all valid i,j,k.
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The trivial rank is n*m*p = 64.
    # Strassen-based block decomposition gives rank 56 (score 1.1429).
    
    # Let's try to find a better decomposition using a different structural approach.
    # 
    # Key insight: The 2x4x8 tensor can be viewed as 8 independent 2x4 vectors (one for each k).
    # C[:,k] = A @ B[:,k]
    # This is a linear map from B[:,k] (a 4-vector) to C[:,k] (a 2-vector), applied 8 times.
    #
    # Alternative view: The tensor is the Kronecker product structure.
    # T = sum over all (i,j,k) of e_i ⊗ e_j ⊗ e_k
    #
    # Let's try a rank-15 decomposition using the following idea:
    # Split the problem into smaller sub-problems and use known low-rank decompositions.
    #
    # Actually, let me try a direct construction with rank 15.
    # The idea: use 15 rank-1 tensors that together cover all 64 entries.
    #
    # For a 2x4x8 tensor, we need:
    # sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    #
    # Let me try a structured construction:
    # Use 3 terms for each of 5 "column groups" of k (sizes 2,2,2,1,1 or similar).
    # Or use 5 terms for each of 3 "row groups" of i,j.
    
    # Let me try a different approach: use the fact that this is a bilinear map.
    # The rank of the tensor is the minimum number of rank-1 tensors needed.
    
    # For now, let me implement the Strassen-based rank-56 decomposition as a baseline,
    # and then try to improve it.
    
    # Strassen-based rank-56 decomposition:
    # Split k into 4 groups of 2: K0={0,1}, K1={2,3}, K2={4,5}, K3={6,7}
    # For each group, split j into 2 groups of 2: J0={0,1}, J1={2,3}
    # Each sub-problem is 2x2x2, use Strassen rank 7.
    # Total: 4 * 2 * 7 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank))  # 8 x 56
    V = np.zeros((m * p, rank))  # 32 x 56
    W = np.zeros((p * n, rank))  # 16 x 56
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen coefficients for U (A matrix), V (B matrix), W (C matrix)
    # For a 2x2x2 sub-problem with local indices (i,j,k) in {0,1}:
    # U[i,j] for the A part, V[j,k] for the B part, W[k,i] for the C part.
    
    # Define the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs)
    # where each is a dict mapping (local_i, local_j) or (local_j, local_k) or (local_k, local_i) to coefficient.
    
    strassen_terms = [
        # Term 1: M1 = (a00+a11)(b00+b11)
        # U: (0,0)->1, (1,1)->1
        # V: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->1
        # W: (0,0)->1, (1,1)->1 (contributes to C00 and C11 with signs)
        {'U': {(0,0): 1, (1,1): 1},
         'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
         'W': {(0,0): 1, (1,1): 1}},  # C00 += M1, C11 += M1
        
        # Term 2: M2 = (a10+a11)b00
        {'U': {(1,0): 1, (1,1): 1},
         'V': {(0,0): 1},
         'W': {(0,1): 1, (1,1): -1}},  # C10 += M2, C11 -= M2
        
        # Term 3: M3 = a00(b01-b11)
        {'U': {(0,0): 1},
         'V': {(0,1): 1, (1,1): -1},
         'W': {(0,1): 1, (1,1): 1}},  # C01 += M3, C11 += M3
        
        # Term 4: M4 = a11(b10-b00)
        {'U': {(1,1): 1},
         'V': {(1,0): 1, (0,0): -1},
         'W': {(0,0): 1, (0,1): 1}},  # C00 += M4, C10 += M4
        
        # Term 5: M5 = (a00+a01)b11
        {'U': {(0,0): 1, (0,1): 1},
         'V': {(1,1): 1},
         'W': {(0,0): -1, (0,1): 1}},  # C00 -= M5, C01 += M5
        
        # Term 6: M6 = (a10-a00)(b00+b01)
        {'U': {(1,0): 1, (0,0): -1},
         'V': {(0,0): 1, (0,1): 1},
         'W': {(1,1): 1}},  # C11 += M6
        
        # Term 7: M7 = (a01-a11)(b10+b11)
        {'U': {(0,1): 1, (1,1): -1},
         'V': {(1,0): 1, (1,1): 1},
         'W': {(0,0): 1}},  # C00 += M7
    ]
    
    # Now, for each of the 8 sub-problems (4 k-groups x 2 j-groups), assign 7 ranks.
    k_groups = [(0,1), (2,3), (4,5), (6,7)]
    j_groups = [(0,1), (2,3)]
    
    rank_offset = 0
    for kg_idx, (k_lo, k_hi) in enumerate(k_groups):
        for jg_idx, (j_lo, j_hi) in enumerate(j_groups):
            for s_idx, term in enumerate(strassen_terms):
                r = rank_offset + s_idx
                # Local indices: i in {0,1}, j_local in {0,1}, k_local in {0,1}
                # Global: i in {0,1}, j = j_lo + j_local, k = k_lo + k_local
                for (li, lj), u_val in term['U'].items():
                    U[li * m + (j_lo + lj), r] = u_val
                for (lj, lk), v_val in term['V'].items():
                    V[(j_lo + lj) * p + (k_lo + lk), r] = v_val
                for (lk, li), w_val in term['W'].items():
                    W[(k_lo + lk) * n + li, r] = w_val
            rank_offset += 7
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    valid, info = verify(U, V, W, n, m, p)
    if not valid:
        # Fallback to rank 64
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
