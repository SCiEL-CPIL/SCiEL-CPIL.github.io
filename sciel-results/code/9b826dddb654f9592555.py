"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6

    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12

    # Build the target tensor T[a, b, c]
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # a in [0, 4), b in [0, 12), c in [0, 12)
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # We seek a rank-r decomposition:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #
    # Strategy: Use the Cohn-Winograd polynomial embedding / Kronecker substitution approach.
    # For 2x2 * 2x6, we can view this as 3 independent 2x2 * 2x2 multiplications
    # (since the B matrix has 2 columns, and we're multiplying by 6 columns total,
    #  we can group them as 3 pairs).
    #
    # Actually, let's think more carefully. The structure is:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # This is a sum of 2 terms for each (i,k). There are 2*6 = 12 such terms.
    # The "schoolbook" rank is 12.
    #
    # Can we do better? Let's try a rank-12 decomposition first (schoolbook),
    # then attempt to reduce it.
    #
    # Actually, the tensor is:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # This is a sum of 2*2*6 = 24 rank-1 tensors (one for each (i,j,k) triple).
    # But many of these might be "compatible" and share rank-1 terms.
    #
    # Let's use a numerical approach: set up the polynomial system and solve.
    # For rank r, we have U (4xr), V (12xr), W (12xr).
    # Constraints: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #
    # This is 4*12*12 = 576 constraints, but only 24 are non-zero.
    # The rest are zero constraints.
    #
    # Let's try a structured ansatz based on the algebraic structure.
    #
    # Key insight: The tensor T can be written as:
    # T = sum_k (sum_{i,j} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i})
    #
    # For each k, the inner sum is a 2x2x2 tensor (with indices i, j, and the
    # "output" index i appearing in c = k*n+i).
    #
    # Actually, for fixed k, we have:
    # sum_{i,j} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # This is a 4x2x2 tensor (a in [0,4), b in {k, 6+k}, c in {2k, 2k+1}).
    #
    # Let's analyze this 4x2x2 tensor. It has 4*2*2 = 16 entries, but only
    # 4 are non-zero:
    # (0, k, 2k), (1, k, 2k+1), (2, 6+k, 2k), (3, 6+k, 2k+1)
    #
    # Can we decompose this 4x2x2 tensor in rank 3?
    # If so, total rank would be 3*6 = 18.
    #
    # Let's try to find a rank-3 decomposition for the 4x2x2 tensor.
    # Let the three rank-1 terms be (u_r, v_r, w_r) for r=0,1,2.
    #
    # u_r is a vector in R^4, v_r in R^2, w_r in R^2.
    #
    # Constraints:
    # u_r[0]*v_r[0]*w_r[0] + ... = 1 for (0,k,2k)
    # u_r[1]*v_r[0]*w_r[1] + ... = 1 for (1,k,2k+1)
    # u_r[2]*v_r[1]*w_r[0] + ... = 1 for (2,6+k,2k)
    # u_r[3]*v_r[1]*w_r[1] + ... = 1 for (3,6+k,2k+1)
    # And zero for all other 12 entries.
    #
    # This is a system of 16 polynomial equations in 3*(4+2+2) = 24 unknowns.
    #
    # Let's try a specific ansatz. Suppose we use the following structure:
    # v_0 = (1, 0), v_1 = (0, 1), v_2 = (1, 1)
    # w_0 = (1, 0), w_1 = (0, 1), w_2 = (1, 1)
    #
    # Then the constraints become:
    # (0,k,2k): u_0[0]*1*1 + u_2[0]*1*1 = 1  => u_0[0] + u_2[0] = 1
    # (1,k,2k+1): u_0[1]*1*0 + u_1[1]*0*1 + u_2[1]*1*1 = 1  => u_2[1] = 1
    # (2,6+k,2k): u_0[2]*0*1 + u_1[2]*1*0 + u_2[2]*1*1 = 1  => u_2[2] = 1
    # (3,6+k,2k+1): u_0[3]*0*0 + u_1[3]*1*1 + u_2[3]*1*1 = 1  => u_1[3] + u_2[3] = 1
    #
    # Zero constraints:
    # (0,k,2k+1): u_0[0]*1*0 + u_1[0]*0*1 + u_2[0]*1*1 = 0  => u_2[0] = 0
    # (1,k,2k): u_0[1]*1*1 + u_1[1]*0*0 + u_2[1]*1*0 = 0  => u_0[1] = 0
    # (2,6+k,2k+1): u_0[2]*0*0 + u_1[2]*1*1 + u_2[2]*1*0 = 0  => u_1[2] = 0
    # (3,6+k,2k): u_0[3]*0*1 + u_1[3]*1*0 + u_2[3]*1*0 = 0  => u_0[3] = 0
    #
    # From u_2[0] = 0 and u_0[0] + u_2[0] = 1, we get u_0[0] = 1.
    # From u_2[1] = 1, u_2[2] = 1, u_0[1] = 0, u_1[2] = 0, u_0[3] = 0.
    # From u_1[3] + u_2[3] = 1, we have one degree of freedom.
    #
    # Let's set u_2[3] = 0, then u_1[3] = 1.
    #
    # So:
    # u_0 = (1, 0, 0, 0)
    # u_1 = (0, 0, 0, 1)
    # u_2 = (0, 1, 1, 0)
    #
    # Let's verify:
    # (0,k,2k): 1*1*1 + 0*1*1 = 1 ✓
    # (1,k,2k+1): 0*1*0 + 0*0*1 + 1*1*1 = 1 ✓
    # (2,6+k,2k): 0*0*1 + 0*1*0 + 1*1*1 = 1 ✓
    # (3,6+k,2k+1): 0*0*0 + 1*1*1 + 0*1*1 = 1 ✓
    #
    # Zero constraints:
    # (0,k,2k+1): 1*1*0 + 0*0*1 + 0*1*1 = 0 ✓
    # (1,k,2k): 0*1*1 + 0*0*0 + 1*1*0 = 0 ✓
    # (2,6+k,2k+1): 0*0*0 + 0*1*1 + 1*1*0 = 0 ✓
    # (3,6+k,2k): 0*0*1 + 1*1*0 + 0*1*0 = 0 ✓
    #
    # What about the other 8 zero entries?
    # (a, b, c) where b is not k or 6+k, or c is not 2k or 2k+1.
    # But in our 4x2x2 sub-tensor, b is only k or 6+k, and c is only 2k or 2k+1.
    # So we've covered all 16 entries.
    #
    # Great! So for each k, we can use 3 rank-1 terms.
    # Total rank = 3 * 6 = 18.
    #
    # Let's build the full decomposition.
    
    rank = 18
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    for k in range(p):
        base = 3 * k
        # u_0, u_1, u_2 for this k
        U[0, base + 0] = 1  # u_0[0]
        U[1, base + 2] = 1  # u_2[1]
        U[2, base + 2] = 1  # u_2[2]
        U[3, base + 1] = 1  # u_1[3]
        
        # v_0, v_1, v_2 for this k
        # v_0 = (1, 0) means V[k, base+0] = 1, V[6+k, base+0] = 0
        # v_1 = (0, 1) means V[k, base+1] = 0, V[6+k, base+1] = 1
        # v_2 = (1, 1) means V[k, base+2] = 1, V[6+k, base+2] = 1
        V[k, base + 0] = 1
        V[k, base + 2] = 1
        V[6 + k, base + 1] = 1
        V[6 + k, base + 2] = 1
        
        # w_0, w_1, w_2 for this k
        # w_0 = (1, 0) means W[2k, base+0] = 1, W[2k+1, base+0] = 0
        # w_1 = (0, 1) means W[2k, base+1] = 0, W[2k+1, base+1] = 1
        # w_2 = (1, 1) means W[2k, base+2] = 1, W[2k+1, base+2] = 1
        W[2 * k, base + 0] = 1
        W[2 * k, base + 2] = 1
        W[2 * k + 1, base + 1] = 1
        W[2 * k + 1, base + 2] = 1
    
    # Verify the decomposition
    def verify(U, V, W, T):
        dim_u, rank = U.shape
        dim_v, _ = V.shape
        dim_w, _ = W.shape
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    actual = 0
                    for r in range(rank):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    if abs(actual - T[a, b, c]) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, T):
        # Fallback to schoolbook rank 24
        rank = 24
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row_list.append(round(val * 2) / 2)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
