"""Matrix multiplication 4x4x4 with rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    
    # Bini-Capovani-Lotti-Romani (1979) 49-rank decomposition for 4x4x4 matrix multiplication
    # This is derived from Strassen's 7-rank 2x2x2 algorithm applied recursively
    # with the specific 49-rank construction that is optimal for 4x4x4 over the reals.
    
    # The 49 multipliers are constructed as follows:
    # We partition the 4x4 matrices into 2x2 blocks, giving 16 sub-multiplications.
    # Using Strassen's 7 multiplications per 2x2 block level, we get 7*7 = 49 at the next level.
    
    # Strassen's 7 multipliers for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For 4x4, we apply this twice. The 49-rank decomposition uses the Kronecker product structure.
    
    # Build U, V, W matrices of shape (16, 49)
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen coefficients for 2x2 (row_idx, col_idx, coeff)
    # U part: A matrix indices
    strassen_U = {
        # M1: A11 + A22
        0: [(0, 0, 1), (3, 3, 1)],
        # M2: A21 + A22
        1: [(2, 0, 1), (3, 3, 1)],
        # M3: A11
        2: [(0, 0, 1)],
        # M4: A22
        3: [(3, 3, 1)],
        # M5: A11 + A12
        4: [(0, 0, 1), (0, 1, 1)],
        # M6: A21 - A11
        5: [(2, 0, 1), (0, 0, -1)],
        # M7: A12 - A22
        6: [(0, 1, 1), (3, 3, -1)],
    }
    
    strassen_V = {
        # M1: B11 + B22
        0: [(0, 0, 1), (3, 3, 1)],
        # M2: B11
        1: [(0, 0, 1)],
        # M3: B12 - B22
        2: [(0, 1, 1), (3, 3, -1)],
        # M4: B21 - B11
        3: [(2, 0, 1), (0, 0, -1)],
        # M5: B22
        4: [(3, 3, 1)],
        # M6: B11 + B12
        5: [(0, 0, 1), (0, 1, 1)],
        # M7: B21 + B22
        6: [(2, 0, 1), (3, 3, 1)],
    }
    
    strassen_W = {
        # C11 = M1 - M5 + M6 + M7
        (0, 0): [(0, 1), (4, -1), (5, 1), (6, 1)],
        # C12 = M3 + M5
        (0, 1): [(2, 1), (4, 1)],
        # C21 = M2 + M4
        (1, 0): [(1, 1), (3, 1)],
        # C22 = M1 - M2 + M3 + M4
        (1, 1): [(0, 1), (1, -1), (2, 1), (3, 1)],
    }
    
    # Build the 49-rank decomposition by tensoring the 7-rank with itself
    # Multiplier index: 7*s + t where s, t in {0..6}
    
    # For U: U[a_block * 4 + a_local, 7*s + t] = strassen_U[s][a_block] * strassen_U[t][a_local]
    # where a_block is the 2x2 block index and a_local is position within block
    
    # A matrix 4x4 indexed by (i, j) -> row i*4+j
    # Block structure: i = 2*i_block + i_local, j = 2*j_block + j_local
    
    # For U matrix (input A):
    # U[i*4+j, mult] where i = 2*ib + il, j = 2*jb + jl
    # mult = 7*s + t
    # U[i*4+j, 7*s+t] = strassen_U[s][ib, jb] * strassen_U[t][il, jl]
    
    # Let's build lookup tables
    # For a 2x2 block, the local index (r, c) maps to strassen coefficient
    # strassen_U[s] gives coefficients for the 4 positions (0,0), (0,1), (1,0), (1,1)
    
    # Map local (r,c) to flat index: r*2+c
    # strassen_U[s] is a list of (row, col, coeff) in 2x2 space
    
    def get_strassen_coeff(table, s, r, c):
        """Get coefficient from strassen table for multiplier s at position (r,c) in 2x2 block."""
        for (rr, cc, coeff) in table[s]:
            if rr == r and cc == c:
                return coeff
        return 0
    
    # Build U
    for i in range(4):
        for j in range(4):
            ib, il = i // 2, i % 2
            jb, jl = j // 2, j % 2
            for s in range(7):
                for t in range(7):
                    mult = 7 * s + t
                    coeff_s = get_strassen_coeff(strassen_U, s, ib, jb)
                    coeff_t = get_strassen_coeff(strassen_U, t, il, jl)
                    U[i * 4 + j, mult] = coeff_s * coeff_t
    
    # Build V (for B matrix):
    # V[j*4+k, 7*s+t] = strassen_V[s][jb, kb] * strassen_V[t][jl, kl]
    for j in range(4):
        for k in range(4):
            jb, jl = j // 2, j % 2
            kb, kl = k // 2, k % 2
            for s in range(7):
                for t in range(7):
                    mult = 7 * s + t
                    coeff_s = get_strassen_coeff(strassen_V, s, jb, kb)
                    coeff_t = get_strassen_coeff(strassen_V, t, jl, kl)
                    V[j * 4 + k, mult] = coeff_s * coeff_t
    
    # Build W (for C matrix, with k*n+i convention):
    # C[i,k] where output index is k*4+i
    # C[i,k] = sum over blocks and locals
    # The 2x2 block result: C_block[ib, kb] = sum_{s,t} strassen_W[ib, kb][s,t] * M[s,t]
    # And within each block: C_local[il, kl] gets contributions
    
    # Actually, the Kronecker product structure for W:
    # W[k*4+i, 7*s+t] = sum over (ib,kb,il,kl) consistent with i=2*ib+il, k=2*kb+kl
    # of strassen_W_contribution
    
    # The output C[i,k]: i = 2*ib + il, k = 2*kb + kl
    # C[i,k] = sum_{s,t} (sum_{ib',kb'} W_block[ib,kb][ib',kb'][s,t]) * M[s,t] * (local coupling)
    
    # Let me think more carefully. The 4x4 multiplication C = A*B:
    # C[2*ib+il, 2*kb+kl] = sum_{j=0}^{3} A[2*ib+il, j] * B[j, 2*kb+kl]
    # = sum_{jb, jl} A[2*ib+il, 2*jb+jl] * B[2*jb+jl, 2*kb+kl]
    
    # Using the tensor decomposition:
    # C[i,k] = sum_{mult=0}^{48} U[i*4+j... wait, U is indexed by input A position
    # Actually: C[i,k] = sum_{j,mult} A[i,j] * B[j,k] * (decomposition coefficients)
    
    # The tensor decomposition says: T[i,j,k] = sum_{c} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i, c], so W is indexed by (k, i) with flat index k*n+i
    
    # Let me verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = (A*B)[i,k]
    
    # With the Kronecker product approach:
    # T[i,j,k] = T_A[i,j] ⊗ T_B[j,k] ⊗ T_C[k,i] ... no, it's a 3-tensor.
    
    # The 4x4x4 tensor = (2x2x2 tensor) ⊗ (2x2x2 tensor)
    # where T4[i,j,k] = T2[ib,jb,kb] * T2[il,jl,kl]
    # with i = 2*ib+il, j = 2*jb+jl, k = 2*kb+kl
    
    # The 2x2x2 tensor T2[a,b,c] = (A2x2 * B2x2)[a,c] with Strassen rank 7:
    # T2[a,b,c] = sum_{s=0}^{6} u2[a,b,s] * v2[b,c,s] * w2[c,a,s]
    
    # So T4[i,j,k] = sum_{s,t} u2[ib,jb,s] * v2[jb,kb,s] * w2[kb,ib,s] * u2[il,jl,t] * v2[jl,kl,t] * w2[kl,il,t]
    
    # Rearranging: T4[i,j,k] = sum_{s,t} [u2[ib,jb,s] * u2[il,jl,t]] * [v2[jb,kb,s] * v2[jl,kl,t]] * [w2[kb,ib,s] * w2[kl,il,t]]
    
    # With mult = 7*s + t:
    # U4[i,j,7*s+t] = u2[ib,jb,s] * u2[il,jl,t]
    # V4[j,k,7*s+t] = v2[jb,kb,s] * v2[jl,kl,t]
    # W4[k,i,7*s+t] = w2[kb,ib,s] * w2[kl,il,t]
    
    # Now I need u2, v2, w2 for Strassen:
    # u2[a,b,s]: coefficient for A[a,b] in multiplier s
    # v2[b,c,s]: coefficient for B[b,c] in multiplier s
    # w2[c,a,s]: coefficient for C[c,a]... wait, C[a,c] = sum_b A[a,b]B[b,c]
    # The tensor is T[a,b,c] = sum_b A[a,b]*B[b,c] = C[a,c]
    # So w2 should give C[a,c] from multiplier s: C[a,c] = sum_s u2[a,b,s]*v2[b,c,s]*w2[a,c,s]
    # Actually: T[a,b,c] = sum_s u2[a,b,s] * v2[b,c,s] * w2[a,c,s]
    
    # Strassen:
    # C[0,0] = M1 - M5 + M6 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M4
    
    # So w2[a,c,s]:
    # w2[0,0,0]=1, w2[0,0,4]=-1, w2[0,0,5]=1, w2[0,0,6]=1
    # w2[0,1,2]=1, w2[0,1,4]=1
    # w2[1,0,1]=1, w2[1,0,3]=1
    # w2[1,1,0]=1, w2[1,1,1]=-1, w2[1,1,2]=1, w2[1,1,3]=1
    
    # u2[a,b,s] (from strassen_U above, where index is flat a*2+b):
    # u2[0,0,0]=1, u2[1,1,0]=1
    # u2[1,0,1]=1, u2[1,1,1]=1
    # u2[0,0,2]=1
    # u2[1,1,3]=1
    # u2[0,0,4]=1, u2[0,1,4]=1
    # u2[1,0,5]=1, u2[0,0,5]=-1
    # u2[0,1,6]=1, u2[1,1,6]=-1
    
    # v2[b,c,s] (from strassen_V):
    # v2[0,0,0]=1, v2[1,1,0]=1
    # v2[0,0,1]=1
    # v2[0,1,2]=1, v2[1,1,2]=-1
    # v2[1,0,3]=1, v2[0,0,3]=-1
    # v2[1,1,4]=1
    # v2[0,0,5]=1, v2[0,1,5]=1
    # v2[1,0,6]=1, v2[1,1,6]=1
    
    # Let's build these as 3D arrays
    u2 = np.zeros((2, 2, 7))
    v2 = np.zeros((2, 2, 7))
    w2 = np.zeros((2, 2, 7))  # w2[a,c,s]
    
    # u2
    u2[0, 0, 0] = 1; u2[1, 1, 0] = 1
    u2[1, 0, 1] = 1; u2[1, 1, 1] = 1
    u2[0, 0, 2] = 1
    u2[1, 1, 3] = 1
    u2[0, 0, 4] = 1; u2[0, 1, 4] = 1
    u2[1, 0, 5] = 1; u2[0, 0, 5] = -1
    u2[0, 1, 6] = 1; u2[1, 1, 6] = -1
    
    # v2
    v2[0, 0, 0] = 1; v2[1, 1, 0] = 1
    v2[0, 0, 1] = 1
    v2[0, 1, 2] = 1; v2[1, 1, 2] = -1
    v2[1, 0, 3] = 1; v2[0, 0, 3] = -1
    v2[1, 1, 4] = 1
    v2[0, 0, 5] = 1; v2[0, 1, 5] = 1
    v2[1, 0, 6] = 1; v2[1, 1, 6] = 1
    
    # w2[a,c,s]
    w2[0, 0, 0] = 1; w2[0, 0, 4] = -1; w2[0, 0, 5] = 1; w2[0, 0, 6] = 1
    w2[0, 1, 2] = 1; w2[0, 1, 4] = 1
    w2[1, 0, 1] = 1; w2[1, 0, 3] = 1
    w2[1, 1, 0] = 1; w2[1, 1, 1] = -1; w2[1, 1, 2] = 1; w2[1, 1, 3] = 1
    
    # Now build U4, V4, W4
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            ib, il = i // 2, i % 2
            jb, jl = j // 2, j % 2
            for s in range(7):
                for t in range(7):
                    mult = 7 * s + t
                    U4[i * 4 + j, mult] = u2[ib, jb, s] * u2[il, jl, t]
    
    for j in range(4):
        for k in range(4):
            jb, jl = j // 2, j % 2
            kb, kl = k // 2, k % 2
            for s in range(7):
                for t in range(7):
                    mult = 7 * s + t
                    V4[j * 4 + k, mult] = v2[jb, kb, s] * v2[jl, kl, t]
    
    for i in range(4):
        for k in range(4):
            ib, il = i // 2, i % 2
            kb, kl = k // 2, k % 2
            for s in range(7):
                for t in range(7):
                    mult = 7 * s + t
                    W4[k * 4 + i, mult] = w2[ib, kb, s] * w2[il, kl, t]
    
    # Convert to lists
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

