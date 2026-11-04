import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 110.0
    
    n, m, p = 4, 4, 9
    
    # 1. Strassen 2x2x2 (Rank 7)
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ], dtype=float)
    V2 = np.array([
        [1, -1, 0, 1, -1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 1]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 1]
    ], dtype=float)

    # 2. Construct 4x4x4 Strassen (Rank 49) via Kronecker of 2x2x2
    r4 = 49
    U4 = np.zeros((16, r4))
    V4 = np.zeros((16, r4))
    W4 = np.zeros((16, r4))
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            for c1 in range(7):
                                for c2 in range(7):
                                    idx_c = c1*7 + c2
                                    
                                    u1 = U2[i1*2+j1, c1]
                                    u2 = U2[i2*2+j2, c2]
                                    v1 = V2[j1*2+k1, c1]
                                    v2 = V2[j2*2+k2, c2]
                                    w1 = W2[k1*2+i1, c1]
                                    w2 = W2[k2*2+i2, c2]
                                    
                                    idx_u = (i1*2+i2)*4 + (j1*2+j2)
                                    idx_v = (j1*2+j2)*4 + (k1*2+k2)
                                    idx_w = (k1*2+k2)*4 + (i1*2+i2)
                                    
                                    U4[idx_u, idx_c] += u1 * u2
                                    V4[idx_v, idx_c] += v1 * v2
                                    W4[idx_w, idx_c] += w1 * w2

    # 3. Construct 4x4x1 Schoolbook (Rank 16)
    r1 = 16
    U1 = np.zeros((16, r1))
    V1 = np.zeros((4, r1))
    W1 = np.zeros((4, r1))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i*4+j, col] = 1
            V1[j, col] = 1
            W1[i, col] = 1
            col += 1

    # 4. Assemble 4x4x9
    # Total Rank = 49 + 49 + 16 = 114
    R_init = 114
    U = np.zeros((16, R_init))
    V = np.zeros((36, R_init)) # 4*9
    W = np.zeros((36, R_init)) # 9*4
    
    # Block 1: k=0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*9+k, 0:49] = V4[j*4+k, :]
    for i in range(4):
        for k in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, :]

    # Block 2: k=4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*9+k, 49:98] = V4[j*4+(k-4), :]
    for i in range(4):
        for k in range(4, 8):
            W[k*4+i, 49:98] = W4[(k-4)*4+i, :]

    # Block 3: k=8
    U[:, 98:114] = U1
    for j in range(4):
        V[j*9+8, 98:114] = V1[j, :]
    for i in range(4):
        W[8*4+i, 98:114] = W1[i, :]

    # 5. Local Search for Rank Reduction
    # Strategy: Try to remove columns one by one.
    # If we remove column c, we need to adjust the remaining columns such that the tensor equation holds.
    # This is a linear system problem.
    # Let T be the tensor. T = sum_c u_c v_c^T w_c^T.
    # If we fix all columns except c, we need to find u_c, v_c, w_c such that the residual is matched.
    # Actually, a simpler approach: 
    # Check if column c is a linear combination of others in a way that preserves the tensor?
    # No, that's not how rank works.
    
    # Better approach: Randomized Greedy Removal with Repair.
    # 1. Pick a random column c.
    # 2. Set it to 0.
    # 3. Calculate the residual tensor R = T_target - T_current.
    # 4. Try to represent R as a sum of K rank-1 terms (where K is small, e.g., 1 or 2) by modifying existing columns?
    # This is hard.
    
    # Alternative Structural Idea:
    # The tensor for 4x4x9 is the sum of two 4x4x4 tensors and one 4x4x1 tensor.
    # Is there a way to mix the Strassen blocks with the Schoolbook tail?
    # Probably not easily without breaking the structure.
    
    # Let's try a different structural decomposition:
    # 4x4x9 can be seen as 4x4x8 + 4x4x1.
    # 4x4x8 = two 4x4x4 blocks.
    # What if we use a different algorithm for 4x4x4?
    # The current one is Kronecker of Strassen (49).
    # Is there a better 4x4x4?
    # Rank of 4x4x4 is known to be at most 49 (Strassen) and at least 43 (Biswas).
    # Finding a rank 43-48 decomposition is an open problem / very hard.
    # So 49 is likely the best we can get for the blocks.
    
    # What about the 4x4x1 part? Rank 16 is optimal (schoolbook).
    
    # So the bottleneck is the 49+49+16 = 114.
    # Can we share computation between the two 4x4x4 blocks?
    # The two blocks are A*B1 and A*B2.
    # They share the same A.
    # In the tensor decomposition, the U matrix (coefficients for A) is the same for both blocks.
    # The V matrix (coefficients for B) is different.
    # The W matrix (coefficients for C) is different.
    # In our construction, U[:, 0:49] == U[:, 49:98].
    # This means the 98 columns of U are not linearly independent?
    # Wait, U is 16x114. The first 49 columns are identical to the next 49 columns.
    # This implies the rank of U is at most 49+16 = 65? No, U is just a matrix.
    # But the tensor rank is determined by the number of terms in the sum.
    # If U[:, c1] == U[:, c2], does that mean term c1 and c2 are dependent?
    # Term c: u_c v_c^T w_c^T.
    # If u_1 = u_2, v_1 != v_2, w_1 != w_2, they are still independent rank-1 tensors.
    # However, we might be able to merge them?
    # u (v1^T w1^T + v2^T w2^T)? No, that's not a rank-1 tensor.
    
    # Let's look at the structure again.
    # T = sum_{c=1}^{49} u_c v_{c,1}^T w_{c,1}^T + sum_{c=1}^{49} u_c v_{c,2}^T w_{c,2}^T + sum_{c=1}^{16} u'_c v'_{c}^T w'_{c}^T
    # Notice that u_c is the same for block 1 and block 2.
    # This suggests a potential for compression if we can combine v and w.
    # But v and w act on different subspaces of B and C.
    # B is 4x9. Block 1 uses B[:, 0:4], Block 2 uses B[:, 4:8].
    # C is 9x4. Block 1 uses C[0:4, :], Block 2 uses C[4:8, :].
    # These subspaces are disjoint.
    # So v_{c,1} is non-zero only in indices corresponding to B[:, 0:4].
    # v_{c,2} is non-zero only in indices corresponding to B[:, 4:8].
    # Thus v_{c,1} and v_{c,2} are orthogonal (disjoint support).
    # Similarly for w.
    # This means the rank-1 tensors from Block 1 and Block 2 are orthogonal in the V and W spaces.
    # They cannot be merged.
    
    # What about the tail?
    # Tail uses B[:, 8] and C[8, :].
    # Also disjoint.
    
    # So the 114 rank seems structurally minimal for this specific decomposition.
    # Is there a different decomposition?
    # Maybe not using Strassen for 4x4x4?
    # If we use Schoolbook for 4x4x4, rank is 16.
    # Total: 16 + 16 + 16 = 48?
    # Wait.
    # 4x4x4 Schoolbook rank is 16.
    # So 4x4x8 would be 16+16=32.
    # 4x4x1 is 16.
    # Total 48.
    # Wait, 4*4*4 = 64 operations. Schoolbook rank is 16?
    # Yes, for C = AB, C_ij = sum_k A_ik B_kj.
    # Tensor is sum_{i,j,k} A_ik B_kj C_ij.
    # Rank is n*m*p? No.
    # For 2x2x2, Schoolbook rank is 8. Strassen is 7.
    # For 4x4x4, Schoolbook rank is 64.
    # My previous thought "Schoolbook 4x4x4 rank 16" was WRONG.
    # Schoolbook rank is n*m*p.
    # So 4x4x4 Schoolbook is 64.
    # Strassen is 49.
    # So Strassen is better.
    
    # Let's re-verify the current rank.
    # Block 1 (4x4x4): Strassen 49.
    # Block 2 (4x4x4): Strassen 49.
    # Block 3 (4x4x1): Schoolbook 16.
    # Total 114.
    
    # Is there a better way to do 4x4x9?
    # Maybe 4x4x9 is not the best split?
    # What if we do 4x4x9 as a single tensor?
    # The rank of 4x4x9 is unknown but bounded by 4*4*9 = 144 (Schoolbook) and 49+49+16=114 (Strassen+Schoolbook).
    # Can we do better than 114?
    # Perhaps by mixing the blocks?
    # Or using a different algorithm for 4x4x4?
    # If we could find a rank 43 decomposition for 4x4x4, we would have 43+43+16 = 102.
    # But finding that is hard.
    
    # Let's try a random local search to see if 114 can be reduced.
    # We will try to remove columns and solve for the rest.
    
    def validate(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not validate(U, V, W):
        # Fallback to Schoolbook 144
        rank = n * m * p
        U_f = np.zeros((n*m, rank))
        V_f = np.zeros((m*p, rank))
        W_f = np.zeros((p*n, rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_f[ii*m+jj, col] = 1
                    V_f[jj*p+kk, col] = 1
                    W_f[kk*n+ii, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    # Local Search: Try to reduce rank by 1
    # This is a heuristic. We pick a random column, zero it out, and try to fix the error
    # by adjusting one other column? No, that's not possible generally.
    
    # Instead, let's try to find if any column is redundant.
    # A column c is redundant if u_c v_c^T w_c^T is in the span of the other tensors.
    # This is a very high-dimensional check.
    
    # Given the time limit and the difficulty, I will return the 114 rank solution.
    # It is significantly better than 144.
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
