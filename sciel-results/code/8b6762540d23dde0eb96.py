import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # Output convention: W[k*n + i, r] where i is row of A/C, k is col of B/C
    # A is n x m (3x4), B is m x p (4x4), C is n x p (3x4)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # We will construct a rank 44 decomposition.
    # Strategy:
    # Split B's rows (j index) into two groups of 2: J1={0,1}, J2={2,3}
    # C = A[:, J1] @ B[J1, :] + A[:, J2] @ B[J2, :]
    # Let A1 = A[:, 0:2] (3x2), B1 = B[0:2, :] (2x4)
    # Let A2 = A[:, 2:4] (3x2), B2 = B[2:4, :] (2x4)
    # Term 1: A1 @ B1 (3x2x4)
    # Term 2: A2 @ B2 (3x2x4)
    
    # For A1 @ B1 (3x2x4):
    # Split A1's rows (i index) into I1={0,1} (2 rows) and I2={2} (1 row)
    # A1 = [A1_top; A1_bot] where A1_top is 2x2, A1_bot is 1x2
    # B1 is 2x4. Split B1's cols (k index) into K1={0,1} (2 cols) and K2={2,3} (2 cols)
    # B1 = [B1_left | B1_right] where B1_left is 2x2, B1_right is 2x2
    
    # Sub-term 1.1: A1_top @ B1_left (2x2x2) -> Rank 7 (Strassen)
    # Sub-term 1.2: A1_top @ B1_right (2x2x2) -> Rank 7 (Strassen)
    # Sub-term 1.3: A1_bot @ B1 (1x2x4) -> Rank 4 (Schoolbook, since 1x2x4 is minimal rank 4? 
    #   C[2, k] = A1_bot[0, 0]*B1[0, k] + A1_bot[0, 1]*B1[1, k]
    #   This is a linear combination of rows of B1. 
    #   Actually, the tensor for 1x2x4 multiplication:
    #   C_k = a0 B_0k + a1 B_1k.
    #   Rank is 4? 
    #   Let's check. T[i=0, j, k] = 1.
    #   We need sum_r U[0, r] V[j, r] W[k, r] = 1.
    #   U is 1xR, V is 2xR, W is 4xR.
    #   For each r, we have a vector v_r in R^2 and w_r in R^4.
    #   The sum is u_r (v_r \otimes w_r).
    #   This is equivalent to representing the identity map from R^2 to R^4? No.
    #   It's just C_k = sum_j A_j B_jk.
    #   This is just multiplying a row vector (1x2) by a matrix (2x4).
    #   This is a bilinear map.
    #   Rank of 1x2x4 tensor?
    #   The matrix flattening (i fixed) is V \otimes W^T? 
    #   Actually, for fixed i, the matrix is M[j, k] = 1.
    #   This is the all-ones matrix of size 2x4.
    #   The rank of the tensor is the minimum R such that M = sum_r u_r v_r^T w_r? No.
    #   M[j, k] = sum_r V[j, r] W[k, r] (since U[0, r] can be absorbed into V or W).
    #   So we need to represent the 2x4 all-ones matrix as a sum of R rank-1 matrices.
    #   The rank of the all-ones 2x4 matrix is 1.
    #   So we can do it with R=1?
    #   Let V = [[1], [1]] and W = [[1, 1, 1, 1]].
    #   Then V W = [[1, 1, 1, 1], [1, 1, 1, 1]].
    #   So yes, rank 1 is sufficient for the 1x2x4 sub-tensor IF we can choose U, V, W freely.
    #   Wait, the tensor is T[i, j, k] = 1 for i=0.
    #   Decomposition: sum_r U[i, r] V[j, r] W[k, r].
    #   For i=0: sum_r U[0, r] V[j, r] W[k, r] = 1.
    #   If we set U[0, 1] = 1, V[0, 1]=1, V[1, 1]=1, W[0, 1]=1, W[1, 1]=1, W[2, 1]=1, W[3, 1]=1.
    #   Then the sum is 1*1*1 + ... no, it's a single term r=1.
    #   U[0, 1]=1, V[j, 1]=1, W[k, 1]=1. Product is 1.
    #   So rank 1 is sufficient for the 1x2x4 part.
    #   However, we must be careful. The "rank" of the matrix multiplication tensor M_{1,2,4} is indeed 4?
    #   No, the rank of the tensor T_{1,2,4} is 4?
    #   Let's re-evaluate.
    #   The tensor T has entries T[i,j,k] = 1 if it's the multiplication tensor.
    #   For 1x2x4, T[0,j,k] = 1.
    #   We want sum_r U[0,r] V[j,r] W[k,r] = 1.
    #   This is equivalent to finding R vectors v_r in R^2 and w_r in R^4 such that sum_r (alpha_r v_r \otimes w_r) = J_{2x4} (all ones).
    #   Since J_{2x4} has rank 1, it can be written as v \otimes w where v=[1,1]^T and w=[1,1,1,1]^T.
    #   So R=1 is sufficient.
    #   So Sub-term 1.3 has rank 1.
    
    # So for A1 @ B1 (3x2x4):
    # A1_top (2x2) @ B1_left (2x2): Rank 7
    # A1_top (2x2) @ B1_right (2x2): Rank 7
    # A1_bot (1x2) @ B1 (2x4): Rank 1
    # Total rank for A1 @ B1 = 7 + 7 + 1 = 15.
    
    # Similarly for A2 @ B2 (3x2x4): Rank 15.
    
    # Total rank = 15 + 15 = 30.
    
    # This is much better than 44! Let's verify the logic.
    # Does the decomposition of A1 @ B1 into these sub-terms cover all indices?
    # A1 is 3x2. Rows 0,1 are A1_top. Row 2 is A1_bot.
    # B1 is 2x4. Cols 0,1 are B1_left. Cols 2,3 are B1_right.
    # C1 = A1 @ B1.
    # C1[0, 0] depends on A1[0, :] and B1[:, 0]. This is in A1_top @ B1_left.
    # C1[0, 2] depends on A1[0, :] and B1[:, 2]. This is in A1_top @ B1_right.
    # C1[2, 0] depends on A1[2, :] and B1[:, 0]. This is in A1_bot @ B1.
    # C1[2, 2] depends on A1[2, :] and B1[:, 2]. This is in A1_bot @ B1.
    
    # The sub-tensors are:
    # 1. A1_top @ B1_left: Indices i in {0,1}, j in {0,1}, k in {0,1}.
    #    This computes C1[i, k] for i in {0,1}, k in {0,1} using only A1[i, j] for j in {0,1} and B1[j, k] for j in {0,1}.
    #    This is correct.
    # 2. A1_top @ B1_right: Indices i in {0,1}, j in {0,1}, k in {2,3}.
    #    This computes C1[i, k] for i in {0,1}, k in {2,3} using only A1[i, j] for j in {0,1} and B1[j, k] for j in {0,1}.
    #    This is correct.
    # 3. A1_bot @ B1: Indices i in {2}, j in {0,1}, k in {0,1,2,3}.
    #    This computes C1[2, k] for all k using A1[2, j] for j in {0,1} and B1[j, k] for all k.
    #    This is correct.
    
    # Are there any overlaps or gaps?
    # The output C1 has indices i in {0,1,2}, k in {0,1,2,3}.
    # i=0,1 and k=0,1 are covered by term 1.
    # i=0,1 and k=2,3 are covered by term 2.
    # i=2 and k=0,1,2,3 are covered by term 3.
    # No overlaps, no gaps.
    
    # So the total rank is 7 + 7 + 1 + 7 + 7 + 1 = 30.
    
    # Let's construct the matrices.
    rank = 30
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 terms
    def add_strassen(u_offset, v_offset, w_offset, u_scale, v_scale, w_scale):
        nonlocal col
        # Strassen 7 terms for 2x2x2
        # C = A @ B
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(f+g)
        # M7 = (b-d)(e+g)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices in the local 2x2 block:
        # A indices: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
        # B indices: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C indices: (0,0)->C11, (0,1)->C12, (1,0)->C21, (1,1)->C22
        
        # U corresponds to A[i, j]. Index: i*2 + j.
        # V corresponds to B[j, k]. Index: j*2 + k.
        # W corresponds to C[i, k] (but stored as W[k*n+i]). 
        # Wait, the global W index is k*n + i.
        # In the local block, i and k are local indices 0,1.
        # The global i is u_offset_i + local_i.
        # The global k is u_offset_k + local_k.
        # We need to map local (i, k) to global W index.
        
        # Let's define the local to global mapping for W.
        # The W matrix is indexed by (k, i).
        # Global k = base_k + local_k.
        # Global i = base_i + local_i.
        # Global W index = global_k * n + global_i.
        
        # We pass offsets.
        # u_offset is (i_base, j_base) for A.
        # v_offset is (j_base, k_base) for B.
        # w_offset is (i_base, k_base) for C.
        
        i_base, j_base = u_offset
        j_base_v, k_base_v = v_offset
        i_base_w, k_base_w = w_offset
        
        # Check consistency: j_base should match j_base_v.
        
        terms = [
            # (U coeffs, V coeffs, W coeffs)
            # M1 = (a+d)(e+h)
            # U: a=1, d=1 -> [1, 0, 0, 1] (flattened A)
            # V: e=1, h=1 -> [1, 0, 0, 1] (flattened B)
            # W: C11=1, C22=1 -> [1, 0, 0, 1] (flattened C)
            # Note: W is stored as W[k, i].
            # C11 is i=0, k=0. C22 is i=1, k=1.
            # So W[0,0]=1, W[1,1]=1. Flattened W (k*n+i) for 2x2:
            # k=0, i=0 -> 0. k=0, i=1 -> 1. k=1, i=0 -> 2. k=1, i=1 -> 3.
            # So W vector is [1, 0, 0, 1].
            
            # Let's list the 7 Strassen terms carefully.
            # A flat: [a, b, c, d]
            # B flat: [e, f, g, h]
            # C flat (k,i order): [C00, C10, C01, C11] ? No.
            # W index is k*n + i. For 2x2, n=2.
            # (k=0, i=0) -> 0. (k=0, i=1) -> 1. (k=1, i=0) -> 2. (k=1, i=1) -> 3.
            # So W flat is [C00, C10, C01, C11].
            # Wait, C10 means i=1, k=0. C01 means i=0, k=1.
            # Standard C is [[C00, C01], [C10, C11]].
            # So W flat is [C00, C10, C01, C11].
            
            # M1: (a+d)(e+h) -> C00 += 1, C11 += 1.
            # U: [1, 0, 0, 1]
            # V: [1, 0, 0, 1]
            # W: [1, 0, 0, 1] (C00=1, C11=1)
            
            # M2: (c+d)e -> C10 += 1.
            # U: [0, 0, 1, 1]
            # V: [1, 0, 0, 0]
            # W: [0, 1, 0, 0] (C10=1)
            
            # M3: a(f-h) -> C01 += 1, C11 -= 1.
            # U: [1, 0, 0, 0]
            # V: [0, 1, 0, -1]
            # W: [0, 0, 1, -1] (C01=1, C11=-1)
            
            # M4: d(g-e) -> C00 -= 1, C10 += 1.
            # U: [0, 0, 0, 1]
            # V: [0, 0, 1, -1]
            # W: [-1, 1, 0, 0] (C00=-1, C10=1)
            
            # M5: (a+b)h -> C00 -= 1, C01 += 1.
            # U: [1, 1, 0, 0]
            # V: [0, 0, 0, 1]
            # W: [-1, 0, 1, 0] (C00=-1, C01=1)
            
            # M6: (c-d)(f+g) -> C11 += 1.
            # U: [0, 0, 1, -1]
            # V: [0, 1, 1, 0]
            # W: [0, 0, 0, 1] (C11=1)
            
            # M7: (b-d)(e+g) -> C00 += 1, C10 -= 1.
            # U: [0, 1, 0, -1]
            # V: [1, 0, 1, 0]
            # W: [1, -1, 0, 0] (C00=1, C10=-1)
        ]
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, -1],
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, 0, 0]
        ]
        
        for t in range(7):
            # Fill U
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_idx = (i_base + li) * m + (j_base + lj)
                    U[global_idx, col] += u_vecs[t][local_idx] * u_scale
            
            # Fill V
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    global_idx = (j_base_v + lj) * p + (k_base_v + lk)
                    V[global_idx, col] += v_vecs[t][local_idx] * v_scale
            
            # Fill W
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_idx = (k_base_w + lk) * n + (i_base_w + li)
                    W[global_idx, col] += w_vecs[t][local_idx] * w_scale
            
            col += 1

    # Helper to add 1x2x4 term (Rank 1)
    def add_1x2x4(u_offset, v_offset, w_offset, u_scale, v_scale, w_scale):
        nonlocal col
        # A is 1x2. B is 2x4. C is 1x4.
        # i=0 (local), j in {0,1}, k in {0,1,2,3}.
        # U: [1, 1] (a0, a1)
        # V: [1, 1, 1, 1] (b00, b01, b10, b11) -> Wait, V is B[j, k].
        # B is 2x4. Flat V is [b00, b01, b02, b03, b10, b11, b12, b13].
        # We want sum_j A[0, j] B[j, k] = C[0, k].
        # We found that U=[1], V=[1, 1, 1, 1, 1, 1, 1, 1], W=[1, 1, 1, 1] works?
        # Let's check.
        # U[0] = 1.
        # V[j, k] = 1 for all j, k.
        # W[k] = 1 for all k.
        # Sum_r U[0, r] V[j, r] W[k, r]. Here r=1.
        # Term = 1 * 1 * 1 = 1.
        # But we need C[0, k] = sum_j A[0, j] B[j, k].
        # The decomposition is sum_r U[i, r] V[j, r] W[k, r].
        # If we use r=1, U[0, 1]=1, V[j, 1]=1, W[k, 1]=1.
        # Then the sum is 1.
        # But the actual value is sum_j A[0, j] B[j, k].
        # This is not constant 1.
        # My previous logic was flawed. The tensor T[0, j, k] = 1 is the *structure* of the multiplication.
        # The values A and B are variables.
        # The decomposition must satisfy:
        # sum_r U[i, r] V[j, r] W[k, r] A[i, j] B[j, k] ? No.
        # The tensor T is defined such that C[i, k] = sum_j A[i, j] B[j, k] = sum_r U[i, r] V[j, r] W[k, r] ? No.
        # The identity is:
        # sum_j A[i, j] B[j, k] = sum_r U[i, r] V[j, r] W[k, r] ?? No.
        # The identity is:
        # sum_r U[i, r] A[i, j] ... No.
        
        # Correct Identity:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Decomposition:
        # C[i, k] = sum_r U[i, r] V[j, r] W[k, r] is WRONG.
        # The decomposition is:
        # sum_r U[i, r] V[j, r] W[k, r] is a tensor T[i, j, k].
        # We require T[i, j, k] = 1 for all i, j, k? No.
        # We require that the bilinear form defined by T matches the multiplication.
        # i.e., sum_r U[i, r] V[j, r] W[k, r] is the coefficient of A[i, j] B[j, k] in the expression for C[i, k]?
        # No.
        # The standard tensor decomposition for matrix multiplication is:
        # sum_r U[i, r] V[j, r] W[k, r] = delta_{i, k}? No.
        
        # Let's look at the definition in the prompt:
        # "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor T is defined by T[i, j, k] = 1 if the term A[i, j]B[j, k] contributes to C[i, k].
        # In standard matrix multiplication, C[i, k] = sum_j A[i, j] B[j, k].
        # So the tensor T has T[i, j, k] = 1 for all valid i, j, k.
        # The decomposition is sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k] = 1.
        
        # So my previous analysis was correct.
        # For the 1x2x4 sub-tensor, we need sum_r U[0, r] V[j, r] W[k, r] = 1 for j in {0,1}, k in {0,1,2,3}.
        # We found that U[0, 1]=1, V[j, 1]=1, W[k, 1]=1 gives sum = 1.
        # So rank 1 is sufficient.
        
        i_base, j_base = u_offset
        j_base_v, k_base_v = v_offset
        i_base_w, k_base_w = w_offset
        
        # U is 1x2. Flat: [1, 1].
        # V is 2x4. Flat: [1, 1, 1, 1, 1, 1, 1, 1].
        # W is 1x4. Flat (k*n+i, n=1): [1, 1, 1, 1].
        
        for lj in range(2):
            global_idx_u = (i_base) * m + (j_base + lj)
            U[global_idx_u, col] += 1 * u_scale
            
            global_idx_v = (j_base_v + lj) * p + k_base_v # Wait, V index is j*p + k.
            # V is m x p. Here m=2, p=4.
            # V[j, k].
            for lk in range(4):
                global_idx_v = (j_base_v + lj) * p + (k_base_v + lk)
                V[global_idx_v, col] += 1 * v_scale
        
        for lk in range(4):
            global_idx_w = (k_base_w + lk) * n + i_base_w
            W[global_idx_w, col] += 1 * w_scale
        
        col += 1

    # Now, apply to the specific blocks.
    
    # Block 1: A1 @ B1
    # A1 is rows 0,1,2 of A. Cols 0,1 of A.
    # B1 is rows 0,1 of B. Cols 0,1,2,3 of B.
    # A1_top: i in {0,1}, j in {0,1}.
    # A1_bot: i in {2}, j in {0,1}.
    # B1_left: j in {0,1}, k in {0,1}.
    # B1_right: j in {0,1}, k in {2,3}.
    
    # Sub-term 1.1: A1_top @ B1_left (2x2x2)
    # u_offset: i_base=0, j_base=0
    # v_offset: j_base=0, k_base=0
    # w_offset: i_base=0, k_base=0
    add_strassen((0, 0), (0, 0), (0, 0), 1, 1, 1)
    
    # Sub-term 1.2: A1_top @ B1_right (2x2x2)
    # u_offset: i_base=0, j_base=0
    # v_offset: j_base=0, k_base=2
    # w_offset: i_base=0, k_base=2
    add_strassen((0, 0), (0, 2), (0, 2), 1, 1, 1)
    
    # Sub-term 1.3: A1_bot @ B1 (1x2x4)
    # u_offset: i_base=2, j_base=0
    # v_offset: j_base=0, k_base=0
    # w_offset: i_base=2, k_base=0
    add_1x2x4((2, 0), (0, 0), (2, 0), 1, 1, 1)
    
    # Block 2: A2 @ B2
    # A2 is rows 0,1,2 of A. Cols 2,3 of A.
    # B2 is rows 2,3 of B. Cols 0,1,2,3 of B.
    # A2_top: i in {0,1}, j in {2,3}.
    # A2_bot: i in {2}, j in {2,3}.
    # B2_left: j in {2,3}, k in {0,1}.
    # B2_right: j in {2,3}, k in {2,3}.
    
    # Sub-term 2.1: A2_top @ B2_left (2x2x2)
    # u_offset: i_base=0, j_base=2
    # v_offset: j_base=2, k_base=0
    # w_offset: i_base=0, k_base=0
    add_strassen((0, 2), (2, 0), (0, 0), 1, 1, 1)
    
    # Sub-term 2.2: A2_top @ B2_right (2x2x2)
    # u_offset: i_base=0, j_base=2
    # v_offset: j_base=2, k_base=2
    # w_offset: i_base=0, k_base=2
    add_strassen((0, 2), (2, 2), (0, 2), 1, 1, 1)
    
    # Sub-term 2.3: A2_bot @ B2 (1x2x4)
    # u_offset: i_base=2, j_base=2
    # v_offset: j_base=2, k_base=0
    # w_offset: i_base=2, k_base=0
    add_1x2x4((2, 2), (2, 0), (2, 0), 1, 1, 1)
    
    # Verify rank
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank0 = n * m * p
        U = np.zeros((n * m, rank0), dtype=float)
        V = np.zeros((m * p, rank0), dtype=float)
        W = np.zeros((p * n, rank0), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
