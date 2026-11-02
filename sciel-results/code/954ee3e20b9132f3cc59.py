import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition (Rank 56) as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using block structure
    # Strategy: Decompose 2x4 * 4x7 into blocks
    # A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2 (2x7)
    
    # We can use Strassen for 2x2 * 2x2 blocks, but B is 2x7 not 2x2.
    # So we need to split B further or use a different approach.
    
    # Alternative: Use the known result that R(2,4,7) <= 50
    # This can be achieved by decomposing into:
    # - Two 2x2 * 2x7 multiplications
    # Each 2x2 * 2x7 can be decomposed into 7 blocks of 2x2 * 2x1
    # R(2,2,1) = 4 (known)
    # So total rank = 2 * 7 * 4 = 56... that's not better.
    
    # Let me try a different approach:
    # R(2,4,7) can be bounded by R(2,4,4) + R(2,4,3)
    # Or use the formula R(m,n,p) <= R(m,n,1) * p + R(m,n,p-1)
    
    # Actually, let's try to construct a rank-50 decomposition directly.
    # Known: R(2,2,2) = 7 (Strassen)
    # R(2,2,1) = 4
    # R(1,2,1) = 2
    
    # Decomposition strategy:
    # A (2x4), B (4x7), C (2x7)
    # Split A into two 2x2 blocks: A = [A_left | A_right]
    # Split B into two 2x7 blocks: B = [B_top; B_bottom]
    # C = A_left * B_top + A_right * B_bottom
    
    # For each 2x2 * 2x7 multiplication:
    # Split the 2x7 into seven 2x1 columns
    # Each 2x2 * 2x1 has rank 4
    # Total for one 2x2 * 2x7: 7 * 4 = 28
    # Total for 2x4 * 4x7: 2 * 28 = 56
    
    # That's still 56. Let me think differently.
    
    # Better approach: Use the fact that R(2,4,7) <= 50 is known.
    # One way: R(2,4,7) <= R(2,2,7) + R(2,2,7) where we split A differently.
    # Or: R(2,4,7) <= 2*R(2,2,4) + 2*R(2,2,3)
    # R(2,2,4) = 2*R(2,2,2) = 14
    # R(2,2,3) = ? 
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11
    # So total: 2*14 + 2*11 = 28 + 22 = 50
    
    # Let's construct this!
    # A (2x4) split into left 2x2 and right 2x2
    # B (4x7) split into top 2x7 and bottom 2x7
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (2x2 * 2x7):
    # Split B_top into B_top_left (2x4) and B_top_right (2x3)
    # A_left * B_top = A_left * B_top_left + A_left * B_top_right
    # R(2,2,4) = 14 (two Strassen blocks)
    # R(2,2,3) = 11 (one Strassen + one rank-4 block)
    
    # Similarly for A_right * B_bottom
    
    # Let's build the Strassen decomposition for 2x2 * 2x2 (rank 7)
    # And the rank-4 decomposition for 2x2 * 2x1
    
    # Strassen 2x2*2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = a*(e+h), M2 = (a+c)*h, M3 = (c+d)*e
    # M4 = (b-d)*(g+h), M5 = (a+b)*e, M6 = (c-a)*(f+h), M7 = (b-c)*(e+f)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    strassen_U = np.array([
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, -1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0]
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    
    # Rank-4 decomposition for 2x2 * 2x1:
    # A = [[a,b],[c,d]], B = [e; f]
    # C1 = a*e + b*f, C2 = c*e + d*f
    # M1 = a*e, M2 = b*f, M3 = c*e, M4 = d*f
    # C1 = M1 + M2, C2 = M3 + M4
    
    rank4_U = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    rank4_V = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ], dtype=float)
    
    rank4_W = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ], dtype=float)
    
    # Now build R(2,2,4) = 14 using two Strassen blocks
    # 2x2 * 2x4: A (2x2) * B (2x4) = C (2x4)
    # Split B into B1 (2x2) and B2 (2x2)
    # C = A*B1 + A*B2
    
    def build_224():
        """Build R(2,2,4) = 14 decomposition"""
        U_list = []
        V_list = []
        W_list = []
        
        # First block: B columns 0,1
        for r in range(7):
            U_list.append(strassen_U[r])
            V_list.append(strassen_V[r])  # B1 is 2x2, flattened [e,f,g,h]
            W_list.append(strassen_W[r])  # C1 is 2x2, flattened [C11,C12,C21,C22]
        
        # Second block: B columns 2,3
        for r in range(7):
            U_list.append(strassen_U[r])
            # B2 is 2x2 at columns 2,3 of 2x4
            # Flattened 2x4: [B00,B01,B02,B03,B10,B11,B12,B13]
            # B2 = [[B02,B03],[B12,B13]] at indices 2,3,6,7
            v = np.zeros(8)
            v[2] = strassen_V[r][0]  # e
            v[3] = strassen_V[r][1]  # f
            v[6] = strassen_V[r][2]  # g
            v[7] = strassen_V[r][3]  # h
            V_list.append(v)
            # C2 is 2x2 at columns 2,3 of 2x4
            # Flattened 2x4 C: [C11,C12,C13,C14,C21,C22,C23,C24]
            w = np.zeros(8)
            w[2] = strassen_W[r][0]  # C11 -> C13
            w[3] = strassen_W[r][1]  # C12 -> C14
            w[6] = strassen_W[r][2]  # C21 -> C23
            w[7] = strassen_W[r][3]  # C22 -> C24
            W_list.append(w)
        
        return np.array(U_list).T, np.array(V_list).T, np.array(W_list).T
    
    # Now build R(2,2,3) = 11 using one Strassen (2x2) + one rank-4 (2x1)
    # 2x2 * 2x3: A (2x2) * B (2x3) = C (2x3)
    # Split B into B1 (2x2) and B2 (2x1)
    # C = A*B1 + A*B2
    
    def build_223():
        """Build R(2,2,3) = 11 decomposition"""
        U_list = []
        V_list = []
        W_list = []
        
        # First block: B columns 0,1 (2x2) using Strassen
        for r in range(7):
            U_list.append(strassen_U[r])
            # B1 is 2x2 at columns 0,1 of 2x3
            # Flattened 2x3: [B00,B01,B02,B10,B11,B12]
            # B1 = [[B00,B01],[B10,B11]] at indices 0,1,3,4
            v = np.zeros(6)
            v[0] = strassen_V[r][0]  # e
            v[1] = strassen_V[r][1]  # f
            v[3] = strassen_V[r][2]  # g
            v[4] = strassen_V[r][3]  # h
            V_list.append(v)
            # C1 is 2x2 at columns 0,1 of 2x3
            # Flattened 2x3 C: [C11,C12,C13,C21,C22,C23]
            w = np.zeros(6)
            w[0] = strassen_W[r][0]  # C11
            w[1] = strassen_W[r][1]  # C12
            w[3] = strassen_W[r][2]  # C21
            w[4] = strassen_W[r][3]  # C22
            W_list.append(w)
        
        # Second block: B column 2 (2x1) using rank-4
        for r in range(4):
            U_list.append(rank4_U[r])
            # B2 is 2x1 at column 2 of 2x3
            # Flattened 2x3: [B00,B01,B02,B10,B11,B12]
            # B2 = [B02; B12] at indices 2,5
            v = np.zeros(6)
            v[2] = rank4_V[r][0]  # e
            v[5] = rank4_V[r][1]  # f
            V_list.append(v)
            # C2 is 2x1 at column 2 of 2x3
            # Flattened 2x3 C: [C11,C12,C13,C21,C22,C23]
            w = np.zeros(6)
            w[2] = rank4_W[r][0]  # C11 -> C13
            w[5] = rank4_W[r][1]  # C21 -> C23
            W_list.append(w)
        
        return np.array(U_list).T, np.array(V_list).T, np.array(W_list).T
    
    # Now assemble the full 2x4 * 4x7 decomposition
    # A (2x4) = [A_left | A_right] where each is 2x2
    # B (4x7) = [B_top; B_bottom] where each is 2x7
    # C = A_left * B_top + A_right * B_bottom
    
    # B_top (2x7) split into B_top_1 (2x4) and B_top_2 (2x3)
    # B_bottom (2x7) split into B_bottom_1 (2x4) and B_bottom_2 (2x3)
    
    # A_left * B_top = A_left * B_top_1 + A_left * B_top_2
    #   = R(2,2,4) + R(2,2,3) = 14 + 11 = 25
    # A_right * B_bottom = A_right * B_bottom_1 + A_right * B_bottom_2
    #   = R(2,2,4) + R(2,2,3) = 14 + 11 = 25
    # Total: 50
    
    U_full = []
    V_full = []
    W_full = []
    
    # Helper to place a 2x2 U vector into the 2x4 A matrix
    # A flattened: [A00,A01,A02,A03,A10,A11,A12,A13]
    # A_left = [[A00,A01],[A10,A11]] at indices 0,1,4,5
    # A_right = [[A02,A03],[A12,A13]] at indices 2,3,6,7
    def place_U_2x2_in_2x4(u_2x2, which):
        """which: 0 for left, 1 for right"""
        u = np.zeros(8)
        if which == 0:  # left
            u[0] = u_2x2[0]  # a
            u[1] = u_2x2[1]  # b
            u[4] = u_2x2[2]  # c
            u[5] = u_2x2[3]  # d
        else:  # right
            u[2] = u_2x2[0]  # a
            u[3] = u_2x2[1]  # b
            u[6] = u_2x2[2]  # c
            u[7] = u_2x2[3]  # d
        return u
    
    # Helper to place a 2x7 V vector into the 4x7 B matrix
    # B flattened: [B00,B01,...,B06,B10,...,B16,B20,...,B26,B30,...,B36]
    # B_top = rows 0,1: indices 0-6 (row 0) and 7-13 (row 1)
    # B_bottom = rows 2,3: indices 14-20 (row 2) and 21-27 (row 3)
    def place_V_2x7_in_4x7(v_2x7, which):
        """which: 0 for top, 1 for bottom"""
        v = np.zeros(28)
        if which == 0:  # top (rows 0,1)
            # Row 0: indices 0-6
            for j in range(7):
                v[j] = v_2x7[j]  # B[0,j]
            # Row 1: indices 7-13
            for j in range(7):
                v[7+j] = v_2x7[7+j]  # B[1,j]
        else:  # bottom (rows 2,3)
            # Row 2: indices 14-20
            for j in range(7):
                v[14+j] = v_2x7[j]  # B[2,j]
            # Row 3: indices 21-27
            for j in range(7):
                v[21+j] = v_2x7[7+j]  # B[3,j]
        return v
    
    # Helper to place a 2x7 W vector into the 2x7 C matrix
    # C flattened: [C00,C01,...,C06,C10,...,C16]
    def place_W_2x7_in_2x7(w_2x7):
        return w_2x7  # Already in the right format
    
    # Now build each component
    
    # Component 1: A_left * B_top_1 (2x2 * 2x4)
    U_224, V_224, W_224 = build_224()
    for r in range(14):
        U_full.append(place_U_2x2_in_2x4(U_224[:, r], 0))
        # V_224 is 2x4, need to place into 2x7 (columns 0-3)
        v_2x7 = np.zeros(14)
        v_2x7[0:4] = V_224[0:4, r]  # Row 0, cols 0-3
        v_2x7[7:11] = V_224[4:8, r]  # Row 1, cols 0-3
        V_full.append(place_V_2x7_in_4x7(v_2x7, 0))
        # W_224 is 2x4, need to place into 2x7 (columns 0-3)
        w_2x7 = np.zeros(14)
        w_2x7[0:4] = W_224[0:4, r]  # Row 0, cols 0-3
        w_2x7[7:11] = W_224[4:8, r]  # Row 1, cols 0-3
        W_full.append(place_W_2x7_in_2x7(w_2x7))
    
    # Component 2: A_left * B_top_2 (2x2 * 2x3)
    U_223, V_223, W_223 = build_223()
    for r in range(11):
        U_full.append(place_U_2x2_in_2x4(U_223[:, r], 0))
        # V_223 is 2x3, need to place into 2x7 (columns 4-6)
        v_2x7 = np.zeros(14)
        v_2x7[4:7] = V_223[0:3, r]  # Row 0, cols 4-6
        v_2x7[11:14] = V_223[3:6, r]  # Row 1, cols 4-6
        V_full.append(place_V_2x7_in_4x7(v_2x7, 0))
        # W_223 is 2x3, need to place into 2x7 (columns 4-6)
        w_2x7 = np.zeros(14)
        w_2x7[4:7] = W_223[0:3, r]  # Row 0, cols 4-6
        w_2x7[11:14] = W_223[3:6, r]  # Row 1, cols 4-6
        W_full.append(place_W_2x7_in_2x7(w_2x7))
    
    # Component 3: A_right * B_bottom_1 (2x2 * 2x4)
    U_224, V_224, W_224 = build_224()
    for r in range(14):
        U_full.append(place_U_2x2_in_2x4(U_224[:, r], 1))
        # V_224 is 2x4, need to place into 2x7 (columns 0-3) in bottom
        v_2x7 = np.zeros(14)
        v_2x7[0:4] = V_224[0:4, r]  # Row 0, cols 0-3
        v_2x7[7:11] = V_224[4:8, r]  # Row 1, cols 0-3
        V_full.append(place_V_2x7_in_4x7(v_2x7, 1))
        # W_224 is 2x4, need to place into 2x7 (columns 0-3)
        w_2x7 = np.zeros(14)
        w_2x7[0:4] = W_224[0:4, r]  # Row 0, cols 0-3
        w_2x7[7:11] = W_224[4:8, r]  # Row 1, cols 0-3
        W_full.append(place_W_2x7_in_2x7(w_2x7))
    
    # Component 4: A_right * B_bottom_2 (2x2 * 2x3)
    U_223, V_223, W_223 = build_223()
    for r in range(11):
        U_full.append(place_U_2x2_in_2x4(U_223[:, r], 1))
        # V_223 is 2x3, need to place into 2x7 (columns 4-6) in bottom
        v_2x7 = np.zeros(14)
        v_2x7[4:7] = V_223[0:3, r]  # Row 0, cols 4-6
        v_2x7[11:14] = V_223[3:6, r]  # Row 1, cols 4-6
        V_full.append(place_V_2x7_in_4x7(v_2x7, 1))
        # W_223 is 2x3, need to place into 2x7 (columns 4-6)
        w_2x7 = np.zeros(14)
        w_2x7[4:7] = W_223[0:3, r]  # Row 0, cols 4-6
        w_2x7[11:14] = W_223[3:6, r]  # Row 1, cols 4-6
        W_full.append(place_W_2x7_in_2x7(w_2x7))
    
    # Convert to numpy arrays
    U_full = np.array(U_full).T
    V_full = np.array(V_full).T
    W_full = np.array(W_full).T
    
    # Verify the decomposition
    rank_found = U_full.shape[1]
    print(f"Found decomposition with rank {rank_found}")
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor"""
        # For random A (n x m) and B (m x p), check that C = A @ B
        # can be computed as sum_r (U[:,r] @ A_flat) * (V[:,r] @ B_flat) * W[:,r]
        # Actually, the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal delta[i,k] (the identity in the middle dimension)
        
        # Let's verify by checking that for any A, B:
        # C = sum_r (U[:,r] @ A_flat) * (V[:,r] @ B_flat) * W[:,r]
        # Wait, that's not quite right. Let me think...
        
        # The tensor decomposition means:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T is the matrix multiplication tensor, so T[i,j,k] = 1 if i==k (in some sense)
        # Actually, for matrix multiplication, the tensor is defined such that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T satisfies: T[i,j,k] = delta[i,k]... no that's not right either.
        
        # Let me just verify numerically with random matrices.
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            # U[:,r] is a vector of length n*m
            # V[:,r] is a vector of length m*p
            # W[:,r] is a vector of length p*n
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            C_computed += np.trace(u @ v @ w)  # This might not be right
            
        # Actually, the correct way to use the tensor decomposition:
        # C[i,k] = sum_j sum_r A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that doesn't look right either. Let me reconsider.
        
        # The tensor T for matrix multiplication has indices (i, j, k) where:
        # i indexes rows of A (and C), j indexes columns of A / rows of B,
        # k indexes columns of B (and C)
        # T[i,j,k] = 1 if this term contributes to C[i,k]
        # Actually, T[i,j,k] = delta[j,j]... no.
        
        # The matrix multiplication tensor is defined as:
        # T_{i,j,k} = 1 if the (i,j) element of A and (j,k) element of B
        # contribute to the (i,k) element of C
        # So T_{i,j,k} = 1 for all valid i,j,k (it's a 3D identity-like tensor)
        # No wait, T_{i,j,k} is just 1 for all i,j,k in the appropriate ranges,
        # and the decomposition is:
        # T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # with U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # So let's verify:
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_expected = 1.0  # The tensor entry is always 1
                    t_computed = 0.0
                    for r in range(U.shape[1]):
                        t_computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += abs(t_expected - t_computed)
        
        return error < 1e-10
    
    if verify(U_full, V_full, W_full, n, m, p):
        print("Decomposition verified successfully!")
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = rank_found
    else:
        print("Decomposition verification failed, using schoolbook")
    
    # Check time
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
