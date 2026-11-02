"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [2, 4, 5]
TIME_LIMIT = 120.0
DEADLINE_MARGIN = 10.0

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Generate Schoolbook Decomposition (Rank 40)
    def schoolbook(n, m, p):
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
        return U, V, W

    best_U, best_V, best_W = schoolbook(n, m, p)
    best_rank = n * m * p

    def verify(U, V, W):
        # Check if U, V, W is a valid decomposition
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
        # This is equivalent to checking if the tensor formed by U,V,W matches the multiplication tensor.
        # We can check this by flattening.
        # Tensor T is of shape (n*m, m*p, p*n)
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We need T to match the multiplication tensor.
        # Multiplication tensor M[a, b, c] is 1 if a, b, c correspond to the same (i, j, k) in a valid multiplication term.
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        # So M[a, b, c] = 1 iff (a % m == b // p) and (b % p == c // n) and (a // m == c % n)
        
        # Constructing the full tensor might be expensive if rank is high, but here n,m,p are small.
        # Let's construct the target tensor T_target.
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        # Construct T from U, V, W
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a contraction.
        # We can do this efficiently using einsum or loops.
        # Since rank is small (< 50), loops are fine.
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(len(U[0])):
            u_r = U_arr[:, r]
            v_r = V_arr[:, r]
            w_r = W_arr[:, r]
            # Outer product u_r v_r w_r
            # T += np.einsum('a,b,c->abc', u_r, v_r, w_r)
            T_calc += np.outer(u_r, v_r) * w_r[:, None, None]
            
        # Compare
        # Use allclose with tight tolerance
        return np.allclose(T_calc, T_target, atol=1e-9)

    # 2. Try to reduce rank using random restarts and optimization
    # Strategy: 
    # a) Start with a valid decomposition of rank R.
    # b) Try to find a decomposition of rank R-1.
    # c) We can use a heuristic: 
    #    - Pick a random subset of columns of size R-1.
    #    - Check if the tensor spanned by these columns contains T_target.
    #    - This is equivalent to checking if T_target is in the span of {U[:, i] v[:, i] w[:, i]} for i in subset.
    #    - This is a subspace membership problem.
    
    # Better Strategy: 
    # Use the fact that we can permute and combine columns.
    # Let's try a simpler approach: 
    # We know rank 36 is possible (from knowledge).
    # Let's try to construct it explicitly using block decomposition.
    # 2x4x5 can be split into 2x2x5 + 2x2x5.
    # Rank(2x2x5) <= 2 * Rank(2x2x2) + 2 * Rank(2x2x1)? No.
    # Rank(2x2xp) is known. 
    # For p=2, rank is 7.
    # For p=1, rank is 4.
    # For p=5, we can split 5 into 2+2+1.
    # So Rank(2x2x5) <= Rank(2x2x2) + Rank(2x2x2) + Rank(2x2x1) = 7 + 7 + 4 = 18.
    # Then Rank(2x4x5) <= 2 * 18 = 36.
    
    # Let's implement this constructive approach.
    
    def strassen_222():
        # Rank 7 decomposition for 2x2x2
        # Indices: i,j in 0..1 for A (2x2), j,k in 0..1 for B (2x2), k,i in 0..1 for C (2x2)
        # A indices: a = i*2 + j
        # B indices: b = j*2 + k
        # C indices: c = k*2 + i
        U = [
            [1, 0, 0, 0, 0, 0, 0], # a=00
            [0, 1, 0, 0, 0, 0, 0], # a=01
            [0, 0, 1, 0, 0, 0, 0], # a=10
            [0, 0, 0, 1, 1, 0, 0]  # a=11
        ]
        V = [
            [1, 0, 0, 0, 0, 0, 0], # b=00
            [0, 1, 0, 0, 0, 0, 0], # b=01
            [0, 0, 1, 0, 0, 0, 0], # b=10
            [0, 0, 0, 1, -1, 0, 0] # b=11
        ]
        W = [
            [1, 0, 0, 0, 0, 0, 0], # c=00
            [0, 1, 0, 0, 0, 0, 0], # c=01
            [0, 0, 1, 0, 0, 0, 0], # c=10
            [0, 0, 0, 0, 1, 0, 0]  # c=11
        ]
        # Wait, standard Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's verify the mapping to U, V, W.
        # Term r contributes U_a * V_b * W_c
        # We need U, V, W such that sum_r U_a V_b W_c = T[a,b,c]
        
        # Let's define the 7 terms explicitly.
        # Term 1: M1
        # A part: A00, A11. So U[00,1]=1, U[11,1]=1. Others 0.
        # B part: B00, B11. So V[00,1]=1, V[11,1]=1. Others 0.
        # C part: C00 (+1), C11 (+1). So W[00,1]=1, W[11,1]=1.
        
        # Term 2: M2
        # A part: A10, A11. U[10,2]=1, U[11,2]=1.
        # B part: B00. V[00,2]=1.
        # C part: C10 (+1). W[10,2]=1.
        
        # Term 3: M3
        # A part: A00. U[00,3]=1.
        # B part: B01 (-1), B11 (-1). V[01,3]=-1, V[11,3]=-1.
        # C part: C01 (+1). W[01,3]=1.
        
        # Term 4: M4
        # A part: A11. U[11,4]=1.
        # B part: B10 (+1), B00 (-1). V[10,4]=1, V[00,4]=-1.
        # C part: C00 (-1), C10 (+1). W[00,4]=-1, W[10,4]=1.
        
        # Term 5: M5
        # A part: A00, A01. U[00,5]=1, U[01,5]=1.
        # B part: B11. V[11,5]=1.
        # C part: C00 (-1), C01 (+1). W[00,5]=-1, W[01,5]=1.
        
        # Term 6: M6
        # A part: A10 (+1), A00 (-1). U[10,6]=1, U[00,6]=-1.
        # B part: B00, B01. V[00,6]=1, V[01,6]=1.
        # C part: C10 (+1). W[10,6]=1.
        
        # Term 7: M7
        # A part: A01 (+1), A11 (-1). U[01,7]=1, U[11,7]=-1.
        # B part: B10, B11. V[10,7]=1, V[11,7]=1.
        # C part: C11 (+1). W[11,7]=1.
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # Fill Term 1
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1
        
        # Fill Term 2
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1
        
        # Fill Term 3
        U_s[0][2] = 1
        V_s[1][2] = -1; V_s[3][2] = -1
        W_s[1][2] = 1
        
        # Fill Term 4
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = -1; W_s[2][3] = 1
        
        # Fill Term 5
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1
        
        # Fill Term 6
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[2][5] = 1
        
        # Fill Term 7
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[3][6] = 1
        
        return U_s, V_s, W_s

    def schoolbook_221():
        # Rank 4 for 2x2x1
        # C is 2x1.
        # a in 0..3 (i,j), b in 0..1 (j,k=0), c in 0..1 (k=0,i)
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        for i in range(2):
            for j in range(2):
                a = i*2 + j
                b = j*1 + 0
                c = 0*2 + i
                r = i*2 + j
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
        return U, V, W

    def combine_decompositions(list_of_decs):
        # list_of_decs is a list of (U, V, W)
        # We need to sum them up.
        # This works if the decompositions are for disjoint blocks of the matrix multiplication.
        # For 2x4x5, we split m=4 into m1=2, m2=2.
        # A is 2x4, B is 4x5.
        # A1 is top 2x2, A2 is bottom 2x2.
        # B1 is left 2x5, B2 is right 2x5.
        # C = A1*B1 + A2*B2.
        
        # The indices for U, V, W need to be mapped to the global indices.
        # Global A index: i in 0..1, j in 0..3. a = i*4 + j.
        # Global B index: j in 0..3, k in 0..4. b = j*5 + k.
        # Global C index: k in 0..4, i in 0..1. c = k*2 + i.
        
        # For term 1 (i in 0..1, j in 0..1):
        # Local A index: i, j_local (0..1).
        # Global A index: i*4 + j_local.
        # Local B index: j_local, k (0..4).
        # Global B index: j_local*5 + k.
        # Local C index: k, i.
        # Global C index: k*2 + i.
        
        # For term 2 (i in 0..1, j in 2..3):
        # Local A index: i, j_local (0..1).
        # Global A index: i*4 + (j_local + 2).
        # Local B index: j_local, k.
        # Global B index: (j_local + 2)*5 + k.
        # Local C index: k, i.
        # Global C index: k*2 + i.
        
        # We will construct U, V, W for each part and then combine.
        
        # Part 1: 2x2x5 (j=0,1)
        # We need a decomposition for 2x2x5.
        # Split k=5 into k=2,2,1.
        # Part 1.1: 2x2x2 (k=0,1) -> Rank 7
        # Part 1.2: 2x2x2 (k=2,3) -> Rank 7
        # Part 1.3: 2x2x1 (k=4) -> Rank 4
        # Total Rank for 2x2x5 = 18.
        
        # Let's implement a helper to place a local decomposition into global coordinates.
        
        def place_decomposition(local_U, local_V, local_W, i_offset, j_offset, k_offset, global_n, global_m, global_p):
            # local_U is for n x m_local
            # local_V is for m_local x p_local
            # local_W is for p_local x n
            # i_offset: offset for i in A and C (0 for both parts since n=2)
            # j_offset: offset for j in A and B
            # k_offset: offset for k in B and C
            
            n = len(local_W[0]) if local_W else 0 # n is actually len(local_U) // m_local
            # Better to pass n, m_local, p_local
            m_local = len(local_V) // (global_p - k_offset) # This is getting messy.
            
        # Let's just hardcode the 2x4x5 construction using the 2x2x5 building block.
        
        def build_225():
            # Returns U, V, W for 2x2x5 with local indices.
            # n=2, m=2, p=5.
            U = [[0]*18 for _ in range(4)]
            V = [[0]*18 for _ in range(10)]
            W = [[0]*18 for _ in range(10)]
            
            col = 0
            # Block 1: k=0,1 (2x2x2)
            U1, V1, W1 = strassen_222()
            for i in range(4):
                for r in range(7):
                    U[i][col + r] = U1[i][r]
            for j in range(2):
                for k in range(2):
                    b_local = j*2 + k
                    for r in range(7):
                        V[b_local][col + r] = V1[j*2+k][r] # Wait, V1 indices are j*p+k. Here p=2.
            for k in range(2):
                for i in range(2):
                    c_local = k*2 + i
                    for r in range(7):
                        W[c_local][col + r] = W1[k*2+i][r]
            col += 7
            
            # Block 2: k=2,3 (2x2x2)
            U2, V2, W2 = strassen_222()
            for i in range(4):
                for r in range(7):
                    U[i][col + r] = U2[i][r]
            for j in range(2):
                for k in range(2):
                    b_local = j*2 + (k+2)
                    for r in range(7):
                        V[b_local][col + r] = V2[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    c_local = (k+2)*2 + i
                    for r in range(7):
                        W[c_local][col + r] = W2[k*2+i][r]
            col += 7
            
            # Block 3: k=4 (2x2x1)
            U3, V3, W3 = schoolbook_221()
            for i in range(4):
                for r in range(4):
                    U[i][col + r] = U3[i][r]
            for j in range(2):
                b_local = j*1 + 4
                for r in range(4):
                    V[b_local][col + r] = V3[j][r] # V3 indices are j*1+k, k=0.
            for i in range(2):
                c_local = 4*2 + i
                for r in range(4):
                    W[c_local][col + r] = W3[i][r] # W3 indices are k*2+i, k=0.
            
            return U, V, W

        U_225, V_225, W_225 = build_225()
        
        # Now construct 2x4x5
        # Part 1: j=0,1
        U_global = [[0]*36 for _ in range(8)]
        V_global = [[0]*36 for _ in range(20)]
        W_global = [[0]*36 for _ in range(10)]
        
        col = 0
        # Map Part 1 (j=0,1)
        for i in range(2):
            for j in range(2):
                a_local = i*2 + j
                a_global = i*4 + j
                for r in range(18):
                    U_global[a_global][col + r] = U_225[a_local][r]
        for j in range(2):
            for k in range(5):
                b_local = j*5 + k
                b_global = j*5 + k
                for r in range(18):
                    V_global[b_global][col + r] = V_225[b_local][r]
        for k in range(5):
            for i in range(2):
                c_local = k*2 + i
                c_global = k*2 + i
                for r in range(18):
                    W_global[c_global][col + r] = W_225[c_local][r]
        col += 18
        
        # Map Part 2 (j=2,3)
        for i in range(2):
            for j in range(2):
                a_local = i*2 + j
                a_global = i*4 + (j+2)
                for r in range(18):
                    U_global[a_global][col + r] = U_225[a_local][r]
        for j in range(2):
            for k in range(5):
                b_local = j*5 + k
                b_global = (j+2)*5 + k
                for r in range(18):
                    V_global[b_global][col + r] = V_225[b_local][r]
        for k in range(5):
            for i in range(2):
                c_local = k*2 + i
                c_global = k*2 + i
                for r in range(18):
                    W_global[c_global][col + r] = W_225[c_local][r]
        
        return U_global, V_global, W_global

    try:
        U_36, V_36, W_36 = build_225_and_combine()
        if verify(U_36, V_36, W_36):
            best_U, best_V, best_W = U_36, V_36, W_36
            best_rank = 36
    except Exception as e:
        pass

    # If we have time, try to optimize further? 
    # Given the time limit and complexity, 36 is a solid improvement over 40.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# Helper function to avoid name collision and structure
def build_225_and_combine():
    def strassen_222():
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1
        U_s[0][2] = 1
        V_s[1][2] = -1; V_s[3][2] = -1
        W_s[1][2] = 1
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = -1; W_s[2][3] = 1
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[2][5] = 1
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[3][6] = 1
        return U_s, V_s, W_s

    def schoolbook_221():
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                a = i*2 + j
                b = j
                c = i
                r = i*2 + j
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
        return U, V, W

    def build_225():
        U = [[0]*18 for _ in range(4)]
        V = [[0]*18 for _ in range(10)]
        W = [[0]*18 for _ in range(10)]
        col = 0
        U1, V1, W1 = strassen_222()
        for i in range(4):
            for r in range(7):
                U[i][col + r] = U1[i][r]
        for j in range(2):
            for k in range(2):
                b_local = j*2 + k
                for r in range(7):
                    V[b_local][col + r] = V1[j*2+k][r]
        for k in range(2):
            for i in range(2):
                c_local = k*2 + i
                for r in range(7):
                    W[c_local][col + r] = W1[k*2+i][r]
        col += 7
        U2, V2, W2 = strassen_222()
        for i in range(4):
            for r in range(7):
                U[i][col + r] = U2[i][r]
        for j in range(2):
            for k in range(2):
                b_local = j*2 + (k+2)
                for r in range(7):
                    V[b_local][col + r] = V2[j*2+k][r]
        for k in range(2):
            for i in range(2):
                c_local = (k+2)*2 + i
                for r in range(7):
                    W[c_local][col + r] = W2[k*2+i][r]
        col += 7
        U3, V3, W3 = schoolbook_221()
        for i in range(4):
            for r in range(4):
                U[i][col + r] = U3[i][r]
        for j in range(2):
            b_local = j + 4
            for r in range(4):
                V[b_local][col + r] = V3[j][r]
        for i in range(2):
            c_local = 4*2 + i
            for r in range(4):
                W[c_local][col + r] = W3[i][r]
        return U, V, W

    U_225, V_225, W_225 = build_225()
    U_global = [[0]*36 for _ in range(8)]
    V_global = [[0]*36 for _ in range(20)]
    W_global = [[0]*36 for _ in range(10)]
    col = 0
    for i in range(2):
        for j in range(2):
            a_local = i*2 + j
            a_global = i*4 + j
            for r in range(18):
                U_global[a_global][col + r] = U_225[a_local][r]
    for j in range(2):
        for k in range(5):
            b_local = j*5 + k
            b_global = j*5 + k
            for r in range(18):
                V_global[b_global][col + r] = V_225[b_local][r]
    for k in range(5):
        for i in range(2):
            c_local = k*2 + i
            c_global = k*2 + i
            for r in range(18):
                W_global[c_global][col + r] = W_225[c_local][r]
    col += 18
    for i in range(2):
        for j in range(2):
            a_local = i*2 + j
            a_global = i*4 + (j+2)
            for r in range(18):
                U_global[a_global][col + r] = U_225[a_local][r]
    for j in range(2):
        for k in range(5):
            b_local = j*5 + k
            b_global = (j+2)*5 + k
            for r in range(18):
                V_global[b_global][col + r] = V_225[b_local][r]
    for k in range(5):
        for i in range(2):
            c_local = k*2 + i
            c_global = k*2 + i
            for r in range(18):
                W_global[c_global][col + r] = W_225[c_local][r]
    return U_global, V_global, W_global

# EVOLVE-BLOCK-END
