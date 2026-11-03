import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback (Rank 64)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Define the base Strassen decomposition for 2x2 * 2x2 -> 2x2
    # Indices: U (i,j), V (j,k), W (k,i)
    # T[i,j,k] = 1
    # Strassen terms:
    # M1 = (A11 + A22)(B21 + B22) -> C11 += M1, C22 -= M1
    # M2 = (A21 + A22)B11 -> C21 += M2, C11 -= M2
    # M3 = A11(B11 - B21) -> C12 += M3, C22 += M3
    # M4 = A22(B22 - B12) -> C12 -= M4, C21 -= M4
    # M5 = (A11 + A12)B22 -> C12 += M5, C11 += M5
    # M6 = (A21 - A11)(B11 + B12) -> C21 += M6, C22 -= M6
    # M7 = (A12 - A22)(B21 + B22) -> C11 += M7, C22 += M7
    
    # We represent a decomposition as a list of tuples (u, v, w)
    # where u is a dict {index: coeff}, v is dict {index: coeff}, w is dict {index: coeff}
    # Indices for 2x2:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    def make_term(u_dict, v_dict, w_dict):
        return (u_dict, v_dict, w_dict)

    # Strassen terms for 2x2x2
    strassen_2x2x2 = [
        make_term({0:1, 3:1}, {2:1, 3:1}, {0:1, 3:-1}),
        make_term({2:1, 3:1}, {0:1}, {2:1, 0:-1}),
        make_term({0:1}, {0:1, 2:-1}, {1:1, 3:1}),
        make_term({3:1}, {3:1, 1:-1}, {1:-1, 2:-1}),
        make_term({0:1, 1:1}, {3:1}, {1:1, 0:1}),
        make_term({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:-1}),
        make_term({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1})
    ]
    
    # Schoolbook terms for 2x2x2
    schoolbook_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                schoolbook_2x2x2.append(make_term({i*2+j:1}, {j*2+k:1}, {k*2+i:1}))

    # Function to combine decompositions using Kronecker product logic
    # We want to decompose (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2) -> (n1*n2) x (p1*p2)
    # Using blocks of size n1 x m1 * m1 x p1 etc.
    # This is complex. Instead, let's use a simpler "block diagonal" or "block sum" approach.
    
    # Alternative Strategy: 
    # Use the fact that R(n1*n2, m1*m2, p1*p2) <= R(n1,m1,p1) * R(n2,m2,p2) is not generally true for rank,
    # but we can use the tensor product of decompositions.
    # If T1 has rank r1 and T2 has rank r2, then T1 x T2 has rank r1*r2.
    # This is not useful for reducing rank below schoolbook usually, unless we use shared structures.
    
    # Let's try a different structural approach:
    # Split the 2x4 * 4x8 multiplication into two parts:
    # Part 1: 2x2 * 2x4 (using rows 0-1 of A, cols 0-1 of B, etc.)
    # This is getting too abstract. Let's stick to the proven block structure but optimize the selection.
    
    # The current best is 56 (8 blocks of Strassen 2x2x2).
    # Can we do better?
    # R(2,4,8) <= R(2,2,4) * 2? No.
    # R(2,4,8) <= R(2,2,2) * R(1,2,4)?
    # R(1,2,4) is just dot product of length 4, rank 4.
    # So R(2,4,8) <= 7 * 4 = 28?
    # Let's verify this logic.
    # A (2x4) = [A1, A2] where A1, A2 are 2x2.
    # B (4x8) = [B1; B2] where B1, B2 are 2x8? No, B is 4x8.
    # B = [[B11, B12], [B21, B22]] where Bij are 2x4.
    # C = A1*B11 + A2*B21 (left 2x4 part)
    # C = A1*B12 + A2*B22 (right 2x4 part)
    
    # Let's compute the left part C_left (2x4).
    # C_left = A1*B11 + A2*B21.
    # This is a sum of two 2x2 * 2x4 products.
    # We can compute A1*B11 with Strassen (rank 7? No, 2x2*2x4 is not square).
    # 2x2 * 2x4 can be viewed as 2x2 * (2x2 | 2x2).
    # So A1*B11 = A1*[[B11a, B11b]] = [A1*B11a, A1*B11b].
    # Each is 2x2*2x2, rank 7. Total 14.
    # Similarly A2*B21 = [A2*B21a, A2*B21b], rank 14.
    # Total for C_left is 28? But we can share the Strassen pre-computations?
    # Strassen for A1*B11a and A1*B11b shares A1 terms but B terms are different.
    # So no sharing between B11a and B11b.
    # However, we can compute the 2x2*2x4 product directly.
    # What is the rank of 2x2 * 2x4?
    # It is equivalent to 2x2 * 2x2 for two independent outputs.
    # So rank is 7+7=14.
    # So C_left requires 14 (for A1*B11) + 14 (for A2*B21) = 28 terms?
    # Wait, the outputs are added: C_left = A1*B11 + A2*B21.
    # The terms from A1*B11 and A2*B21 are independent sums.
    # So yes, 28 terms for the left half of C.
    # Similarly 28 terms for the right half of C.
    # Total 56. This matches the current solution.
    
    # Is there a way to do 2x2 * 2x4 with rank < 14?
    # 2x2 * 2x4 is a tensor of size 4x4x8.
    # Schoolbook rank is 4*4=16.
    # Strassen gives 14.
    # Is rank 13 possible? Unlikely for small cases.
    
    # What if we view 2x4 * 4x8 differently?
    # Pad A to 4x4? No.
    
    # Let's try to find a decomposition of 2x2 * 2x4 with rank 13 or less using a search.
    # Given time limits, a full search is risky.
    # However, we can try to construct a specific rank 13 decomposition if it exists.
    # Literature: R(2,2,4) is 14? Or 13?
    # Actually, R(2,2,4) <= R(2,2,2) + R(2,2,2) - 1? No.
    
    # Let's assume the rank 56 is close to optimal for this simple block structure.
    # But maybe we can use a "mixed" approach:
    # Use Strassen for the "diagonal" blocks and schoolbook for "off-diagonal" if it saves rank?
    # No, Strassen is always better or equal for 2x2.
    
    # Let's try to implement a generic Kronecker product based decomposition to see if we can find a lower rank.
    # We will try to decompose the tensor T_{2,4,8} as a sum of Kronecker products of smaller tensors.
    
    # T_{n,m,p} can be embedded in T_{n',m',p'} if n|n', m|m', p|p'.
    # But we want to minimize rank.
    
    # Let's try the "tensor product" of decompositions.
    # If we have a decomposition for 2x2*2x2 (rank 7) and 1x2*2x4 (rank 4, which is just dot products),
    # can we combine them?
    # The tensor T_{2,4,8} is not a simple Kronecker product of T_{2,2,2} and something else.
    
    # However, we can use the following identity:
    # T_{n1*n2, m1*m2, p1*p2} <= T_{n1,m1,p1} x T_{n2,m2,p2} + ... (cross terms)
    # This is not standard.
    
    # Let's stick to the block decomposition but verify if we can reduce the number of blocks.
    # Current: 8 blocks of 2x2x2.
    # What if we use 4 blocks of 2x2x4?
    # Each 2x2x4 has rank 14.
    # 4 * 14 = 56. Same.
    
    # What if we use 2 blocks of 2x4x4?
    # 2x4 * 4x4. Rank?
    # 2x4 * 4x4 is 8x16x4 tensor.
    # Schoolbook rank 32.
    # Strassen-like?
    # Split 2x4 into [2x2, 2x2].
    # Split 4x4 into [2x2, 2x2; 2x2, 2x2].
    # Product = A1*B11 + A2*B21 (left 2x2) + A1*B12 + A2*B22 (right 2x2).
    # Each 2x2*2x2 is rank 7.
    # Total 4*7 = 28.
    # So R(2,4,4) <= 28.
    # Then R(2,4,8) <= 2 * R(2,4,4) = 56. Same.
    
    # It seems 56 is a robust bound from these simple block decompositions.
    # To beat 56, we need a more sophisticated decomposition that shares terms across blocks.
    # This is equivalent to finding a low-rank decomposition of the 2x4*4x8 tensor directly.
    
    # Given the difficulty, I will return the rank 56 solution which is verified.
    
    def get_block_decomposition():
        U_22 = np.zeros((4, 7))
        V_22 = np.zeros((4, 7))
        W_22 = np.zeros((4, 7))
        
        U_22[0, 0] = 1; U_22[3, 0] = 1
        V_22[0, 0] = 1; V_22[3, 0] = 1
        W_22[0, 0] = 1; W_22[3, 0] = -1
        U_22[2, 1] = 1; U_22[3, 1] = 1
        V_22[0, 1] = 1
        W_22[2, 1] = 1; W_22[3, 1] = -1
        U_22[0, 2] = 1
        V_22[1, 2] = 1; V_22[3, 2] = -1
        W_22[1, 2] = 1; W_22[3, 2] = 1
        U_22[3, 3] = 1
        V_22[2, 3] = 1; V_22[0, 3] = -1
        W_22[0, 3] = 1; W_22[2, 3] = 1
        U_22[0, 4] = 1; U_22[1, 4] = 1
        V_22[3, 4] = 1
        W_22[0, 4] = -1; W_22[1, 4] = 1
        U_22[2, 5] = 1; U_22[0, 5] = -1
        V_22[0, 5] = 1; V_22[1, 5] = 1
        W_22[3, 5] = 1
        U_22[1, 6] = 1; U_22[3, 6] = -1
        V_22[2, 6] = 1; V_22[3, 6] = 1
        W_22[0, 6] = 1; W_22[1, 6] = -1

        U_global = np.zeros((8, 56))
        V_global = np.zeros((32, 56))
        W_global = np.zeros((16, 56))
        
        def add_strassen_block(U_g, V_g, W_g, u_map, v_map, w_map, start_col):
            for c in range(7):
                for l in range(4):
                    if U_22[l, c] != 0:
                        U_g[u_map[l], start_col + c] += U_22[l, c]
                    if V_22[l, c] != 0:
                        V_g[v_map[l], start_col + c] += V_22[l, c]
                    if W_22[l, c] != 0:
                        W_g[w_map[l], start_col + c] += W_22[l, c]

        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [0, 1, 8, 9], [0, 1, 2, 3], 0)
        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [2, 3, 10, 11], [4, 5, 6, 7], 7)
        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [4, 5, 12, 13], [8, 9, 10, 11], 14)
        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [6, 7, 14, 15], [12, 13, 14, 15], 21)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [16, 17, 24, 25], [0, 1, 2, 3], 28)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [18, 19, 26, 27], [4, 5, 6, 7], 35)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [20, 21, 28, 29], [8, 9, 10, 11], 42)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [22, 23, 30, 31], [12, 13, 14, 15], 49)
        
        return U_global, V_global, W_global

    U_block, V_block, W_block = get_block_decomposition()
    
    def verify(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((8, 32, 16))
        for c in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        T_orig = np.zeros((8, 32, 16))
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    T_orig[i*4+j, j*8+k, k*2+i] = 1
        
        return np.allclose(T_recon, T_orig, atol=1e-9)

    if verify(U_block, V_block, W_block):
        return {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
