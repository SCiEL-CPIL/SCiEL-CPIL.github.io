import numpy as np

def get_2x2x7_factors():
    # Returns U, V, W for 2x2x7 multiplication
    # U: (4, 25), V: (14, 25), W: (14, 25)
    rank = 25
    U = np.zeros((4, rank))
    V = np.zeros((14, rank))
    W = np.zeros((14, rank))
    
    # Strassen for 2x2x2
    # Indices i,j in {0,1}
    # k in {0,1}
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Strassen terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to add these to U, V, W.
    # Each term t contributes to U, V, W.
    
    # Helper to add a term
    # U_idx, V_idx, W_idx, val
    # But Strassen terms are sums of products.
    # We need to expand them into rank-1 terms?
    # No, Strassen itself is a rank-7 decomposition.
    # The rank-7 decomposition of 2x2x2 is:
    # U, V, W matrices of shape (4,7), (4,7), (4,7).
    
    # Let's define the Strassen factors.
    # U_s: 4x7
    # V_s: 4x7
    # W_s: 4x7
    
    # Standard Strassen decomposition:
    # t=0: M1
    # U: [1, 0, 1, 0] ? No.
    # M1 = (A11+A22)(B11+B22)
    # This is not a single A_ij B_jk term.
    # The decomposition is:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The tensor factors U, V, W must satisfy:
    # sum_t U[i,j,t] V[j,k,t] W[k,i,t] = delta_{ik}? No.
    # It must reproduce the multiplication.
    
    # Let's use the known rank-7 factors for 2x2x2.
    # U:
    # t=0: A11+A22 -> [1,0,0,1] (indices 00,01,10,11)
    # t=1: A21+A22 -> [0,0,1,1]
    # t=2: A11 -> [1,0,0,0]
    # t=3: A22 -> [0,0,0,1]
    # t=4: A11+A12 -> [1,1,0,0]
    # t=5: A21-A11 -> [-1,0,1,0]
    # t=6: A12-A22 -> [0,1,0,-1]
    
    # V:
    # t=0: B11+B22 -> [1,0,0,1] (indices 00,01,10,11)
    # t=1: B11 -> [1,0,0,0]
    # t=2: B12-B22 -> [0,1,-1,0] ? No, B12 is (0,1), B22 is (1,1).
    # Indices for V: j*2+k.
    # j=0,k=0 -> 0. j=0,k=1 -> 1. j=1,k=0 -> 2. j=1,k=1 -> 3.
    # B11 is (0,0). B12 is (0,1). B21 is (1,0). B22 is (1,1).
    # t=0: B11+B22 -> [1,0,0,1]
    # t=1: B11 -> [1,0,0,0]
    # t=2: B12-B22 -> [0,1,0,-1]
    # t=3: B21-B11 -> [-1,0,1,0]
    # t=4: B22 -> [0,0,0,1]
    # t=5: B11+B12 -> [1,1,0,0]
    # t=6: B21+B22 -> [0,0,1,1]
    
    # W:
    # W indices: k*2+i.
    # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    # C11 (i=0,k=0 -> 0): M1 + M4 - M5 + M7
    # C12 (i=0,k=1 -> 1): M3 + M5
    # C21 (i=1,k=0 -> 2): M2 + M4
    # C22 (i=1,k=1 -> 3): M1 - M2 + M3 + M6
    
    # W[t] is a vector of length 4.
    # t=0 (M1): [1, 0, 0, 1]
    # t=1 (M2): [0, 0, 1, -1]
    # t=2 (M3): [0, 1, 0, 1]
    # t=3 (M4): [1, 0, 1, 0]
    # t=4 (M5): [-1, 1, 0, 0]
    # t=5 (M6): [0, 0, 0, 1]
    # t=6 (M7): [1, 0, 0, 0]
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]).T # Shape 4x7
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]).T # Shape 4x7
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]).T # Shape 4x7
    
    return U_s, V_s, W_s

# For 2x2x1 (Matrix Vector)
def get_2x2x1_factors():
    # U: (4, 4), V: (2, 4), W: (2, 4)
    # V indices: j*1+k = j. j in {0,1}.
    # W indices: k*2+i = i. i in {0,1}.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    
    U_mv = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).T # 4x4
    
    V_mv = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T # 2x4
    
    W_mv = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]).T # 2x4
    
    return U_mv, V_mv, W_mv
