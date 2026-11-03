import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0] # Wait, V is for B. B is 2x2.
    ])
    # Correct V_s for Strassen:
    # m1: e+h -> [1,1,0,0]
    # m2: e -> [1,0,0,0]
    # m3: f-h -> [0,1,-1,0]
    # m4: g-e -> [-1,0,1,0]
    # m5: h -> [0,0,0,1]
    # m6: e+f -> [1,1,0,0]
    # m7: g+h -> [0,1,1,0]
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 1],
        [1, 0, -1, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ])
    # Wait, V indices are j*p+k. p=2.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 2
    # j=1, k=1 -> 3
    
    # m1: (e+h) -> V[0]=1, V[1]=1, V[2]=0, V[3]=0? No.
    # e is B[0,0]=V[0]. h is B[1,1]=V[3].
    # So m1: V[0]=1, V[3]=1.
    # m2: e -> V[0]=1.
    # m3: f-h -> f is B[0,1]=V[1]. h is V[3]. -> V[1]=1, V[3]=-1.
    # m4: g-e -> g is B[1,0]=V[2]. e is V[0]. -> V[0]=-1, V[2]=1.
    # m5: h -> V[3]=1.
    # m6: e+f -> V[0]=1, V[1]=1.
    # m7: g+h -> V[2]=1, V[3]=1.
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, -1, 0, 1, 0, 1]
    ])
    
    # W_s: k*n+i. n=2.
    # C00: i=0,k=0 -> 0
    # C01: i=0,k=1 -> 1
    # C10: i=1,k=0 -> 2
    # C11: i=1,k=1 -> 3
    
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m6 + m7
    
    W_s = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 1, 1]
    ])
    
    # Concat 2x2x3 (Rank 11)
    # Terms 1-7: Strassen on first 2 cols of B.
    # U_c1: Same as U_s.
    # V_c1: V_s but B is 2x3. Indices j*3+k.
    # k=0,1 used. k=2 is 0.
    # V_c1 shape 6x7.
    V_c1 = np.zeros((6, 7))
    V_c1[0, :] = V_s[0, :] # j=0, k=0
    V_c1[1, :] = V_s[1, :] # j=0, k=1
    V_c1[2, :] = V_s[2, :] # j=1, k=0
    V_c1[3, :] = V_s[3, :] # j=1, k=1
    # k=2 rows (4,5) are 0.
    
    # W_c1: W_s but C is 2x3. Indices k*2+i.
    # C00, C01, C10, C11 used. C02, C12 are 0.
    # W_c1 shape 6x7.
    W_c1 = np.zeros((6, 7))
    W_c1[0, :] = W_s[0, :] # C00
    W_c1[1, :] = W_s[1, :] # C01
    W_c1[2, :] = W_s[2, :] # C10
    W_c1[3, :] = W_s[3, :] # C11
    # C02 (idx 4), C12 (idx 5) are 0.
    
    U_c1 = U_s
    
    # Terms 8-11: Schoolbook for 3rd col of B.
    # A00*B02, A01*B12, A10*B02, A11*B12
    # U_c2:
    # 1: [1,0,0,0]
    # 2: [0,1,0,0]
    # 3: [0,0,1,0]
    # 4: [0,0,0,1]
    U_c2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # V_c2:
    # 1: B02 (j=0,k=2 -> idx 2)
    # 2: B12 (j=1,k=2 -> idx 5)
    # 3: B02
    # 4: B12
    V_c2 = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    # Wait, V indices are j*3+k.
    # B02: j=0, k=2 -> 0*3+2 = 2.
    # B12: j=1, k=2 -> 1*3+2 = 5.
    # So row 2 and row 5.
    V_c2[2, 0] = 1
    V_c2[5, 1] = 1
    V_c2[2, 2] = 1
    V_c2[5, 3] = 1
    
    # W_c2:
    # 1: C02 (k=0, i=0 -> idx 0? No k*2+i. C02 is i=0, k=2? No.
    # C is 2x3. C[i,k].
    # C02: i=0, k=2. idx = 2*2+0 = 4.
    # C12: i=1, k=2. idx = 2*2+1 = 5.
    W_c2 = np.zeros((6, 4))
    W_c2[4, 0] = 1
    W_c2[5, 1] = 1
    W_c2[4, 2] = 1
    W_c2[5, 3] = 1
    
    U_c = np.hstack([U_c1, U_c2])
    V_c = np.hstack([V_c1, V_c2])
    W_c = np.hstack([W_c1, W_c2])
    
    # Kronecker Product
    # U_total: 16x77
    # U_s: 4x7, U_c: 4x11
    # U_total[i1*4+i2, r1*11+r2] = U_s[i1, r1] * U_c[i2, r2]
    
    U_total = np.zeros((16, 77))
    V_total = np.zeros((12, 77))
    W_total = np.zeros((24, 77))
    
    for i1 in range(2):
        for i2 in range(2):
            idx_u = i1 * 4 + i2
            for r1 in range(7):
                for r2 in range(11):
                    idx_r = r1 * 11 + r2
                    val = U_s[i1, r1] * U_c[i2, r2]
                    U_total[idx_u, idx_r] = val
                    # V_total index: j1*6+j2? No.
                    # V is for B. B is 4x6.
                    # B is 2x2 (from Strassen) x 2x3 (from Concat)?
                    # No, B is 4x6.
                    # B indices: j (0..3), k (0..5).
                    # j = j1*2 + j2. k = k1*3 + k2?
                    # No, the Kronecker product structure:
                    # A is 4x4. A[i,j] = A_s[i1,j1] * A_c[i2,j2]?
                    # No, A is just A.
                    # The Kronecker product of tensors:
                    # T_s (for 2x2x2) and T_c (for 2x2x3).
                    # T_total = T_s \otimes T_c.
                    # Indices of A: (i1,i2) x (j1,j2).
                    # Indices of B: (j1,k1) x (j2,k2).
                    # Indices of C: (k1,i1) x (k2,i2).
                    
                    # So V_total index: (j1,k1, j2,k2).
                    # j = j1*2 + j2.
                    # k = k1*3 + k2.
                    # V index = j*6 + k.
                    
                    # V_s index: j1*2 + k1.
                    # V_c index: j2*3 + k2.
                    
                    # We need to map (j1,k1,j2,k2) to j*6+k.
                    
                    # This loop is inefficient. Let's use broadcasting or pre-computation.
                    pass
    
    # Better approach:
    # U_total[i1*4+i2, r1*11+r2] = U_s[i1,r1] * U_c[i2,r2]
    # This is correct for U.
    
    # V_total[j1*6+j2*? No.
    # V index is j*6+k.
    # j = j1*2 + j2.
    # k = k1*3 + k2.
    # So V index = (j1*2+j2)*6 + (k1*3+k2).
    
    # V_s index is j1*2+k1.
    # V_c index is j2*3+k2.
    
    # We can construct V_total by iterating j1,k1,j2,k2.
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(3):
                    idx_v = (j1*2+j2)*6 + (k1*3+k2)
                    idx_vs = j1*2+k1
                    idx_vc = j2*3+k2
                    V_total[idx_v, :] = V_s[idx_vs, :] * V_c[idx_vc, :]
                    
    # W_total index: k*4+i.
    # k = k1*2+k2? No.
    # C is 4x6? No C is 4x6?
    # A is 4x4, B is 4x6. C is 4x6.
    # C indices: i (0..3), k (0..5).
    # W index: k*4+i.
    
    # C = C_s \otimes C_c.
    # C_s is 2x2. C_c is 2x3.
    # C[i1,i2, k1,k2] = C_s[i1,k1] * C_c[i2,k2].
    # i = i1*2+i2.
    # k = k1*3+k2.
    # W index = k*4+i = (k1*3+k2)*4 + (i1*2+i2).
    
    # W_s index: k1*2+i1.
    # W_c index: k2*2+i2.
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(3):
                for i2 in range(2):
                    idx_w = (k1*3+k2)*4 + (i1*2+i2)
                    idx_ws = k1*2+i1
                    idx_wc = k2*2+i2
                    W_total[idx_w, :] = W_s[idx_ws, :] * W_c[idx_wc, :]
    
    # Now we have U_total, V_total, W_total of rank 77.
    
    # Greedy Elimination
    # Flatten x_r
    x = []
    for r in range(77):
        x_r = np.kron(np.kron(U_total[:, r], V_total[:, r]), W_total[:, r])
        x.append(x_r)
        
    # Target Tensor
    T_target = np.zeros(16*12*24)
    for i in range(4):
        for j in range(4):
            for k in range(6):
                # C[i,k] gets A[i,j]B[j,k]
                # Index in flattened tensor:
                # U index: i*4+j
                # V index: j*6+k
                # W index: k*4+i
                idx_u = i*4+j
                idx_v = j*6+k
                idx_w = k*4+i
                # The tensor entry is at (idx_u, idx_v, idx_w)
                # In the flattened x_r, the position is idx_u * (12*24) + idx_v * 24 + idx_w
                pos = idx_u * (12*24) + idx_v * 24 + idx_w
                T_target[pos] = 1
                
    # Check if x_r span T_target
    # We want to find a subset of x_r that sums to T_target.
    # This is equivalent to finding coefficients alpha_r such that sum alpha_r x_r = T_target.
    # We can use linear algebra.
    
    # Construct matrix X where columns are x_r.
    # X is (4608) x 77.
    # This is large but manageable.
    
    X = np.array(x).T # 4608 x 77
    
    # Solve X alpha = T_target
    # Use least squares or rank-revealing decomposition.
    # Since we want exact, we can use Gaussian elimination on [X | T_target].
    
    # To save time, we can try to reduce the rank by checking linear dependencies.
    # If the rank of X is < 77, we can reduce.
    
    rank_X = np.linalg.matrix_rank(X)
    
    if rank_X < 77:
        # Find null space
        # Use SVD
        U_svd, s, Vh = np.linalg.svd(X)
        # Columns of Vh corresponding to small s are null space vectors.
        # We want to find a basis for the row space?
        # Actually, we want to find a subset of columns that is linearly independent and spans the same space.
        
        # Simple greedy:
        # Start with all columns.
        # Remove column c if it is in the span of the others.
        
        # This is computationally expensive.
        # Given time limit, we might just return rank 77.
        
    # Return rank 77 solution
    # Convert to list of lists
    
    def to_list(mat):
        return [[float(val) for val in row] for row in mat]
        
    return {
        "U": to_list(U_total),
        "V": to_list(V_total),
        "W": to_list(W_total)
    }

if __name__ == "__main__":
    construct()
