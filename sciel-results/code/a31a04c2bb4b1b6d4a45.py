import time
import numpy as np

DIMENSIONS = [3, 4, 4]

_DEADLINE = time.time() + 150.0
_SAFE_MARGIN = 5.0

def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, tol=1e-7):
    n, m, p = DIMENSIONS
    T = _target_tensor()
    r = U.shape[1]
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r)
    W2 = W.reshape(p, n, r)
    C = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
    return np.max(np.abs(C - T)) < tol

def _construct_strassen_222():
    # Returns U, V, W for M_{2,2,2} with rank 7
    # U is 2x2, V is 2x2, W is 2x2 (reshaped to 4x7)
    # Standard Strassen:
    # m1 = (A11-A22)(B22-B11)
    # m2 = (A11+A22)B11
    # m3 = A12(B12-B22)
    # m4 = A21(B11-B21)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We need to map this to the tensor indices.
    # n=2, m=2, p=2
    # U (2*2, 7), V (2*2, 7), W (2*2, 7)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices for U (A): 0=A11, 1=A12, 2=A21, 3=A22
    # Indices for V (B): 0=B11, 1=B12, 2=B21, 3=B22
    # Indices for W (C): 0=C11, 1=C12, 2=C21, 3=C22
    
    # m1
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = -1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2
    U[0, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # m3
    U[1, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # m4
    U[2, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    
    # m6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > _DEADLINE - _SAFE_MARGIN:
        return schoolbook()

    # Build Rank 44 decomposition
    # M_{3,4,4} = M_{1,4,4} (top row) + M_{2,4,4} (bottom 2 rows)
    
    U_total = []
    V_total = []
    W_total = []
    
    # Part 1: M_{1,4,4} (n=1, m=4, p=4)
    # Rank 16
    # Split into two M_{1,4,2}
    # For each M_{1,4,2}, rank 8
    # U (1*4, 8) -> 4x8
    # V (4*2, 8) -> 8x8
    # W (2*1, 8) -> 2x8
    
    for col_offset in range(2): # 2 blocks of p=2
        for j in range(4):
            for k in range(2):
                # Term for A_j * B_{j, k+col_offset*2}
                # U index: j
                # V index: j*2 + (k + col_offset*2)
                # W index: (k + col_offset*2) * 1 + 1 = k + col_offset*2 + 1
                
                u = np.zeros(4)
                v = np.zeros(8)
                w = np.zeros(2)
                
                u[j] = 1.0
                v[j*2 + k + col_offset*2] = 1.0
                w[k + col_offset*2] = 1.0
                
                U_total.append(u)
                V_total.append(v)
                W_total.append(w)
                
    # Part 2: M_{2,4,4} (n=2, m=4, p=4)
    # Rank 28
    # Split into two M_{2,4,2}
    # For each M_{2,4,2}, rank 14
    # U (2*4, 14) -> 8x14
    # V (4*2, 14) -> 8x14
    # W (2*2, 14) -> 4x14
    
    # Strassen for M_{2,2,2}
    U_s, V_s, W_s = _construct_strassen_222()
    
    for col_offset in range(2): # 2 blocks of p=2
        # We need to compute A_{2x4} * B_{4x2}
        # A = [A1, A2] where A1, A2 are 2x2
        # B = [B1, B2]^T where B1, B2 are 2x2
        # C = A1 B1 + A2 B2
        
        # A1 B1 uses Strassen on A1 (rows 0,1 of A1) and B1 (rows 0,1 of B1)
        # A2 B2 uses Strassen on A2 (rows 2,3 of A1? No, A is 2x4, so A1 is cols 0,1, A2 is cols 2,3)
        
        # A indices in U (8x14):
        # A1: 0,1,2,3
        # A2: 4,5,6,7
        
        # B indices in V (8x14):
        # B1: 0,1,2,3 (rows 0,1 of B, cols 0,1 of B_block)
        # B2: 4,5,6,7 (rows 2,3 of B, cols 0,1 of B_block)
        
        # Wait, B is 4x2.
        # B1 is top 2x2. B2 is bottom 2x2.
        # V indices:
        # B1: j=0,1; k=0,1 -> 0,1,2,3
        # B2: j=2,3; k=0,1 -> 4,5,6,7
        
        # C indices in W (4x14):
        # C is 2x2.
        # 0,1,2,3
        
        for c in range(7):
            u = np.zeros(8)
            v = np.zeros(8)
            w = np.zeros(4)
            
            # A1 B1 part
            # U_s indices: 0=A11, 1=A12, 2=A21, 3=A22
            # Map to U_total: A1 is rows 0,1 of A (2x4).
            # A11 is U[0], A12 is U[1], A21 is U[2], A22 is U[3]
            
            # B1 indices: 0=B11, 1=B12, 2=B21, 3=B22
            # Map to V_total: B1 is rows 0,1 of B (4x2).
            # B11 is V[0], B12 is V[1], B21 is V[2], B22 is V[3]
            
            # C indices: 0=C11, 1=C12, 2=C21, 3=C22
            # Map to W_total: C is 2x2.
            # C11 is W[0], C12 is W[1], C21 is W[2], C22 is W[3]
            
            # Apply Strassen term c to A1 B1
            for idx_u, val in enumerate(U_s[:, c]):
                if val != 0:
                    u[idx_u] += val
            for idx_v, val in enumerate(V_s[:, c]):
                if val != 0:
                    v[idx_v] += val
            for idx_w, val in enumerate(W_s[:, c]):
                if val != 0:
                    w[idx_w] += val
                    
            # Apply Strassen term c to A2 B2
            # A2 is rows 2,3 of A (2x4).
            # A21 is U[4], A22 is U[5], A23 is U[6], A24 is U[7]
            # Wait, A is 2x4.
            # A1 = A[:, 0:2]. A2 = A[:, 2:4].
            # U indices for A1: 0,1,2,3.
            # U indices for A2: 4,5,6,7.
            
            # B2 is rows 2,3 of B.
            # V indices for B1: 0,1,2,3.
            # V indices for B2: 4,5,6,7.
            
            # C is the same 2x2 output.
            
            for idx_u, val in enumerate(U_s[:, c]):
                if val != 0:
                    u[idx_u + 4] += val
            for idx_v, val in enumerate(V_s[:, c]):
                if val != 0:
                    v[idx_v + 4] += val
            # W is same
            
            U_total.append(u)
            V_total.append(v)
            W_total.append(w)
            
    # Combine
    U_arr = np.array(U_total).T
    V_arr = np.array(V_total).T
    W_arr = np.array(W_total).T
    
    if not _verify(U_arr, V_arr, W_arr, tol=1e-6):
        return schoolbook()
        
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

construct()
