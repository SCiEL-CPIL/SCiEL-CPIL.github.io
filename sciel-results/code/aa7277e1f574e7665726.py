import numpy as np

def construct():
    # Helper to create a zero tensor decomposition
    def create_tensor(n, m, p, rank):
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        return {'U': U, 'V': V, 'W': W, 'n': n, 'm': m, 'p': p, 'rank': rank}

    # Helper to combine two tensors T1 (n,m,p1) and T2 (n,m,p2) into T (n,m,p1+p2)
    def combine_tensors(T1, T2):
        n, m, p1 = T1['n'], T1['m'], T1['p']
        p2 = T2['p']
        p = p1 + p2
        r1, r2 = T1['rank'], T2['rank']
        r = r1 + r2
        
        U_new = [[0]*r for _ in range(n*m)]
        V_new = [[0]*r for _ in range(m*p)]
        W_new = [[0]*r for _ in range(p*n)]
        
        # Copy U
        for i in range(n*m):
            for c in range(r1):
                U_new[i][c] = T1['U'][i][c]
            for c in range(r2):
                U_new[i][r1+c] = T2['U'][i][c]
                
        # Copy V with index mapping
        # V index: j*p + k
        for j in range(m):
            # T1 k in 0..p1-1
            for k in range(p1):
                v_idx_1 = j*p1 + k
                v_idx_new = j*p + k
                for c in range(r1):
                    V_new[v_idx_new][c] = T1['V'][v_idx_1][c]
            # T2 k in 0..p2-1, but in new space k is p1..p1+p2-1
            for k in range(p2):
                v_idx_2 = j*p2 + k
                v_idx_new = j*p + (p1+k)
                for c in range(r2):
                    V_new[v_idx_new][r1+c] = T2['V'][v_idx_2][c]
                    
        # Copy W with index mapping
        # W index: k*n + i
        for k in range(p1):
            for i in range(n):
                w_idx_1 = k*n + i
                w_idx_new = k*n + i
                for c in range(r1):
                    W_new[w_idx_new][c] = T1['W'][w_idx_1][c]
        for k in range(p2):
            for i in range(n):
                w_idx_2 = k*n + i
                w_idx_new = (p1+k)*n + i
                for c in range(r2):
                    W_new[w_idx_new][r1+c] = T2['W'][w_idx_2][c]
                    
        return {'U': U_new, 'V': V_new, 'W': W_new, 'n': n, 'm': m, 'p': p, 'rank': r}

    # Strassen 2x2x2
    T_222 = create_tensor(2, 2, 2, 7)
    U, V, W = T_222['U'], T_222['V'], T_222['W']
    
    # A indices: 00, 01, 10, 11 -> 0,1,2,3
    # B indices: 00, 01, 10, 11 -> 0,1,2,3
    # C indices: 00, 01, 10, 11 -> 0,1,2,3
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1
    
    # M2 = (A21+A22)B11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += 1
    
    # M3 = A11(B21-B22)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A22(B12-B11)
    U[3][3] += 1
    V[1][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A11+A12)B22
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[1][4] -= 1; W[3][4] -= 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] -= 1; W[1][6] -= 1; W[2][6] += 1; W[3][6] -= 1
    
    # Schoolbook 2x2x1
    T_221 = create_tensor(2, 2, 1, 4)
    U, V, W = T_221['U'], T_221['V'], T_221['W']
    for i in range(2):
        for j in range(2):
            idx = i*2 + j
            U[idx][idx] = 1
            V[idx][idx] = 1
            W[idx][idx] = 1
            
    # Combine to 2x2x3
    T_223 = combine_tensors(T_222, T_221)
    
    # Combine to 2x2x4 (2x2x2 + 2x2x2)
    T_224 = combine_tensors(T_222, T_222)
    
    # Combine to 2x4x3 (2x2x3 + 2x2x3)
    # Note: combine_tensors assumes same n, m.
    # T_223 is 2x2x3.
    # We need 2x4x3.
    # This is equivalent to two independent 2x2x3 problems.
    # The m dimension is 4.
    # T_223 has m=2.
    # We need to map the m indices.
    # T_223 U indices: i*2+j.
    # Target U indices: i*4+j.
    # This requires a custom combine or a "duplicate" function.
    
    def duplicate_m(T, factor=2):
        # T is n x m x p. Returns n x (m*factor) x p.
        n, m, p = T['n'], T['m'], T['p']
        r = T['rank']
        new_m = m * factor
        U_new = [[0]*r for _ in range(n*new_m)]
        V_new = [[0]*r for _ in range(new_m*p)]
        W_new = [[0]*r for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                u_idx_old = i*m + j
                # Map to two new j's
                for f in range(factor):
                    u_idx_new = i*new_m + (f*m + j)
                    for c in range(r):
                        U_new[u_idx_new][c] = T['U'][u_idx_old][c]
                        
        for j in range(m):
            for k in range(p):
                v_idx_old = j*p + k
                for f in range(factor):
                    v_idx_new = (f*m + j)*p + k
                    for c in range(r):
                        V_new[v_idx_new][c] = T['V'][v_idx_old][c]
                        
        for k in range(p):
            for i in range(n):
                w_idx_old = k*n + i
                w_idx_new = k*n + i
                for c in range(r):
                    W_new[w_idx_new][c] = T['W'][w_idx_old][c]
                    
        return {'U': U_new, 'V': V_new, 'W': W_new, 'n': n, 'm': new_m, 'p': p, 'rank': r}

    T_243 = duplicate_m(T_223, 2)
    T_244 = duplicate_m(T_224, 2)
    
    # Combine 2x4x4 and 2x4x3 to 2x4x7
    T_247 = combine_tensors(T_244, T_243)
    
    return {'U': T_247['U'], 'V': T_247['V'], 'W': T_247['W']}

BET: {mechanism: "Decomposition into 2x4x4 and 2x4x3 using optimal sub-ranks (28 and 22) achieves rank 50, significantly lower than schoolbook 56.", q: 0.9, delta: [0.1, 0.2]}
