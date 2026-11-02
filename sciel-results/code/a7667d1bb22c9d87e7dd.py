import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 3, 7

    # We will build U, V, W as lists of lists to ensure integer/half-integer entries
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (21, rank)
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        # u_vec: length 9
        # v_vec: length 21 (but only first 9 used for 3x3x3 part)
        # w_vec: length 21 (but only first 9 used for 3x3x3 part)
        
        # Append to U, V, W
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Part 1: C(3,3,3) - Winograd 23-mult
    # Indices:
    # A[i,j] -> u[i*m+j]
    # B[j,k] -> v[j*p+k] (here p=3 for this sub-tensor, but global p=7. So v[j*3+k])
    # C[i,k] -> w[k*n+i] (here n=3 for this sub-tensor, but global n=3. So w[k*3+i])
    
    # The 23 terms
    # Format: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i,j): val}
    # v_coeffs: dict {(j,k): val}
    # w_coeffs: dict {(i,k): val} -- Note: w is indexed by k*n+i, so we need to map (i,k) to index k*n+i.
    
    terms_23 = [
        # 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # 2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # 3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # 4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,1):1}),
        # 5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        # 6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),
        # 7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):1}),
        # 8
        ({(0,2):1}, {(0,0):1, (1,0):1, (2,0):1}, {(0,0):1}),
        # 9
        ({(1,2):1}, {(0,0):1, (1,0):1, (2,0):1}, {(1,0):1}),
        # 10
        ({(2,2):1}, {(0,0):1, (1,0):1, (2,0):1}, {(2,0):1}),
        # 11
        ({(0,0):1, (0,2):1}, {(2,0):1, (2,1):1}, {(0,2):1}),
        # 12
        ({(1,0):1, (1,2):1}, {(2,0):1, (2,1):1}, {(1,2):1}),
        # 13
        ({(2,0):1, (2,2):1}, {(2,0):1, (2,1):1}, {(2,2):1}),
        # 14
        ({(0,2):1}, {(0,1):1, (1,1):1, (2,1):1}, {(0,1):1}),
        # 15
        ({(1,2):1}, {(0,1):1, (1,1):1, (2,1):1}, {(1,1):1}),
        # 16
        ({(2,2):1}, {(0,1):1, (1,1):1, (2,1):1}, {(2,1):1}),
        # 17
        ({(0,0):1, (0,2):1}, {(2,1):1, (2,0):-1}, {(0,1):1}),
        # 18
        ({(1,0):1, (1,2):1}, {(2,1):1, (2,0):-1}, {(1,1):1}),
        # 19
        ({(2,0):1, (2,2):1}, {(2,1):1, (2,0):-1}, {(2,1):1}),
        # 20
        ({(0,0):1, (0,2):1}, {(2,2):1}, {(0,2):1}),
        # 21
        ({(1,0):1, (1,2):1}, {(2,2):1}, {(1,2):1}),
        # 22
        ({(2,0):1, (2,2):1}, {(2,2):1}, {(2,2):1}),
        # 23
        ({(0,0):1, (0,2):1}, {(2,2):1, (2,1):-1}, {(0,2):1}),
    ]
    
    for u_dict, v_dict, w_dict in terms_23:
        u_vec = [0]*9
        v_vec = [0]*21 # Global size
        w_vec = [0]*21 # Global size
        
        for (i,j), val in u_dict.items():
            u_vec[i*m+j] = val
            
        for (j,k), val in v_dict.items():
            # k is 0..2 for this part
            v_vec[j*p+k] = val
            
        for (i,k), val in w_dict.items():
            # k is 0..2 for this part
            w_vec[k*n+i] = val
            
        add_term(u_vec, v_vec, w_vec)
        
    # Part 2: C(3,3,4) - Schoolbook
    # A[i,j] (i,j in 0..2) * B[j,k] (j in 0..2, k in 3..6) = C[i,k] (i in 0..2, k in 3..6)
    for i in range(n):
        for j in range(m):
            for k in range(3, 7):
                u_vec = [0]*9
                v_vec = [0]*21
                w_vec = [0]*21
                
                u_vec[i*m+j] = 1
                v_vec[j*p+k] = 1
                w_vec[k*n+i] = 1
                
                add_term(u_vec, v_vec, w_vec)
                
    # Convert to list of lists
    U_final = [[0]*len(U) for _ in range(9)]
    V_final = [[0]*len(V) for _ in range(21)]
    W_final = [[0]*len(W) for _ in range(21)]
    
    for r in range(len(U)):
        for i in range(9):
            U_final[i][r] = U[r][i]
        for j in range(21):
            V_final[j][r] = V[r][j]
        for k in range(21):
            W_final[k][r] = W[r][k]
            
    return {"U": U_final, "V": V_final, "W": W_final}
