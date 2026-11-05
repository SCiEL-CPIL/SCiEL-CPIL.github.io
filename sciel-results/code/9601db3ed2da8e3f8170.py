import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
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
                
    start_time = time.time()
    deadline = start_time + 110

    def verify_solution(U, V, W, r, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        err = T - target
        if np.max(np.abs(err)) < 1e-8:
            # Check integer/half-integer
            if np.all(np.abs(2*U - np.round(2*U)) < 1e-9) and \
               np.all(np.abs(2*V - np.round(2*V)) < 1e-9) and \
               np.all(np.abs(2*W - np.round(2*W)) < 1e-9):
                return True
        return False

    # Construct rank-10 decomposition for M_{3,3,2}
    # Local indices: i,j in {0,1,2}, k in {0,1}
    # U: 3x3x10, V: 3x2x10, W: 2x3x10
    U_local = np.zeros((3, 3, 10))
    V_local = np.zeros((3, 2, 10))
    W_local = np.zeros((2, 3, 10))
    
    # Part 1: Strassen for 2x2x2 (i,j in {0,1}, k in {0,1})
    # t=0: P1
    U_local[0,0,0] = 1; U_local[1,1,0] = 1
    V_local[0,0,0] = 1; V_local[1,1,0] = 1
    W_local[0,0,0] = 1; W_local[1,1,0] = 1
    
    # t=1: P2
    U_local[1,0,1] = 1; U_local[1,1,1] = 1
    V_local[0,0,1] = 1
    W_local[1,0,1] = 1
    
    # t=2: P3
    U_local[0,0,2] = 1
    V_local[1,0,2] = 1; V_local[1,1,2] = -1
    W_local[0,1,2] = 1; W_local[1,1,2] = 1
    
    # t=3: P4
    U_local[1,1,3] = 1
    V_local[0,0,3] = 1; V_local[0,1,3] = -1
    W_local[0,0,3] = 1; W_local[1,0,3] = 1
    
    # t=4: P5
    U_local[0,0,4] = 1; U_local[1,0,4] = 1
    V_local[1,1,4] = 1
    W_local[0,0,4] = -1; W_local[0,1,4] = 1
    
    # t=5: P6
    U_local[1,0,5] = 1; U_local[0,0,5] = -1
    V_local[0,0,5] = 1; V_local[0,1,5] = 1
    W_local[1,1,5] = 1
    
    # t=6: P7
    U_local[1,1,6] = 1; U_local[0,1,6] = -1
    V_local[0,1,6] = 1; V_local[1,1,6] = 1
    W_local[0,0,6] = 1; W_local[1,1,6] = 1
    
    # Part 2: A12 * Bbot (i in {0,1}, j=2, k in {0,1})
    # t=7
    U_local[0,2,7] = 1; U_local[1,2,7] = 1
    V_local[2,0,7] = 1; V_local[2,1,7] = 1
    W_local[0,0,7] = 1; W_local[0,1,7] = 1
    W_local[1,0,7] = 1; W_local[1,1,7] = 1
    
    # Part 3: A21 * Btop (i=2, j in {0,1}, k in {0,1})
    # t=8
    U_local[2,0,8] = 1; U_local[2,1,8] = 1
    V_local[0,0,8] = 1; V_local[0,1,8] = 1
    V_local[1,0,8] = 1; V_local[1,1,8] = 1
    W_local[0,2,8] = 1; W_local[1,2,8] = 1
    
    # Part 4: A22 * Bbot (i=2, j=2, k in {0,1})
    # t=9
    U_local[2,2,9] = 1
    V_local[2,0,9] = 1; V_local[2,1,9] = 1
    W_local[0,2,9] = 1; W_local[1,2,9] = 1

    # Construct full U, V, W for M_{3,3,8}
    # 4 blocks of size 2 in k dimension
    r_total = 40
    U = np.zeros((9, r_total))
    V = np.zeros((24, r_total))
    W = np.zeros((24, r_total))
    
    for l in range(4):
        offset = l * 10
        # U is same for all blocks
        U[:, offset:offset+10] = U_local.reshape(9, 10)
        
        # V: k index maps to global k = 2*l + k_local
        for j in range(3):
            for k_loc in range(2):
                k_glob = 2*l + k_loc
                V[j*8 + k_glob, offset:offset+10] = V_local[j, k_loc, :]
        
        # W: k index maps to global k = 2*l + k_local
        for k_loc in range(2):
            k_glob = 2*l + k_loc
            for i in range(3):
                W[k_glob*3 + i, offset:offset+10] = W_local[k_loc, i, :]

    # Verify
    if verify_solution(U, V, W, r_total, n, m, p):
        U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
        V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
        W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
