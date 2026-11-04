import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partitions
    # I: {0,1}, {2}
    # J: {0,1}, {2,3}, {4}
    # K: {0,1}, {2,3}, {4,5}, {6}
    
    I_parts = [[0, 1], [2]]
    J_parts = [[0, 1], [2, 3], [4]]
    K_parts = [[0, 1], [2, 3], [4, 5], [6]]
    
    # Precompute Strassen 2x2x2 factors
    # U, V, W are 4x7 for 2x2x2
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    U_strassen = [[0]*7 for _ in range(4)]
    V_strassen = [[0]*7 for _ in range(4)]
    W_strassen = [[0]*7 for _ in range(4)]
    
    # r=0
    U_strassen[0][0] = 1; U_strassen[3][0] = 1
    V_strassen[0][0] = 1; V_strassen[3][0] = 1
    W_strassen[0][0] = 1; W_strassen[3][0] = 1
    # r=1
    U_strassen[2][1] = 1; U_strassen[3][1] = 1
    V_strassen[0][1] = 1
    W_strassen[2][1] = 1; W_strassen[3][1] = -1
    # r=2
    U_strassen[0][2] = 1
    V_strassen[1][2] = 1; V_strassen[3][2] = -1
    W_strassen[1][2] = 1; W_strassen[3][2] = 1
    # r=3
    U_strassen[3][3] = 1
    V_strassen[2][3] = 1; V_strassen[0][3] = -1
    W_strassen[0][3] = 1; W_strassen[2][3] = 1
    # r=4
    U_strassen[0][4] = 1; U_strassen[2][4] = 1
    V_strassen[3][4] = 1
    W_strassen[0][4] = -1; W_strassen[1][4] = 1
    # r=5
    U_strassen[1][5] = 1; U_strassen[0][5] = -1
    V_strassen[0][5] = 1; V_strassen[2][5] = 1
    W_strassen[3][5] = 1
    # r=6
    U_strassen[2][6] = 1; U_strassen[3][6] = -1
    V_strassen[1][6] = 1; V_strassen[3][6] = 1
    W_strassen[0][6] = 1

    # Determine total rank
    total_rank = 0
    blocks = []
    for I_idx, I_part in enumerate(I_parts):
        for J_idx, J_part in enumerate(J_parts):
            for K_idx, K_part in enumerate(K_parts):
                ni = len(I_part)
                nj = len(J_part)
                nk = len(K_part)
                if ni == 2 and nj == 2 and nk == 2:
                    rank = 7
                    typ = 'strassen'
                else:
                    rank = ni * nj * nk
                    typ = 'schoolbook'
                blocks.append((I_part, J_part, K_part, rank, typ))
                total_rank += rank
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    for I_part, J_part, K_part, rank, typ in blocks:
        ni = len(I_part)
        nj = len(J_part)
        nk = len(K_part)
        
        if typ == 'strassen':
            # Map local indices to global
            for r_l in range(7):
                r_g = current_rank + r_l
                for i_l in range(2):
                    for j_l in range(2):
                        i_g = I_part[i_l]
                        j_g = J_part[j_l]
                        U[i_g * m + j_g][r_g] = U_strassen[i_l * 2 + j_l][r_l]
                for j_l in range(2):
                    for k_l in range(2):
                        j_g = J_part[j_l]
                        k_g = K_part[k_l]
                        V[j_g * p + k_g][r_g] = V_strassen[j_l * 2 + k_l][r_l]
                for k_l in range(2):
                    for i_l in range(2):
                        k_g = K_part[k_l]
                        i_g = I_part[i_l]
                        W[k_g * n + i_g][r_g] = W_strassen[k_l * 2 + i_l][r_l]
        else:
            # Schoolbook
            for i_l in range(ni):
                for j_l in range(nj):
                    for k_l in range(nk):
                        r_g = current_rank
                        i_g = I_part[i_l]
                        j_g = J_part[j_l]
                        k_g = K_part[k_l]
                        U[i_g * m + j_g][r_g] = 1
                        V[j_g * p + k_g][r_g] = 1
                        W[k_g * n + i_g][r_g] = 1
                        current_rank += 1
            current_rank += (rank - ni*nj*nk) # Already incremented inside loop? 
            # Wait, the loop increments current_rank for each term.
            # So current_rank is already updated.
            
        current_rank += (rank if typ == 'strassen' else 0) # Strassen rank is 7, loop adds 0.
        # Correction: In schoolbook, current_rank is incremented inside.
        # In Strassen, we need to advance current_rank by 7.
        if typ == 'strassen':
            current_rank += 7
            
    return {"U": U, "V": V, "W": W}
