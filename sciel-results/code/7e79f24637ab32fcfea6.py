import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0

    # Schoolbook fallback
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[0][3] = -1
        
        # M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[2][4] = 1
        
        # M6 = (a10-a00)(b00+b10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # M7 = (a01-a11)(b01+b11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[1][6] = -1; W[3][6] = 1
        
        return U, V, W

    # Schoolbook 2x1x2 decomposition (rank 4)
    def schoolbook_2x1x2():
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(4)]
        
        # C_00 = A_0 B_0
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # C_01 = A_0 B_1
        U[0][1] = 1
        V[1][1] = 1
        W[1][1] = 1
        
        # C_10 = A_1 B_0
        U[1][2] = 1
        V[0][2] = 1
        W[2][2] = 1
        
        # C_11 = A_1 B_1
        U[1][3] = 1
        V[1][3] = 1
        W[3][3] = 1
        
        return U, V, W

    # Build the 108 rank decomposition
    def build_108_rank():
        U_S, V_S, W_S = strassen_2x2x2()
        U_SB, V_SB, W_SB = schoolbook_2x1x2()
        
        total_rank = 108
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for i_block in range(2):
            for k_block in range(3):
                base_offset = (i_block * 3 + k_block) * 18
                
                # Block 1: j in {0, 1}
                # Strassen 2x2x2
                for u_row in range(4):
                    i_loc = u_row // 2
                    j_loc = u_row % 2
                    i_glob = i_block * 2 + i_loc
                    j_glob = j_loc
                    u_glob = i_glob * m + j_glob
                    for r in range(7):
                        U[u_glob][base_offset + r] += U_S[u_row][r]
                
                for v_row in range(4):
                    j_loc = v_row // 2
                    k_loc = v_row % 2
                    j_glob = j_loc
                    k_glob = k_block * 2 + k_loc
                    v_glob = j_glob * p + k_glob
                    for r in range(7):
                        V[v_glob][base_offset + r] += V_S[v_row][r]
                
                for w_row in range(4):
                    k_loc = w_row // 2
                    i_loc = w_row % 2
                    k_glob = k_block * 2 + k_loc
                    i_glob = i_block * 2 + i_loc
                    w_glob = k_glob * n + i_glob
                    for r in range(7):
                        W[w_glob][base_offset + r] += W_S[w_row][r]
                        
                # Block 2: j in {2, 3}
                # Strassen 2x2x2
                for u_row in range(4):
                    i_loc = u_row // 2
                    j_loc = u_row % 2
                    i_glob = i_block * 2 + i_loc
                    j_glob = 2 + j_loc
                    u_glob = i_glob * m + j_glob
                    for r in range(7):
                        U[u_glob][base_offset + 7 + r] += U_S[u_row][r]
                
                for v_row in range(4):
                    j_loc = v_row // 2
                    k_loc = v_row % 2
                    j_glob = 2 + j_loc
                    k_glob = k_block * 2 + k_loc
                    v_glob = j_glob * p + k_glob
                    for r in range(7):
                        V[v_glob][base_offset + 7 + r] += V_S[v_row][r]
                
                for w_row in range(4):
                    k_loc = w_row // 2
                    i_loc = w_row % 2
                    k_glob = k_block * 2 + k_loc
                    i_glob = i_block * 2 + i_loc
                    w_glob = k_glob * n + i_glob
                    for r in range(7):
                        W[w_glob][base_offset + 7 + r] += W_S[w_row][r]
                        
                # Block 3: j in {4}
                # Schoolbook 2x1x2
                for u_row in range(2):
                    i_loc = u_row
                    i_glob = i_block * 2 + i_loc
                    j_glob = 4
                    u_glob = i_glob * m + j_glob
                    for r in range(4):
                        U[u_glob][base_offset + 14 + r] += U_SB[u_row][r]
                
                for v_row in range(2):
                    k_loc = v_row
                    j_glob = 4
                    k_glob = k_block * 2 + k_loc
                    v_glob = j_glob * p + k_glob
                    for r in range(4):
                        V[v_glob][base_offset + 14 + r] += V_SB[v_row][r]
                
                for w_row in range(4):
                    k_loc = w_row // 2
                    i_loc = w_row % 2
                    k_glob = k_block * 2 + k_loc
                    i_glob = i_block * 2 + i_loc
                    w_glob = k_glob * n + i_glob
                    for r in range(4):
                        W[w_glob][base_offset + 14 + r] += W_SB[w_row][r]
                        
        return {"U": U, "V": V, "W": W}

    # Verify the decomposition
    def verify_decomposition(U, V, W):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(108):
                            s += U[i*m+j][r] * V[jp*p+k][r] * W[k*n+i][r]
                        expected = 1 if j == jp else 0
                        if abs(s - expected) > 1e-10:
                            return False
        return True

    try:
        result = build_108_rank()
        if verify_decomposition(result["U"], result["V"], result["W"]):
            return result
        else:
            return get_schoolbook()
    except Exception as e:
        return get_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    # Quick verification
    n, m, p = DIMENSIONS
    U, V, W = res["U"], res["V"], res["W"]
    valid = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[jp*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                    expected = 1 if j == jp else 0
                    if abs(s - expected) > 1e-10:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
    print("Valid:", valid)
