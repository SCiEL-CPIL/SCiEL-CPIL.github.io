DIMENSIONS = [4, 5, 6]
import time

def construct():
    # Strassen 2x2x2 rank-7 decomposition
    sU = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    sV = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    sW = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 2x2x6: rank 21
    U_226 = [[0]*21 for _ in range(4)]
    V_226 = [[0]*21 for _ in range(12)]
    W_226 = [[0]*21 for _ in range(12)]
    for t in range(3):
        for i in range(4):
            for c in range(7):
                U_226[i][t*7+c] = sU[i][c]
        for j in range(2):
            for kp in range(2):
                for c in range(7):
                    V_226[t*2+j*2+kp][t*7+c] = sV[j*2+kp][c]
        for kp in range(2):
            for i in range(2):
                for c in range(7):
                    W_226[t*2+kp*2+i][t*7+c] = sW[kp*2+i][c]
    
    # 4x2x6: rank 42
    U_426 = [[0]*42 for _ in range(8)]
    V_426 = [[0]*42 for _ in range(12)]
    W_426 = [[0]*42 for _ in range(24)]
    for t in range(2):
        for ip in range(2):
            for j in range(2):
                for c in range(21):
                    U_426[(t*2+ip)*2+j][t*21+c] = U_226[ip*2+j][c]
        for j in range(2):
            for k in range(6):
                for c in range(21):
                    V_426[j*6+k][t*21+c] = V_226[j*6+k][c]
        for k in range(6):
            for ip in range(2):
                for c in range(21):
                    W_426[k*4+(t*2+ip)][t*21+c] = W_226[k*2+ip][c]
    
    # Full 4x5x6: rank 85
    rank = 85
    U = [[0]*rank for _ in range(20)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(24)]
    
    # j=0,1 (rank 42, offset 0)
    for i in range(4):
        for j in range(2):
            for c in range(42):
                U[i*5+j][c] = U_426[i*2+j][c]
    for j in range(2):
        for k in range(6):
            for c in range(42):
                V[j*6+k][c] = V_426[j*6+k][c]
    for k in range(6):
        for i in range(4):
            for c in range(42):
                W[k*4+i][c] = W_426[k*4+i][c]
    
    # j=2 (rank 1, offset 42)
    for i in range(4):
        U[i*5+2][42] = 1
    for k in range(6):
        V[12+k][42] = 1
    for k in range(6):
        for i in range(4):
            W[k*4+i][42] = 1
    
    # j=3,4 (rank 42, offset 43)
    for i in range(4):
        for j in range(2):
            for c in range(42):
                U[i*5+j+2][43+c] = U_426[i*2+j][c]
    for j in range(2):
        for k in range(6):
            for c in range(42):
                V[(j+2)*6+k][43+c] = V_426[j*6+k][c]
    for k in range(6):
        for i in range(4):
            for c in range(42):
                W[k*4+i][43+c] = W_426[k*4+i][c]
    
    return {"U": U, "V": V, "W": W}
