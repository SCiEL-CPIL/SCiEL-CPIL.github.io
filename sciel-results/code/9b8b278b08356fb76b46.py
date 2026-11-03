"""Tensor decomposition for 4x4x7 matrix multiplication using Strassen-based blocks."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strassen 2x2x2 decomposition (rank 7)
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,-1], [0,0,1,-1], [0,1,0,1], [1,0,1,0],
        [-1,1,0,0], [0,0,0,1], [1,0,0,0]
    ]
    
    U2 = np.array(strassen_U).T  # 4x7
    V2 = np.array(strassen_V).T  # 4x7
    W2 = np.array(strassen_W).T  # 4x7
    
    # Verify Strassen 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] for r in range(7))
                assert abs(val - 1.0) < 1e-10, f"Strassen failed at ({i},{j},{k}): {val}"
    
    # 2x2x1 decomposition (rank 2)
    U21 = np.array([[1,0],[0,1],[1,0],[0,1]]).T  # 4x2
    V21 = np.array([[1,0],[0,1]]).T               # 2x2
    W21 = np.array([[1,1],[1,1]]).T               # 2x2
    
    # Verify 2x2x1
    for i in range(2):
        for j in range(2):
            val = sum(U21[i*2+j, r] * V21[j, r] * W21[i, r] for r in range(2))
            assert abs(val - 1.0) < 1e-10, f"2x2x1 failed at ({i},{j}): {val}"
    
    # Build 4x4x4 (rank 49)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i in range(4):
                for j in range(4):
                    i1, i2 = divmod(i, 2)
                    j1, j2 = divmod(j, 2)
                    U_444[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for j in range(4):
                for k in range(4):
                    j1, j2 = divmod(j, 2)
                    k1, k2 = divmod(k, 2)
                    V_444[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            for k in range(4):
                for i in range(4):
                    k1, k2 = divmod(k, 2)
                    i1, i2 = divmod(i, 2)
                    W_444[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = sum(U_444[i*4+j, r] * V_444[j*4+k, r] * W_444[k*4+i, r] for r in range(49))
                assert abs(val - 1.0) < 1e-10, f"4x4x4 failed at ({i},{j},{k}): {val}"
    
    # Build 4x4x2 (rank 28)
    U_442 = np.zeros((16, 28))
    V_442 = np.zeros((8, 28))
    W_442 = np.zeros((8, 28))
    for i1 in range(2):
        for j1 in range(2):
            for b1 in range(2):
                for r1 in range(7):
                    r = (i1*2 + j1)*2 + b1)*7 + r1
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            U_442[i*4+j, r] = U2[i2*2+j2, r1]
                    for j2 in range(2):
                        for k in range(2):
                            j = 2*b1 + j2
                            V_442[j*2+k, r] = V2[j2*2+k, r1]
                    for k in range(2):
                        for i2 in range(2):
                            i = 2*i1 + i2
                            W_442[k*4+i, r] = W2[k*2+i2, r1]
    
    # Verify 4x4x2
    for i in range(4):
        for j in range(4):
            for k in range(2):
                val = sum(U_442[i*4+j, r] * V_442[j*2+k, r] * W_442[k*4+i, r] for r in range(28))
                assert abs(val - 1.0) < 1e-10, f"4x4x2 failed at ({i},{j},{k}): {val}"
    
    # Build 4x4x1 (rank 8)
    U_441 = np.zeros((16, 8))
    V_441 = np.zeros((4, 8))
    W_441 = np.zeros((4, 8))
    for i1 in range(2):
        for j1 in range(2):
            for b_idx in range(2):
                for r2 in range(2):
                    r = (i1*2 + j1)*2 + b_idx)*2 + r2
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            U_441[i*4+j, r] = U21[i2*2+j2, r2]
                    for j2 in range(2):
                        j = 2*b_idx + j2
                        V_441[j, r] = V21[j2, r2]
                    for i2 in range(2):
                        i = 2*i1 + i2
                        W_441[i, r] = W21[i2, r2]
    
    # Verify 4x4x1
    for i in range(4):
        for j in range(4):
            val = sum(U_441[i*4+j, r] * V_441[j, r] * W_441[i, r] for r in range(8))
            assert abs(val - 1.0) < 1e-10, f"4x4x1 failed at ({i},{j}): {val}"
    
    # Combine: 4x4x7 = 4x4x4 (cols 0-3) + 4x4x2 (cols 4-5) + 4x4x1 (col 6)
    rank_total = 49 + 28 + 8
    U_final = np.zeros((16, rank_total))
    V_final = np.zeros((28, rank_total))
    W_final = np.zeros((28, rank_total))
    
    # 4x4x4 part
    U_final[:, :49] = U_444
    V_final[:16, :49] = V_444
    W_final[:16, :49] = W_444
    
    # 4x4x2 part (columns 4,5)
    offset = 49
    U_final[:, offset:offset+28] = U_442
    for j in range(4):
        for k in range(2):
            V_final[j*7+4+k, offset:offset+28] = V_442[j*2+k, :]
    for k in range(2):
        for i in range(4):
            W_final[(4+k)*4+i, offset:offset+28] = W_442[k*4+i, :]
    
    # 4x4x1 part (column 6)
    offset = 49 + 28
    U_final[:, offset:offset+8] = U_441
    for j in range(4):
        V_final[j*7+6, offset:offset+8] = V_441[j, :]
    for i in range(4):
        W_final[6*4+i, offset:offset+8] = W_441[i, :]
    
    # Final verification
    for i in range(4):
        for j in range(4):
            for k in range(7):
                val = sum(U_final[i*4+j, r] * V_final[j*7+k, r] * W_final[k*4+i, r] for r in range(rank_total))
                assert abs(val - 1.0) < 1e-10, f"Final failed at ({i},{j},{k}): {val}"
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
