import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: Strassen(2x2) x Schoolbook(3x3) -> Rank 189
    def get_fallback():
        U2 = np.zeros((4, 7), dtype=int)
        V2 = np.zeros((4, 7), dtype=int)
        W2 = np.zeros((4, 7), dtype=int)
        
        # Strassen 2x2
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1
        
        U2[3, 3] = 1
        V2[0, 3] = -1; V2[2, 3] = 1
        W2[0, 3] = 1; W2[1, 3] = 1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1
        
        U2[0, 5] = -1; U2[2, 5] = 1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1

        U3 = np.zeros((9, 27), dtype=int)
        V3 = np.zeros((9, 27), dtype=int)
        W3 = np.zeros((9, 27), dtype=int)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3[i * 3 + j, r] = 1
                    V3[j * 3 + k, r] = 1
                    W3[k * 3 + i, r] = 1

        rank = 7 * 27
        U6 = np.zeros((36, rank), dtype=int)
        V6 = np.zeros((36, rank), dtype=int)
        W6 = np.zeros((36, rank), dtype=int)
        
        for r2 in range(7):
            for r3 in range(27):
                r = r2 * 27 + r3
                for i2 in range(2):
                    for j2 in range(2):
                        if U2[i2*2+j2, r2] != 0:
                            val_u2 = U2[i2*2+j2, r2]
                            for i3 in range(3):
                                for j3 in range(3):
                                    if U3[i3*3+j3, r3] != 0:
                                        val_u3 = U3[i3*3+j3, r3]
                                        i6 = i2*3 + i3
                                        j6 = j2*3 + j3
                                        U6[i6*6+j6, r] += val_u2 * val_u3
                                        
                for j2 in range(2):
                    for k2 in range(2):
                        if V2[j2*2+k2, r2] != 0:
                            val_v2 = V2[j2*2+k2, r2]
                            for j3 in range(3):
                                for k3 in range(3):
                                    if V3[j3*3+k3, r3] != 0:
                                        val_v3 = V3[j3*3+k3, r3]
                                        j6 = j2*3 + j3
                                        k6 = k2*3 + k3
                                        V6[j6*6+k6, r] += val_v2 * val_v3

                for k2 in range(2):
                    for i2 in range(2):
                        if W2[k2*2+i2, r2] != 0:
                            val_w2 = W2[k2*2+i2, r2]
                            for k3 in range(3):
                                for i3 in range(3):
                                    if W3[k3*3+i3, r3] != 0:
                                        val_w3 = W3[k3*3+i3, r3]
                                        k6 = k2*3 + k3
                                        i6 = i2*3 + i3
                                        W6[k6*6+i6, r] += val_w2 * val_w3
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}

    return get_fallback()
# EVOLVE-BLOCK-END
