import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build Strassen 4x4x4 decomposition
    tensors_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),
    ]

    def build_4x4x4_tensors():
        """Returns list of (u16, v16, w16) for 4x4x4 Strassen."""
        result = []
        
        # M1: (A11+A22)(B11+B22) -> C11
        for (u2, v2, w2) in tensors_222:
            u4 = [0] * 16
            v4 = [0] * 16
            w4 = [0] * 16
            u4[0] += u2[0]; u4[10] += u2[0]
            u4[1] += u2[1]; u4[11] += u2[1]
            u4[4] += u2[2]; u4[14] += u2[2]
            u4[5] += u2[3]; u4[15] += u2[3]
            v4[0] += v2[0]; v4[10] += v2[0]
            v4[1] += v2[1]; v4[11] += v2[1]
            v4[4] += v2[2]; v4[14] += v2[2]
            v4[5] += v2[3]; v4[15] += v2[3]
            w4[0] += w2[0]; w4[1] += w2[1]
            w4[4] += w2[2]; w4[5] += w2[3]
            result.append((u4, v4, w4))
        
        # M2: (A21+A22)B11 -> C21
        for (u2, v2, w2) in tensors_222:
            u4 = [0] * 16; v4 = [0] * 16; w4 = [0] * 16
            u4[8] += u2[0]; u4[10] += u2[0]
            u4[9] += u2[1]; u4[11] += u2[1]
            u4[12] += u2[2]; u4[14] += u2[2]
            u4[13] += u2[3]; u4[15] += u2[3]
            v4[0] += v2[0]; v4[1] += v2[1]
            v4[4] += v2[2]; v4[5] += v2[3]
            w4[8] += w2[0]; w4[9] += w2[1]
            w4[12] += w2[2]; w4[13] += w2[3]
            result.append((u4, v4, w4))
        
        # M3: A11(B12-B22) -> C12
        for (u2, v2, w2) in tensors_222:
            u4 = [0] * 16; v4 = [0] * 16; w4 = [0] * 16
            u4[0] += u2[0]; u4[1] += u2[1]
            u4[4] += u2[2]; u4[5] += u2[3]
            v4[2] += v2[0]; v4[10] -= v2[0]
            v4[3] += v2[1]; v4[11] -= v2[1]
            v4[6] += v2[2]; v4[14] -= v2[2]
            v4[7] += v2[3]; v4[15] -= v2[3]
            w4[2] += w2[0]; w4[3] += w2[1]
            w4[6] += w2[2]; w4[7] += w2[3]
            result.append((u4, v4, w4))
        
        # M4: A22(B21-B11) -> C21
        for (u2, v2, w2) in tensors_222:
            u4 = [0] * 16; v4 = [0] * 16; w4 = [0] * 16
            u4[10] += u2[0]; u4[11] += u2[1]
            u4[14] += u2[2]; u4[15] += u2[3]
            v4[8] += v2[0]; v4[0] -= v2[0]
            v4[9] += v2[1]; v4[1] -= v2[1]
            v4[12] += v2[2]; v4[4] -= v2[2]
            v4[13] += v2[3]; v4[5] -= v2[3]
            w4[8] += w2[0]; w4[9] += w2[1]
            w4[12] += w2[2]; w4[13] += w2[3]
            result.append((u4, v4, w4))
        
        # M5: (A11+A12)B22 -> C12
        for (u2, v2, w2) in tensors_222:
            u4 = [0] * 16; v4 = [0] * 16; w4 = [0] * 16
            u4[0] += u2[0]; u4[2] += u2[0]
            u4[1] += u2[1]; u4[3] += u2[1]
            u4[4] += u2[2]; u4[6] += u2[2]
            u4[5] += u2[3]; u4[7] += u2[3]
            v4[10] += v2[0]; v4[11] += v2[1]
            v4[14] += v2[2]; v4[15] += v2[3]
            w4[2] += w2[0]; w4[3] += w2[1]
            w4[6] += w2[2]; w4[7] += w2[3]
            result.append((u4, v4, w4))
        
        # M6: (A21-A11)(B11+B12) -> C11
        for (u2, v2, w2) in tensors_222:
            u4 = [0] * 16; v4 = [0] * 16; w4 = [0] * 16
            u4[8] += u2[0]; u4[0] -= u2[0]
            u4[9] += u2[1]; u4[1] -= u2[1]
            u4[12] += u2[2]; u4[4] -= u2[2]
            u4[13] += u2[3]; u4[5] -= u2[3]
            v4[0] += v2[0]; v4[2] += v2[0]
            v4[1] += v2[1]; v4[3] += v2[1]
            v4[4] += v2[2]; v4[6] += v2[2]
            v4[5] += v2[3]; v4[7] += v2[3]
            w4[0] += w2[0]; w4[1] += w2[1]
            w4[4] += w2[2]; w4[5] += w2[3]
            result.append((u4, v4, w4))
        
        # M7: (A12-A22)(B21+B22) -> C22
        for (u2, v2, w2) in tensors_222:
            u4 = [0] * 16; v4 = [0] * 16; w4 = [0] * 16
            u4[2] += u2[0]; u4[10] -= u2[0]
            u4[3] += u2[1]; u4[11] -= u2[1]
            u4[6] += u2[2]; u4[14] -= u2[2]
            u4[7] += u2[3]; u4[15] -= u2[3]
            v4[8] += v2[0]; v4[10] += v2[0]
            v4[9] += v2[1]; v4[11] += v2[1]
            v4[12] += v2[2]; v4[14] += v2[2]
            v4[13] += v2[3]; v4[15] += v2[3]
            w4[10] += w2[0]; w4[11] += w2[1]
            w4[14] += w2[2]; w4[15] += w2[3]
            result.append((u4, v4, w4))
        
        return result

    tensors_444 = build_4x4x4_tensors()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: output columns 0-3 (C rows 0-3)
    for (u16, v16, w16) in tensors_444:
        U_list.append(u16[:])
        # Map 4x4 B indices to 4x9 B indices
        v36 = [0] * 36
        for k in range(4):
            for j in range(4):
                v36[k * 9 + j] = v16[k * 4 + j]
        V_list.append(v36)
        # Map 4x4 C indices to 9x4 C indices (trace convention: c = j*4+i)
        w36 = [0] * 36
        for i in range(4):
            for j in range(4):
                w36[j * 4 + i] = w16[i * 4 + j]
        W_list.append(w36)
    
    # Block 2: output columns 4-7 (C rows 4-7)
    for (u16, v16, w16) in tensors_444:
        U_list.append(u16[:])
        v36 = [0] * 36
        for k in range(4):
            for j in range(4):
                v36[k * 9 + (j + 4)] = v16[k * 4 + j]
        V_list.append(v36)
        w36 = [0] * 36
        for i in range(4):
            for j in range(4):
                w36[(j + 4) * 4 + i] = w16[i * 4 + j]
        W_list.append(w36)
    
    # Block 3: output column 8 (C row 8), rank 1
    u16_block3 = [1] * 16
    v36_block3 = [0] * 36
    for k in range(4):
        v36_block3[k * 9 + 8] = 1
    w36_block3 = [0] * 36
    for i in range(4):
        w36_block3[8 * 4 + i] = 1
    
    U_list.append(u16_block3)
    V_list.append(v36_block3)
    W_list.append(w36_block3)
    
    rank = len(U_list)
    U = np.array(U_list, dtype=float).T
    V = np.array(V_list, dtype=float).T
    W = np.array(W_list, dtype=float).T
    
    return {"U": U, "V": V, "W": W}

print(construct())
