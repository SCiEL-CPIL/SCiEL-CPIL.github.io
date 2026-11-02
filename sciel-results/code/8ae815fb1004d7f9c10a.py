import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U, V, W are 4x7 matrices (flattened indices).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = -1; V[3, 2] = 1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = -1; V[3, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _schoolbook_4x4x3():
    """
    Return the schoolbook decomposition for 4x4x3 matrix multiplication.
    U, V, W are (16, 48), (28, 48), (28, 48).
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * 7, rank))
    W = np.zeros((7 * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                U[i * m + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = DIMENSIONS
    
    # Get Strassen 2x2x2
    U_S, V_S, W_S = _strassen_2x2x2()
    
    # Construct 4x4x4 part
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((28, rank_444))
    W_444 = np.zeros((28, rank_444))
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for i in range(4):
                ib = i // 2
                il = i % 2
                for j in range(4):
                    jb = j // 2
                    jl = j % 2
                    U_444[i * 4 + j, col] = U_S[ib * 2 + il, r] * U_S[jb * 2 + jl, s]
                    
            for j in range(4):
                jb = j // 2
                jl = j % 2
                for k in range(4):
                    kb = k // 2
                    kl = k % 2
                    V_444[j * 4 + k, col] = V_S[jb * 2 + kb, r] * V_S[jl * 2 + kl, s]
                    
            for k in range(4):
                kb = k // 2
                kl = k % 2
                for i in range(4):
                    ib = i // 2
                    il = i % 2
                    W_444[k * 4 + i, col] = W_S[kb * 2 + ib, r] * W_S[kl * 2 + il, s]
                    
    # Get Schoolbook 4x4x3
    U_443, V_443, W_443 = _schoolbook_4x4x3()
    
    # Combine
    U = np.hstack([U_444, U_443])
    V = np.hstack([V_444, V_443])
    W = np.hstack([W_444, W_443])
    
    # Convert to list of lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
