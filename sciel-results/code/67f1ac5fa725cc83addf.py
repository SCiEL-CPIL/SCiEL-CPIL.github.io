import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS # 3, 5, 6
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook decomposition:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] = A[i, j] B[j, k]
    # Rank-1 term for each (i, j, k) triple? No.
    # The multiplication tensor is T[i, j, k] = 1 if we consider the structure?
    # Actually, the tensor for matrix multiplication M_{n,m,p} has entries:
    # T_{i,j,k} = 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
    # Wait, the standard tensor T is defined such that C = A*B corresponds to
    # <T, A \otimes B> = C.
    # The entries of T are T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # In the schoolbook algorithm, each term is A[i,j] * B[j,k].
    # This contributes to C[i,k].
    # So for each (i, j, k), we have a rank-1 component:
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    # where r is the index of the term (i, j, k).
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
