import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank
    rank = n * m * p
    
    # Build the target tensor
    # T[u_idx, v_idx, w_idx] = 1 where:
    # u_idx = i*m + j
    # v_idx = j*p + k  
    # w_idx = k*n + i
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Store the non-zero pattern of the tensor
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx, 1.0))
    
    # For now, return schoolbook decomposition
    # U has shape (U_size, rank), V has shape (V_size, rank), W has shape (W_size, rank)
    U = np.zeros((U_size, rank), dtype=np.float64)
    V = np.zeros((V_size, rank), dtype=np.float64)
    W = np.zeros((W_size, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    rank = len(U[0])
    
    # Verify the decomposition
    error_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0.0
                for r in range(rank):
                    total += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(total - 1.0) > 1e-9:
                    error_count += 1
                    print(f"Error at ({i},{j},{k}): got {total}")
    
    # Also check that non-target entries are zero
    # For any (u_idx, v_idx, w_idx) not in the tensor, the sum should be 0
    tensor_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_set.add((u_idx, v_idx, w_idx))
    
    zero_error_count = 0
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                if (u_idx, v_idx, w_idx) not in tensor_set:
                    total = 0.0
                    for r in range(rank):
                        total += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(total) > 1e-9:
                        zero_error_count += 1
                        if zero_error_count <= 5:
                            print(f"Non-zero at ({u_idx},{v_idx},{w_idx}): {total}")
    
    print(f"Decomposition rank: {rank}")
    print(f"Target entry errors: {error_count}")
    print(f"Non-target entry errors: {zero_error_count}")
    if error_count == 0 and zero_error_count == 0:
        print("Decomposition is VALID")
    else:
        print("Decomposition is INVALID")
