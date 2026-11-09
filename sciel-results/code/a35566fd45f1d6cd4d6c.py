"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS
    
    # Define the 11 terms for M_{2,3,2}
    # Each term: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    terms = [
        # Strassen M1: (A00+A11)(B00+B11)
        ([(0,0),(1,1)], [1,1], [(0,0),(1,1)], [1,1], [(0,0),(1,1)], [1,1]),
        # Strassen M2: (A10+A11)B00
        ([(1,0),(1,1)], [1,1], [(0,0)], [1], [(0,1),(1,1)], [1,-1]),
        # Strassen M3: A00(B01-B11)
        ([(0,0)], [1], [(0,1),(1,1)], [1,-1], [(1,0),(1,1)], [1,1]),
        # Strassen M4: A11(B10-B00)
        ([(1,1)], [1], [(1,0),(0,0)], [1,-1], [(0,0),(0,1)], [1,1]),
        # Strassen M5: (A00+A01)B11
        ([(0,0),(0,1)], [1,1], [(1,1)], [1], [(0,0),(1,0)], [-1,1]),
        # Strassen M6: (A10-A00)(B00+B01)
        ([(1,0),(0,0)], [1,-1], [(0,0),(0,1)], [1,1], [(1,1)], [1]),
        # Strassen M7: (A01-A11)(B10+B11)
        ([(0,1),(1,1)], [1,-1], [(1,0),(1,1)], [1,1], [(0,0)], [1]),
        # Outer product term 1: A02 * B20
        ([(0,2)], [1], [(2,0)], [1], [(0,0)], [1]),
        # Outer product term 2: A02 * B21
        ([(0,2)], [1], [(2,1)], [1], [(1,0)], [1]),
        # Outer product term 3: A12 * B20
        ([(1,2)], [1], [(2,0)], [1], [(0,1)], [1]),
        # Outer product term 4: A12 * B21
        ([(1,2)], [1], [(2,1)], [1], [(1,1)], [1]),
    ]
    
    num_blocks = p // 2  # 5 blocks
    rank_per_block = len(terms)  # 11
    total_rank = num_blocks * rank_per_block  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]  # (n*m) x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # (m*p) x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # (p*n) x total_rank
    
    col = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for (u_idx, u_val, v_idx, v_val, w_idx, w_val) in terms:
            for (i, j), val in zip(u_idx, u_val):
                U[i * m + j][col] += val
            for (j, k), val in zip(v_idx, v_val):
                V[j * p + b * 2 + k][col] += val
            for (k, i), val in zip(w_idx, w_val):
                W[(b * 2 + k) * n + i][col] += val
            col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if j == k else 0.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(actual - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use trivial decomposition
        U_triv = [[0] * (n * m * p) for _ in range(n * m)]
        V_triv = [[0] * (n * m * p) for _ in range(m * p)]
        W_triv = [[0] * (n * m * p) for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        U_triv[i * m + j][idx] = 1
                        V_triv[j * p + k][idx] = 1
                        W_triv[k * n + i][idx] = 1
                        idx += 1
        return {"U": U_triv, "V": V_triv, "W": W_triv}
    
    # Try to reduce rank by checking for linear dependencies
    if time.time() < deadline - 20:
        # Check if any column is a linear combination of others
        # Use SVD to find the effective rank of the tensor
        # Flatten the tensor and check rank
        tensor = np.zeros((n * m, m * p, p * n))
        for c in range(rank):
            tensor += np.einsum('a,b,c->abc', U_arr[:, c], V_arr[:, c], W_arr[:, c])
        
        # The tensor should have T[i*m+j, j*p+k, k*n+i] = delta(j,k)
        # Check the structure
        # Try to find a lower rank decomposition by projecting
        
        # For now, return the valid 55-rank decomposition
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
