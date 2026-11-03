import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    n, m, p = DIMENSIONS
    # Rank 84 decomposition
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # Local indices:
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # C: 0:00, 1:01, 2:10, 3:11
    
    # Each factor is a list of (index, value)
    strassen_factors = [
        {
            "U": [(0, 1), (3, 1)],
            "V": [(0, 1), (3, 1)],
            "W": [(0, 1), (3, 1)]
        },
        {
            "U": [(2, 1), (3, 1)],
            "V": [(0, 1)],
            "W": [(2, 1), (3, -1)]
        },
        {
            "U": [(0, 1)],
            "V": [(1, 1), (3, -1)],
            "W": [(1, 1), (3, 1)]
        },
        {
            "U": [(3, 1)],
            "V": [(2, 1), (0, -1)],
            "W": [(0, 1), (2, 1)]
        },
        {
            "U": [(0, 1), (1, 1)],
            "V": [(3, 1)],
            "W": [(0, -1), (1, 1)]
        },
        {
            "U": [(2, 1), (0, -1)],
            "V": [(0, 1), (1, 1)],
            "W": [(3, 1)]
        },
        {
            "U": [(1, 1), (3, -1)],
            "V": [(2, 1), (3, 1)],
            "W": [(0, 1), (3, -1)]
        }
    ]
    
    # 12 blocks:
    # (a_block, b_block, c_block, d_block)
    # A is 2x2 blocks (a,b), B is 2x2 blocks (c,d)
    # Product contributes to C block (a,d)
    # We have 12 products:
    # A00*B00, A01*B10, A00*B01, A01*B11, A00*B02, A01*B12
    # A10*B00, A11*B10, A10*B01, A11*B11, A10*B02, A11*B12
    
    blocks = [
        (0, 0, 0, 0), # A00 B00
        (0, 1, 1, 0), # A01 B10
        (0, 0, 0, 1), # A00 B01
        (0, 1, 1, 1), # A01 B11
        (0, 0, 0, 2), # A00 B02
        (0, 1, 1, 2), # A01 B12
        (1, 0, 0, 0), # A10 B00
        (1, 1, 1, 0), # A11 B10
        (1, 0, 0, 1), # A10 B01
        (1, 1, 1, 1), # A11 B11
        (1, 0, 0, 2), # A10 B02
        (1, 1, 1, 2)  # A11 B12
    ]
    
    col_offset = 0
    for a, b, c, d in blocks:
        for s_idx in range(7):
            factor = strassen_factors[s_idx]
            col = col_offset + s_idx
            
            # Update U
            for local_i, val in factor["U"]:
                # local_i is 0..3 corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
                # A block is at (a, b)
                # Global row index: (2*a + i_local) * m + (2*b + j_local)
                # local_i encodes i_local and j_local?
                # No, local_i in my list is just the index in the flattened 4-vector.
                # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                i_loc = local_i // 2
                j_loc = local_i % 2
                global_row = (2 * a + i_loc) * m + (2 * b + j_loc)
                U[global_row][col] = val
                
            # Update V
            for local_j, val in factor["V"]:
                # local_j is 0..3 corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
                # B block is at (c, d)
                # Global row index: (2*c + j_loc) * p + (2*d + k_loc)
                j_loc = local_j // 2
                k_loc = local_j % 2
                global_row = (2 * c + j_loc) * p + (2 * d + k_loc)
                V[global_row][col] = val
                
            # Update W
            for local_k, val in factor["W"]:
                # local_k is 0..3 corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
                # C block is at (a, d) because A block row is a, B block col is d
                # Global row index: (2*d + k_loc) * n + (2*a + i_loc)
                # Note: W[k*n + i, r]
                k_loc = local_k // 2
                i_loc = local_k % 2
                global_row = (2 * d + k_loc) * n + (2 * a + i_loc)
                W[global_row][col] = val
                
        col_offset += 7
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple validation
    res = construct()
    U = res["U"]
    V = res["V"]
    W = res["W"]
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check a few entries
    # C[i, k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] ... wait
    # The tensor decomposition defines the multiplication.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In tensor form: T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We need to verify that for all i, j, k, T[i, j, k] = 1 if it's a valid decomposition?
    # No, the tensor for matrix multiplication is T[i, j, k] = 1.
    # So we check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i, j, k.
    
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # This might be slow if done naively, but 16*24*24 * 84 is small.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                prod_sum = np.dot(u_val, v_val) # This is wrong.
                # The term is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # We need element-wise product of the three vectors then sum.
                term = u_val * v_val * w_val
                s = np.sum(term)
                if s != 1:
                    valid = False
                    print(f"Error at {i},{j},{k}: sum={s}")
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")
