"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import sys
import signal
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    # Construct the schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to numpy for efficient validation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    best_rank = schoolbook_rank
    best_U = U_np
    best_V = V_np
    best_W = W_np

    # Precompute the target tensor structure
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero_indices.append((a, b, c))
    
    # We will try to improve the rank by finding a better decomposition for the 3x3x3 block (k=0,1,2)
    # and combining it with schoolbook for the rest.
    
    # Target tensor for 3x3x3 (k=0,1,2)
    # Indices a in 0..8, b in 0..11 (j*3+k), c in 0..8 (k*3+i)
    # Note: In the full tensor, b index is j*p+k. For the sub-block, we map local k to global k.
    
    # Let's define the sub-tensor for k in 0..2
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    # We will search for a rank-r decomposition of this sub-tensor.
    # The sub-tensor is essentially M_{3,3,3} but with specific indices in the global V and W.
    
    # To simplify, we can search for a decomposition of the standard M_{3,3,3} tensor
    # and then map the indices.
    
    # Standard M_{3,3,3} tensor T_sub[a_sub, b_sub, c_sub]
    # a_sub = i*3 + j
    # b_sub = j*3 + k
    # c_sub = k*3 + i
    
    # If we find U_sub, V_sub, W_sub of rank r such that
    # T_sub[a_sub, b_sub, c_sub] = sum_r U_sub[a_sub, r] V_sub[b_sub, r] W_sub[c_sub, r]
    
    # Then for the global tensor:
    # U_global[a, r] = U_sub[a, r] (since a is the same)
    # V_global[b, r] = V_sub[b_sub, r] where b_sub = j*3+k if b corresponds to j,k in 0..2
    # W_global[c, r] = W_sub[c_sub, r] where c_sub = k*3+i if c corresponds to k,i in 0..2
    
    # For the remaining part (k=3..7), we use schoolbook.
    
    # Let's try a simple random search for the 3x3x3 part with rank < 27.
    
    def search_3x3x3(max_rank=26, max_iter=500):
        # Target tensor entries
        target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    target[a, b, c] = 1.0
        
        # Random search
        np.random.seed(42)
        best_r = 27
        best_U_sub = None
        best_V_sub = None
        best_W_sub = None
        
        for r in range(10, max_rank + 1):
            for _ in range(max_iter // 10): # Reduce iterations per rank
                U_s = np.random.randn(9, r)
                V_s = np.random.randn(9, r)
                W_s = np.random.randn(9, r)
                
                # Check validity
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # We can check this by computing the tensor and comparing.
                # Efficient way: for each r, add outer product.
                T_approx = np.zeros((9, 9, 9))
                for rr in range(r):
                    T_approx += np.einsum('i,j,k->ijk', U_s[:, rr], V_s[:, rr], W_s[:, rr])
                
                # Check error
                err = np.max(np.abs(T_approx - target))
                if err < 1e-5:
                    # Snap to integers/half-integers if possible
                    # For now, just return the float solution
                    return r, U_s, V_s, W_s
                    
        return None

    # Perform search
    search_result = search_3x3x3(max_rank=26, max_iter=200)
    
    if search_result:
        r_sub, U_sub, V_sub, W_sub = search_result
        # Construct global decomposition
        # Rank = r_sub + (27 - 9) + 18 ? No.
        # Schoolbook for k=3..7 is 5 * 9 = 45 terms.
        # Total rank = r_sub + 45.
        
        new_rank = r_sub + 45
        if new_rank < best_rank:
            U_new = np.zeros((9, new_rank))
            V_new = np.zeros((24, new_rank))
            W_new = np.zeros((24, new_rank))
            
            # Copy sub-decomposition
            U_new[:, :r_sub] = U_sub
            # Map V_sub indices
            # V_sub index b_sub = j*3+k. Global V index b = j*8+k.
            for j in range(3):
                for k in range(3):
                    b_sub = j * 3 + k
                    b_global = j * 8 + k
                    V_new[b_global, :r_sub] = V_sub[b_sub]
            # Map W_sub indices
            # W_sub index c_sub = k*3+i. Global W index c = k*3+i (same, since n=3).
            # Wait, global W index is k*n+i. n=3. So c_global = k*3+i.
            # So W_sub indices match W_new indices directly for k=0..2.
            W_new[:, :r_sub] = W_sub
            
            # Add schoolbook for k=3..7
            col = r_sub
            for i in range(3):
                for j in range(3):
                    for k in range(3, 8):
                        U_new[i * 3 + j, col] = 1
                        V_new[j * 8 + k, col] = 1
                        W_new[k * 3 + i, col] = 1
                        col += 1
                        
            best_rank = new_rank
            best_U = U_new
            best_V = V_new
            best_W = W_new

    U_result = to_list(best_U)
    V_result = to_list(best_V)
    W_result = to_list(best_W)

    return {"U": U_result, "V": V_result, "W": W_result}

def to_list(mat):
    result = []
    for row in mat:
        new_row = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                new_row.append(int(round(x)))
            else:
                if abs(2*x - round(2*x)) < 1e-9:
                    new_row.append(round(2*x)/2.0)
                else:
                    new_row.append(round(x, 6))
        result.append(new_row)
    return result

# EVOLVE-BLOCK-END
