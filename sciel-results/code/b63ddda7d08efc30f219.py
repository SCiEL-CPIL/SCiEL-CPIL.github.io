"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using Strassen-like block decomposition
    # For 2x3 x 3x6, we can try block structures
    
    # Known result: 2x2x2 has rank 7 (Strassen)
    # 2x3 x 3x6 = can be decomposed into blocks
    
    # Let's try a direct search for rank r < 36
    # The tensor is T[i,j,k] = delta_{i,k} (for matrix mult)
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 if this is the (i,j,k) term
    
    # Let's use a different approach: try small ranks with integer/half-integer entries
    
    # For 2x3 x 3x6, theoretical lower bound is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But actual rank is likely higher
    
    # Let's try to construct a decomposition using known results
    # 2x3 x 3x6 can be seen as combining 2x3x3 (rank ?) and 2x3x3
    
    # Actually, let me just try a simple approach: use the schoolbook but see if we can
    # find a better one through random search with small integer values
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try ranks from 18 to 35
    for rank in range(18, 36):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Use small integer values
            U = np.random.randint(-2, 3, size=(n*m, rank))
            V = np.random.randint(-2, 3, size=(m*p, rank))
            W = np.random.randint(-2, 3, size=(p*n, rank))
            
            # Check if this decomposes the tensor
            # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For matrix mult, T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
            # Wait, no. The tensor for matrix multiplication is:
            # T[i,j,k] = delta_{j,k} ? No...
            
            # Actually, the matrix multiplication tensor is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # So the trilinear form is sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]
            # where T[i,j,k] = 1 if the indices match appropriately
            # For standard matrix mult: T[i,j,k] = 1 for all valid i,j,k where the multiplication makes sense
            # Actually T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} = 1 for all i,j,k in range
            
            # Wait, I need to be more careful. The tensor is:
            # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
            # This equals sum_i sum_k C[i,k] (sum_j A[i,j] B[j,k])
            # So T[i,j,k] = 1 for all i,j,k
            
            # Hmm, that doesn't seem right either. Let me think again.
            
            # The matrix multiplication tensor T is defined such that:
            # <T, A ⊗ B ⊗ C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = trace(A * B * C^T) or similar
            
            # Actually for standard matrix multiplication:
            # C = A * B, so C[i,k] = sum_j A[i,j] B[j,k]
            # The bilinear form is: sum_{i,k} C[i,k] * (sum_j A[i,j] B[j,k])
            # = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
            # So T[i,j,k] = 1 for all i,j,k
            
            # That means the tensor is all ones? That can't be right for rank considerations.
            
            # Let me reconsider. The tensor decomposition for matrix multiplication:
            # We want to compute C = A*B using rank r tensors
            # C[i,k] = sum_c U[i,c] V[k,c] ... no
            
            # The standard formulation:
            # We have matrices U (n*m x r), V (m*p x r), W (p*n x r)
            # The decomposition satisfies:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            
            # This is because the tensor T[i,j,k] = 1 for all i,j,k in the appropriate ranges
            
            # Wait, I think I'm overcomplicating. Let me just verify the schoolbook works.
            
            # For schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
            # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_c delta_{col: i*m+j} delta_{col: j*p+k} delta_{col: k*n+i}
            # = 1 if there exists c such that i*m+j = c, j*p+k = c, k*n+i = c
            # In schoolbook, each (i,j,k) has its own column, so the sum is 1 for each (i,j,k)
            # and 0 otherwise. So yes, T[i,j,k] = 1 for all i,j,k.
            
            # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            
            # Let me check my random initialization
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists with integer/half-integer entries
                U_list = [[int(U[i, c]) if U[i, c] == int(U[i, c]) else U[i, c] for c in range(rank)] for i in range(n*m)]
                V_list = [[int(V[i, c]) if V[i, c] == int(V[i, c]) else V[i, c] for c in range(rank)] for i in range(m*p)]
                W_list = [[int(W[i, c]) if W[i, c] == int(W[i, c]) else W[i, c] for c in range(rank)] for i in range(p*n)]
                
                # Verify all entries are integer or half-integer
                def is_valid_entry(x):
                    return abs(x * 2 - round(x * 2)) < 1e-9
                
                all_valid = True
                for row in U_list:
                    for x in row:
                        if not is_valid_entry(x):
                            all_valid = False
                            break
                    if not all_valid:
                        break
                if all_valid:
                    for row in V_list:
                        for x in row:
                            if not is_valid_entry(x):
                                all_valid = False
                                break
                        if not all_valid:
                            break
                if all_valid:
                    for row in W_list:
                        for x in row:
                            if not is_valid_entry(x):
                                all_valid = False
                                break
                            if not all_valid:
                                break
                
                if all_valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break  # Found a valid decomposition for this rank
        
        if best_rank < rank:
            break  # Found a lower rank
    
    return best_result
# EVOLVE-BLOCK-END
