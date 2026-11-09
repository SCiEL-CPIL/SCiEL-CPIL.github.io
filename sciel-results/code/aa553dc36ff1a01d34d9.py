import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Start with the schoolbook decomposition (rank 60)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using random search
    # The idea: randomly initialize U, V, W with small rank and check if they
    # can be adjusted to exactly decompose the matrix multiplication tensor
    
    # For each candidate rank r < 60, try random initializations
    # and use iterative refinement
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Known lower bounds: for 3x4x5, the tensor rank is at least 3*4*5/2 = 30
    # but likely much higher. Let's try ranks from 35 to 59
    
    for r in range(35, 60):
        if time.time() > deadline:
            break
        
        # Multiple random attempts for each rank
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            W_r = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
            
            # Check if this decomposition works
            # The tensor T[i,j,k] should equal sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
            # For matrix multiplication, T[i,j,k] = 1 if we're computing (A*B)[i,k] where A is n x m, B is m x p
            # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor)
            # Wait, no. The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = delta_{j,k} in the sense that
            # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is T[i,j,k] = 1 for all i in [n], j in [m], k in [p], but with the constraint that
            # the j index of A and B must match.
            
            # Actually, let me reconsider. The matrix multiplication tensor for n x m times m x p is:
            # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
            # The bilinear form is: sum_{j} A[i,j] * B[j,k] = (A*B)[i,k]
            # So the tensor entries are: T[i, (j,k), l] ... no.
            
            # The standard tensor decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
            # This can be written as: C[i,k] = sum_t (sum_j U[i,j,t] * A[i,j]) * (sum_j V[j,k,t] * B[j,k])
            # Wait, that's not right either.
            
            # Let me think again. The tensor for matrix multiplication M_{n,m,p} is a trilinear form:
            # T(A,B,C) = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
            # No, that's not standard.
            
            # The correct formulation: We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor rank is the minimum r such that there exist tensors U, V, W with:
            # C[i,k] = sum_t U[i,?,t] * V[?,k,t] ... 
            # 
            # Actually, the standard way: The matrix multiplication tensor M_{n,m,p} is defined as:
            # M = sum_{i,j,k} e_i^T (e_i e_j^T) ⊗ (e_j e_k^T) (e_k^T)
            # where the first factor is for A (n x m), second for B (m x p), third for C (n x p).
            # 
            # In the interface, U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
            # The indices are: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
            # 
            # The decomposition requires: for all i,j,k:
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
            # 
            # Wait, that can't be right because the left side depends on i,j,k but the right side is always 1.
            # 
            # Let me reconsider. The matrix multiplication tensor has:
            # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
            # 
            # This is a bilinear map. To express it as a sum of rank-1 tensors:
            # (A*B)[i,k] = sum_t (sum_j U[i,j,t] * A[i,j]) * (sum_j V[j,k,t] * B[j,k])
            # 
            # No, that's not bilinear in the right way.
            # 
            # The correct rank-1 decomposition: 
            # (A*B)[i,k] = sum_t alpha_t(A) * beta_t(B) * gamma_t(C)
            # where alpha_t is linear in A, beta_t is linear in B, and we're computing C.
            # 
            # Actually, for matrix multiplication, the tensor is:
            # M = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_i ⊗ e_k)
            # 
            # In the flattened form:
            # M[(i,j), (j,k), (i,k)] = 1 for all i,j,k
            # 
            # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
            # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
            # 
            # That's what the schoolbook does: for each (i,j,k), there's a separate term.
            # 
            # So my check should be: for all i,j,k, sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
            
            # Check the decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(r):
                            s += U_r[i*m+j][t] * V_r[j*p+k][t] * W_r[k*n+i][t]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found a valid decomposition for this rank
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
