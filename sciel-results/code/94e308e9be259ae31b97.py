import numpy as np
import random
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 75
    
    # We attempt to find a lower-rank exact decomposition using a greedy
    # rank-1 peeling approach with exact rational arithmetic (fractions).
    # The tensor T[i,j,k] = 1 if this is a valid (i,j,k) triple for matrix mult.
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # and we want to write T as sum of rank-1 tensors u⊗v⊗w
    # where u is in R^n, v in R^m, w in R^p.
    # The decomposition gives U, V, W matrices such that:
    # (AB)[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] ... 
    # Wait, let me re-read the convention.
    # W[k*n+i, r] means W is indexed by (k,i) where k is the column of B / row of output? 
    # Actually, the standard convention for matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T_{i,j,k} = 1 (scalar) for each triple.
    # A rank-r decomposition: T_{i,j,k} = sum_{r=1}^r u_r[i] v_r[j] w_r[k]
    # Then U has shape (n*m, r) with U[i*m+j, r] = u_r[i] * ... no.
    # Let me re-read: "U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The condition is: for all i,j,k: delta_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where delta_{i,j,k} = 1 for all valid (i,j,k).
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Greedy approach: repeatedly find a rank-1 component that best approximates the remaining tensor,
    # subtract it, and repeat. Use exact fractions to ensure validity.
    
    from fractions import Fraction
    
    # Initialize the residual tensor
    # T[i, j, k] = 1 for all i,j,k
    T = [[[Fraction(1) for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    U_list = []
    V_list = []
    W_list = []
    
    max_rank = rank_schoolbook  # Don't exceed schoolbook
    
    for r in range(max_rank):
        # Find the best rank-1 approximation to T
        # We'll try a few heuristic approaches:
        # 1. For each (i,j), find the best w vector
        # 2. For each (j,k), find the best u vector
        # 3. For each (i,k), find the best v vector
        # Pick the one with the largest "weight"
        
        best_score = Fraction(0)
        best_u = None
        best_v = None
        best_w = None
        
        # Try fixing (i, j) and finding optimal w
        for i in range(n):
            for j in range(m):
                # We want to find w[k] to maximize sum_k w[k] * T[i,j,k] * (something)
                # Actually, for a rank-1 update u⊗v⊗w, the contribution to T[i,j,k] is u[i]*v[j]*w[k]
                # We want to choose u, v, w to best match T.
                # A simple greedy: pick the (i,j,k) with the largest |T[i,j,k]|, set u[i]=1, v[j]=1, w[k]=T[i,j,k]
                # But this is very crude.
                
                # Better: for fixed i,j, the optimal w is proportional to T[i,j,:].
                # Let's just pick the entry with max absolute value in the whole tensor.
                pass
        
        # Simplest greedy: find the (i,j,k) with the largest |T[i,j,k]|, 
        # set u[i]=1, v[j]=1, w[k]=T[i,j,k], subtract u⊗v⊗w from T.
        max_abs = Fraction(0)
        max_i = max_j = max_k = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i][j][k]) > max_abs:
                        max_abs = abs(T[i][j][k])
                        max_i, max_j, max_k = i, j, k
        
        if max_abs == 0:
            break
        
        i0, j0, k0 = max_i, max_j, max_k
        val = T[i0][j0][k0]
        
        # Set u[i0] = 1, v[j0] = 1, w[k0] = val
        # This makes the (i0,j0,k0) entry become 0 after subtraction.
        u = [Fraction(0)] * n
        v = [Fraction(0)] * m
        w = [Fraction(0)] * p
        u[i0] = Fraction(1)
        v[j0] = Fraction(1)
        w[k0] = val
        
        # Subtract u⊗v⊗w from T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i][j][k] -= u[i] * v[j] * w[k]
        
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Check if T is all zeros
        all_zero = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i][j][k] != 0:
                        all_zero = False
                        break
                if not all_zero:
                    break
            if not all_zero:
                break
        if all_zero:
            break
    
    rank = len(U_list)
    
    # If we didn't finish (shouldn't happen if rank <= schoolbook), fall back
    if rank > rank_schoolbook:
        # Fall back to schoolbook
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the output matrices
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U_mat[i * m + j][r] = U_list[r][i]
        for j in range(m):
            for k in range(p):
                V_mat[j * p + k][r] = V_list[r][j]
        for k in range(p):
            for i in range(n):
                W_mat[k * n + i][r] = W_list[r][k]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
