"""Rewritten program for matrix multiplication 2x3x9 with search for lower rank."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = DIMENSIONS
    r_school = n * m * p  # 54

    # Build the schoolbook decomposition as fallback
    U = [[0] * r_school for _ in range(n * m)]
    V = [[0] * r_school for _ in range(m * p)]
    W = [[0] * r_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verification helper: check if (U,V,W) of rank r exactly decomposes the tensor
    def verify(U_mat, V_mat, W_mat, r_val):
        """Check if the rank-r decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_t U[i*m+j, t] * V[j*p+k, t] should equal 1 if k==j? 
                    # No! The tensor T_{i,j,k} = delta_{j, j} ... 
                    # Actually the matrix multiplication tensor has T[i,j,k] = 1 iff j == j (always?) 
                    # Wait, the tensor for matmul: C_{i,k} = sum_j A_{i,j} B_{j,k}
                    # So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
                    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (in the appropriate grouping)
                    # Let me re-read: "exactly decompose the matrix multiplication tensor"
                    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
                    # The decomposition says: for each t, we have a rank-1 term
                    # U_col_t * V_col_t * W_col_t
                    # The tensor entry T[a, b, c] where a indexes (i,j) of A, b indexes (j,k) of B, c indexes (k,i) of C
                    # T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                    # This should equal 1 if the (j) index in A matches the (j) index in B and (k,i) is consistent
                    # a = i*m + j, b = j'*p + k, c = k'*n + i'
                    # T should be 1 iff j == j' and i == i' (k is free? no, k must match too? Let me think)
                    # C_{i,k} = sum_j A_{i,j} B_{j,k}
                    # So the term A_{i,j} B_{j,k} contributes to C_{i,k}
                    # In tensor form: T[(i,j), (j,k), (i,k)] = 1
                    # With a = i*m+j, b = j*p+k, c = k*n+i
                    # So T[a,b,c] = 1 iff j_a == j_b (the middle index matches)
                    # where j_a = a % m, j_b = b // p
                    pass
        
        # More carefully:
        # a = i_A * m + j_A
        # b = j_B * p + k_B  
        # c = k_C * n + i_C
        # T[a,b,c] = 1 iff j_A == j_B (and implicitly i_C == i_A, k_C == k_B for the output to make sense)
        # Actually no - the tensor is defined such that:
        # sum_{j} A[i,j] B[j,k] = C[i,k]
        # So the trilinear form is: sum_{i,j,k} C[i,k] * (sum_j A[i,j] B[j,k]) = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # This equals sum_t (sum_{i,j} U[i*m+j, t] A[i,j]) (sum_{j,k} V[j*p+k, t] B[j,k]) (sum_{i,k} W[k*n+i, t] C[i,k])
        # So T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t] where a=(i,j), b=(j,k), c=(i,k)
        # T[(i,j),(j',k),(i',k')] = 1 iff i==i' AND j==j' AND k==k'
        # i.e., T = 1 when all indices match
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for t in range(r_val):
                        s += U_mat[a][t] * V_mat[b][t] * W_mat[c][t]
                    if s != 1:
                        return False
        # Also need to verify that T[a,b,c] = 0 when indices don't match
        # This is the harder part - need to check all (a,b,c) where they don't all match
        # For efficiency, check a sample or all if small enough
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            b = j2 * p + k2
                            for i2 in range(n):
                                for k3 in range(p):
                                    c = k3 * n + i2
                                    # Should be 0 unless (i==i2, j==j2, k==k3)
                                    if not (i == i2 and j == j2 and k == k3):
                                        s = 0
                                        for t in range(r_val):
                                            s += U_mat[a][t] * V_mat[b][t] * W_mat[c][t]
                                        if s != 0:
                                            return False
        return True

    # Try to find a lower-rank decomposition using random initialization + gradient-free search
    # This is a hard problem, so we'll do a limited search
    
    best_rank = r_school
    best_U, best_V, best_W = U, V, W
    
    # Attempt a few random rank-reduction strategies
    random.seed(42)
    
    # Try rank 53, 52, ... down to some lower bound
    # The lower bound for 2x3x9 is at least max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But actually the Strassen-like lower bounds apply
    # For now, try a few ranks below 54
    
    for r_try in [53, 52, 51, 50, 49, 48]:
        if time.time() > deadline - 30:
            break
            
        # Random initialization with integer entries
        success = False
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
                
            U_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(n * m)]
            V_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(m * p)]
            W_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(p * n)]
            
            if verify(U_r, V_r, W_r, r_try):
                best_rank = r_try
                best_U, best_V, best_W = U_r, V_r, W_r
                success = True
                break
        
        if success:
            break
    
    if best_rank < r_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
