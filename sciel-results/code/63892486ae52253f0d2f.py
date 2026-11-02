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

    # Fast verification helper using numpy
    def verify_fast(U_mat, V_mat, W_mat, r_val):
        """Check if the rank-r decomposition is exact using numpy."""
        U_arr = np.array(U_mat, dtype=np.int64)
        V_arr = np.array(V_mat, dtype=np.int64)
        W_arr = np.array(W_mat, dtype=np.int64)
        
        # Check the "1" entries: T[i,j,k] = 1 for matching indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                    if s != 1:
                        return False
        
        # Check the "0" entries: T[a,b,c] = 0 when indices don't all match
        # We need to check all (a,b,c) where not (i==i2 and j==j2 and k==k3)
        # This is expensive, so we'll use a smarter check.
        # The tensor T is defined such that T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # We want T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k, else 0.
        # Let's construct the full tensor T from the decomposition and compare with the expected tensor.
        
        # Expected tensor E[a,b,c]:
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        # E[a,b,c] = 1 if i==i' and j==j' and k==k', else 0
        
        # Let's build the expected tensor and the computed tensor and compare.
        # T_computed[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
        # This is a 3-way contraction.
        
        # We can compute T_computed as follows:
        # For each t, T_t = U[:,t] outer V[:,t] outer W[:,t]
        # T_computed = sum_t T_t
        
        # But this is large. n*m=6, m*p=27, p*n=18.
        # T_computed is 6x27x18 = 2916 entries.
        # r_val is at most 54.
        # So we do 54 outer products of size 6x27x18. That's fine.
        
        T_comp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for t in range(r_val):
            T_comp += np.einsum('a,b,c->abc', U_arr[:,t], V_arr[:,t], W_arr[:,t])
            
        # Build expected tensor
        T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_exp[a, b, c] = 1
        
        if not np.array_equal(T_comp, T_exp):
            return False
        
        return True

    # Try to find a lower-rank decomposition using structured search
    # The key idea: exploit the block structure.
    # C = A * B where A is 2x3, B is 3x9.
    # We can view B as three 3x3 blocks: B = [B_0, B_1, B_2].
    # Then C = [A*B_0, A*B_1, A*B_2].
    # If we can find a rank-r decomposition for 2x3 * 3x3, then rank for 2x3 * 3x9 is 3*r.
    # Schoolbook rank for 2x3 * 3x3 is 18, so 3*18 = 54.
    # If R(2,3,3) < 18, we can do better.
    # Lower bound for R(2,3,3) is max(6, 9, 6) = 9.
    # So potentially rank 27.
    
    # Let's try to find a rank-r decomposition for 2x3 * 3x3.
    # If we find one with rank r, then total rank is 3*r.
    
    def verify_2x3x3(U3, V3, W3, r3):
        """Verify rank-r3 decomposition for 2x3 * 3x3."""
        n3, m3, p3 = 2, 3, 3
        U_arr = np.array(U3, dtype=np.int64)
        V_arr = np.array(V3, dtype=np.int64)
        W_arr = np.array(W3, dtype=np.int64)
        
        T_comp = np.zeros((n3*m3, m3*p3, p3*n3), dtype=np.int64)
        for t in range(r3):
            T_comp += np.einsum('a,b,c->abc', U_arr[:,t], V_arr[:,t], W_arr[:,t])
        
        T_exp = np.zeros((n3*m3, m3*p3, p3*n3), dtype=np.int64)
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    a = i * m3 + j
                    b = j * p3 + k
                    c = k * n3 + i
                    T_exp[a, b, c] = 1
        
        return np.array_equal(T_comp, T_exp)

    # Try to find a low-rank decomposition for 2x3 * 3x3
    # Start with schoolbook rank 18 and try to reduce
    best_rank_3 = 18
    best_U3, best_V3, best_W3 = None, None, None
    
    n3, m3, p3 = 2, 3, 3
    r3_school = n3 * m3 * p3  # 18
    
    # Build schoolbook for 2x3*3x3
    U3 = [[0] * r3_school for _ in range(n3 * m3)]
    V3 = [[0] * r3_school for _ in range(m3 * p3)]
    W3 = [[0] * r3_school for _ in range(p3 * n3)]
    col3 = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i * m3 + j][col3] = 1
                V3[j * p3 + k][col3] = 1
                W3[k * n3 + i][col3] = 1
                col3 += 1
    
    best_U3, best_V3, best_W3 = U3, V3, W3
    best_rank_3 = r3_school
    
    random.seed(42)
    
    # Try ranks from 17 down to 9
    for r3_try in range(17, 8, -1):
        if time.time() > deadline - 40:
            break
        
        success = False
        for attempt in range(500):
            if time.time() > deadline - 40:
                break
                
            # Random initialization
            U3_r = [[random.randint(-2, 2) for _ in range(r3_try)] for _ in range(n3 * m3)]
            V3_r = [[random.randint(-2, 2) for _ in range(r3_try)] for _ in range(m3 * p3)]
            W3_r = [[random.randint(-2, 2) for _ in range(r3_try)] for _ in range(p3 * n3)]
            
            if verify_2x3x3(U3_r, V3_r, W3_r, r3_try):
                best_rank_3 = r3_try
                best_U3, best_V3, best_W3 = U3_r, V3_r, W3_r
                success = True
                break
        
        if success:
            break
    
    # If we found a better rank for 2x3*3x3, construct the full decomposition
    if best_rank_3 < r3_school:
        total_rank = 3 * best_rank_3
        U_full = [[0] * total_rank for _ in range(n * m)]
        V_full = [[0] * total_rank for _ in range(m * p)]
        W_full = [[0] * total_rank for _ in range(p * n)]
        
        # For each block b in 0,1,2:
        # U_full[i*m + j, b*r3 + t] = U3[i*m3 + j, t]
        # V_full[j*p + (b*p3 + k), b*r3 + t] = V3[j*p3 + k, t]
        # W_full[(b*p3 + k)*n + i, b*r3 + t] = W3[k*n3 + i, t]
        
        for b in range(3):
            for i in range(n3):
                for j in range(m3):
                    for t in range(best_rank_3):
                        U_full[i * m + j][b * best_rank_3 + t] = best_U3[i * m3 + j][t]
            
            for j in range(m3):
                for k in range(p3):
                    for t in range(best_rank_3):
                        V_full[j * p + (b * p3 + k)][b * best_rank_3 + t] = best_V3[j * p3 + k][t]
            
            for k in range(p3):
                for i in range(n3):
                    for t in range(best_rank_3):
                        W_full[(b * p3 + k) * n + i][b * best_rank_3 + t] = best_W3[k * n3 + i][t]
        
        # Verify the full decomposition
        if verify_fast(U_full, V_full, W_full, total_rank):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: try direct search for lower rank on 2x3*3x9
    # This is less likely to succeed but worth trying
    for r_try in [53, 52, 51, 50]:
        if time.time() > deadline - 20:
            break
        
        success = False
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
                
            U_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(n * m)]
            V_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(m * p)]
            W_r = [[random.randint(-2, 2) for _ in range(r_try)] for _ in range(p * n)]
            
            if verify_fast(U_r, V_r, W_r, r_try):
                return {"U": U_r, "V": V_r, "W": W_r}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
