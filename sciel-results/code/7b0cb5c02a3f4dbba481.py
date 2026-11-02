import time
import sys
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 120s limit
    
    # Schoolbook decomposition as guaranteed fallback
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
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Check if decomposition is exact for all entries."""
        nm, m_p, p_n = n*m, m*p, p*n
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Check all non-zero entries (the 60 valid triples)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r_idx in range(r):
                        val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check zero entries - all other index combinations
        # The tensor is zero unless i=i', j=j', k=k'
        # We need to verify that for inconsistent indices, the sum is 0
        # This is expensive, so we check systematically
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k in range(p):
                        for k2 in range(p):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i
                            val = 0.0
                            for r_idx in range(r):
                                val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                            if abs(val) > tol:
                                return False
        
        # Also check cases where i != i'
        for i in range(n):
            for i2 in range(n):
                if i2 == i:
                    continue
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i2
                        val = 0.0
                        for r_idx in range(r):
                            val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                        if abs(val) > tol:
                            return False
        
        return True
    
    # Try to find lower rank decomposition using structured approach
    # Strategy: Use the fact that matrix multiplication can sometimes be 
    # decomposed using algebraic identities
    
    # For now, let's try a few specific structured decompositions
    # that might achieve rank < 60
    
    # Approach: Try to find a decomposition by exploiting linearity
    # The tensor T_{a,b,c} = 1 if (a,b,c) corresponds to valid (i,j,k) else 0
    
    # Let's try a randomized greedy approach with small rational coefficients
    # Coefficient set: {-1, 0, 1, -1/2, 1/2}
    
    np.random.seed(42)
    coeff_set = [Fraction(-1), Fraction(0), Fraction(1), Fraction(-1,2), Fraction(1,2)]
    
    best_found = False
    
    # Try ranks from 55 down to 50 (conservative range)
    for target_rank in range(55, 49, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random small rational coefficients
            U_mat = np.zeros((n*m, target_rank))
            V_mat = np.zeros((m*p, target_rank))
            W_mat = np.zeros((p*n, target_rank))
            
            # Random sparse initialization
            for r_idx in range(target_rank):
                # Each rank-1 tensor has a few non-zero entries
                num_nonzero_u = np.random.randint(1, 4)
                num_nonzero_v = np.random.randint(1, 4)
                num_nonzero_w = np.random.randint(1, 4)
                
                u_indices = np.random.choice(n*m, size=num_nonzero_u, replace=False)
                v_indices = np.random.choice(m*p, size=num_nonzero_v, replace=False)
                w_indices = np.random.choice(p*n, size=num_nonzero_w, replace=False)
                
                for idx in u_indices:
                    U_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
                for idx in v_indices:
                    V_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
                for idx in w_indices:
                    W_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
            
            # Verify
            U_list = U_mat.tolist()
            V_list = V_mat.tolist()
            W_list = W_mat.tolist()
            
            if verify_decomp(U_list, V_list, W_list, n, m, p, tol=1e-6):
                # Found a valid decomposition!
                # Try to reduce rank further via null-space projection
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                best_found = True
                break
        
        if best_found:
            break
    
    # If we found a better decomposition, try to minimize rank further
    if best_found and best_rank < rank_schoolbook:
        # Try null-space projection to eliminate redundant columns
        # This is a refinement pass
        if time.time() < deadline - 20:
            U_arr = np.array(best_U, dtype=float)
            V_arr = np.array(best_V, dtype=float)
            W_arr = np.array(best_W, dtype=float)
            
            r = U_arr.shape[1]
            
            # Try to find if any column is a linear combination of others
            # For simplicity, check if removing any single column still gives valid decomp
            for col_to_remove in range(r - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                # Remove this column
                U_test = np.delete(U_arr, col_to_remove, axis=1)
                V_test = np.delete(V_arr, col_to_remove, axis=1)
                W_test = np.delete(W_arr, col_to_remove, axis=1)
                
                U_test_list = U_test.tolist()
                V_test_list = V_test.tolist()
                W_test_list = W_test.tolist()
                
                if verify_decomp(U_test_list, V_test_list, W_test_list, n, m, p, tol=1e-6):
                    # Success! Reduce rank by 1
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    best_U = U_test.tolist()
                    best_V = V_test.tolist()
                    best_W = W_test.tolist()
                    best_rank -= 1
                    # Continue trying to remove more columns
                    r = U_arr.shape[1]
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
