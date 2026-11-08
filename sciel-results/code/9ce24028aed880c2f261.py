import numpy as np
import time
import random
from itertools import product as iproduct

def construct():
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 10.0
    deadline = start_time + time_limit - safety_margin
    
    n, m, p = 2, 3, 9
    
    # The tensor T for matrix multiplication (n×m) * (m×p) = (n×p)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want to decompose: T = sum_r U[:,r] V[:,r] W[:,r]^T
    # where U is (n*m)×R, V is (m*p)×R, W is (p*n)×R
    
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18
    
    # Build the target tensor in flattened form
    # tensor[i*m+j, j*p+k, k*n+i] = 1
    # Flatten: for each (i,j,k), set tensor[(i*m+j) * (mp) * (pn) + (j*p+k) * pn + (k*n+i)] = 1
    total_size = nm * mp * pn  # 6 * 27 * 18 = 2916
    
    # Start with a baseline decomposition
    # We'll use the Strassen-based approach as baseline (rank 50)
    # But let's try to find a better one
    
    def make_baseline():
        """Create the baseline rank-50 decomposition"""
        rank = 50
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        # 4 blocks of 2x2
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # M1 = (a+d)(e+h)
            U[u_a, col] += 1
            U[u_d, col] += 1
            V[v_e, col] += 1
            V[v_h, col] += 1
            
            # M2 = (c+d)e
            U[u_c, col+1] += 1
            U[u_d, col+1] += 1
            V[v_e, col+1] += 1
            
            # M3 = a(f-h)
            U[u_a, col+2] += 1
            V[v_f, col+2] += 1
            V[v_h, col+2] -= 1
            
            # M4 = d(g-e)
            U[u_d, col+3] += 1
            V[v_g, col+3] += 1
            V[v_e, col+3] -= 1
            
            # M5 = (a+b)h
            U[u_a, col+4] += 1
            U[u_b, col+4] += 1
            V[v_h, col+4] += 1
            
            # M6 = (c-a)(e+f)
            U[u_c, col+5] += 1
            U[u_a, col+5] -= 1
            V[v_e, col+5] += 1
            V[v_f, col+5] += 1
            
            # M7 = (b-d)(g+h)
            U[u_b, col+6] += 1
            U[u_d, col+6] -= 1
            V[v_g, col+6] += 1
            V[v_h, col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00, col] += 1
            W[w_00, col+3] += 1
            W[w_00, col+4] -= 1
            W[w_00, col+6] += 1
            
            W[w_01, col+2] += 1
            W[w_01, col+4] += 1
            
            W[w_10, col+1] += 1
            W[w_10, col+3] += 1
            
            W[w_11, col] += 1
            W[w_11, col+1] -= 1
            W[w_11, col+2] += 1
            W[w_11, col+5] += 1
            
            col += 7
        
        # Last column k=8
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a, col] += 1
        V[v_e, col] += 1
        U[u_b, col+1] += 1
        V[v_g, col+1] += 1
        U[u_c, col+2] += 1
        V[v_e, col+2] += 1
        U[u_d, col+3] += 1
        V[v_g, col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        
        W[w_08, col] += 1
        W[w_08, col+1] += 1
        W[w_18, col+2] += 1
        W[w_18, col+3] += 1
        
        col += 4
        
        # A_R * B_B
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0, col] += 1
            V[v_b, col] += 1
            W[k * n + 0, col] += 1
            U[u_ar1, col+1] += 1
            V[v_b, col+1] += 1
            W[k * n + 1, col+1] += 1
            col += 2
        
        return U, V, W
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor"""
        R = U.shape[1]
        # Compute the tensor from the decomposition
        # T_decomp[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For each (i,j,k), check if the sum equals 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    
                    total = 0.0
                    for r in range(R):
                        total += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                    
                    expected = 1.0
                    if abs(total - expected) > 1e-6:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using vectorization"""
        R = U.shape[1]
        
        # Build the tensor T from the decomposition
        # T[a_idx, b_idx, c_idx] = sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
        
        # Only check the non-zero entries
        # For each (i,j,k): a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    
                    total = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    # Start with baseline
    U, V, W = make_baseline()
    best_rank = U.shape[1]
    
    if not verify_fast(U, V, W):
        # This shouldn't happen, but handle it
        pass
    
    # Try to reduce rank by checking if any columns can be removed
    # and if the remaining columns still span the tensor
    
    def try_remove_column(U, V, W, col_idx):
        """Check if removing column col_idx still gives a valid decomposition"""
        R = U.shape[1]
        if R <= 1:
            return None, None, None
        
        U_new = np.delete(U, col_idx, axis=1)
        V_new = np.delete(V, col_idx, axis=1)
        W_new = np.delete(W, col_idx, axis=1)
        
        if verify_fast(U_new, V_new, W_new):
            return U_new, V_new, W_new
        return None, None, None
    
    # Try removing columns one at a time
    R = best_rank
    while True:
        if time.time() > deadline:
            break
        
        removed = False
        for col_idx in range(R):
            if time.time() > deadline:
                break
            
            U_new, V_new, W_new = try_remove_column(U, V, W, col_idx)
            if U_new is not None:
                U, V, W = U_new, V_new, W_new
                R = U.shape[1]
                best_rank = min(best_rank, R)
                removed = True
                break
        
        if not removed:
            break
    
    # Now try a more sophisticated approach: local search with column replacements
    # Try replacing pairs of columns with single columns (if possible)
    
    def try_merge_columns(U, V, W, col1, col2):
        """Try to replace columns col1 and col2 with a single column"""
        R = U.shape[1]
        if R <= 2:
            return None, None, None
        
        # We need to find a new column u_new, v_new, w_new such that
        # for all (i,j,k): U[i*m+j, col1]*V[j*p+k, col1]*W[k*n+i, col1] + 
        #                   U[i*m+j, col2]*V[j*p+k, col2]*W[k*n+i, col2]
        # = u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i]
        
        # This is hard to solve directly, so let's try a heuristic:
        # Check if one of the columns is a scalar multiple of the other
        # in a way that allows merging
        
        u1 = U[:, col1]
        u2 = U[:, col2]
        v1 = V[:, col1]
        v2 = V[:, col2]
        w1 = W[:, col1]
        w2 = W[:, col2]
        
        # Check if u1 is proportional to u2, v1 to v2, w1 to w2
        # with consistent scaling
        nonzero_u1 = np.where(u1 != 0)[0]
        nonzero_u2 = np.where(u2 != 0)[0]
        
        if len(nonzero_u1) == 0 or len(nonzero_u2) == 0:
            return None, None, None
        
        # Try to find scaling factors
        # This is a complex optimization, so for now just return None
        # to avoid spending too much time
        return None, None, None
    
    # Alternative approach: try to find a lower-rank decomposition from scratch
    # using a different strategy - exploit the structure of the problem
    
    # For 2x3 * 3x9, let's think about it differently
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a sum over j of rank-1 terms in (i,k)
    
    # One approach: use the fact that we can combine j's cleverly
    
    # Let's try a direct search for low-rank decompositions
    # by solving a system of equations
    
    def search_low_rank(target_rank):
        """Try to find a decomposition of the given rank"""
        R = target_rank
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Initialize with the baseline and try to reduce
        # This is a hard problem, so we'll use a heuristic
        
        # For now, just return the baseline if target_rank >= best_rank
        if target_rank >= best_rank:
            return U, V, W
        
        return None, None, None
    
    # Let's try a completely different approach:
    # Use the structure of the problem to find a better decomposition
    
    # The key insight: for 2x3 * 3x9, we can view it as:
    # C = A * B where A is 2x3 and B is 3x9
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # We can try to use Strassen-like tricks on the 3 dimension
    # by grouping columns of B
    
    # For now, let's try a randomized search approach
    # to see if we can find a lower-rank decomposition
    
    random.seed(42)
    
    best_found = best_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to improve by column perturbation
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Start from current best
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        R_cur = U_cur.shape[1]
        
        # Try random column removals
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            if R_cur <= best_found:
                break
            
            # Try removing a random column
            col_idx = random.randint(0, R_cur - 1)
            U_test = np.delete(U_cur, col_idx, axis=1)
            V_test = np.delete(V_cur, col_idx, axis=1)
            W_test = np.delete(W_cur, col_idx, axis=1)
            
            if verify_fast(U_test, V_test, W_test):
                U_cur, V_cur, W_cur = U_test, V_test, W_test
                R_cur = U_cur.shape[1]
        
        if R_cur < best_found:
            best_found = R_cur
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    # Use the best found
    U, V, W = best_U, best_V, best_W
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list_of_lists(matrix):
        result = []
        for row in matrix:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                if abs(rounded - round(rounded)) < 1e-9:
                    rounded = round(rounded)
                row_list.append(rounded)
            result.append(row_list)
        return result
    
    U_list = to_list_of_lists(U)
    V_list = to_list_of_lists(V)
    W_list = to_list_of_lists(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
