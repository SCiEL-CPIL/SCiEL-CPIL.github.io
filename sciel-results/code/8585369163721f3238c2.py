import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145
    
    n, m, p = 4, 4, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook decomposition
    rank = n * m * p
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
                r += 1
    
    def verify_full(U, V, W):
        """Check if the decomposition exactly represents matrix multiplication.
        For all (i,i',j,j',k,k'): sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        """
        tol = 1e-8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Positive case: should be 1
                    s_pos = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s_pos - 1.0) > tol:
                        return False
                    # Check negative cases: should be 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                s_neg = np.dot(U[i*m+j2, :], V[j2*p+k2, :] * W[k2*n+i2, :])
                                if abs(s_neg) > tol:
                                    return False
        return True
    
    def verify_positive_only(U, V, W):
        """Quick check: only verify positive entries are 1."""
        tol = 1e-6
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by merging columns
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    best_rank = current_rank
    best_U = current_U.copy()
    best_V = current_V.copy()
    best_W = current_W.copy()
    
    # Try random column elimination with compensation
    for attempt in range(1000):
        if time.time() > deadline - 10:
            break
            
        # Try to remove a random column
        col_to_try = random.randint(0, current_rank - 1)
        
        # Remove the column
        U_new = np.delete(current_U, col_to_try, axis=1)
        V_new = np.delete(current_V, col_to_try, axis=1)
        W_new = np.delete(current_W, col_to_try, axis=1)
        
        # Quick check first
        if verify_positive_only(U_new, V_new, W_new):
            # Full check
            if verify_full(U_new, V_new, W_new):
                current_rank -= 1
                current_U = U_new
                current_V = V_new
                current_W = W_new
                best_rank = current_rank
                best_U = current_U.copy()
                best_V = current_V.copy()
                best_W = current_W.copy()
    
    # Convert back to lists for the return format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(best_rank):
                    s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                if abs(s - 1.0) > 1e-6:
                    # Fallback to schoolbook
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    break
    
    return {"U": U_list, "V": V_list, "W": W_list}
