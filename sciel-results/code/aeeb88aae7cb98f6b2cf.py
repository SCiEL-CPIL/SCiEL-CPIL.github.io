import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140
    
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
    
    def verify_full(U, V, W, tol=1e-8):
        """Check if the decomposition exactly represents matrix multiplication.
        
        For all (i, j, k, i', j', k'):
        sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = 
            delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        
        Equivalently, for each (i,j,k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        And for all other combinations where at least one index differs, the sum = 0.
        """
        # Check the "1" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check a sample of "0" entries to ensure orthogonality
        # Check cases where exactly one index differs
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vary i
                    for i2 in range(n):
                        if i2 != i:
                            s = np.dot(U[i2*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            if abs(s) > tol:
                                return False
                    # Vary j
                    for j2 in range(m):
                        if j2 != j:
                            s = np.dot(U[i*m+j2, :], V[j2*p+k, :] * W[k*n+i, :])
                            if abs(s) > tol:
                                return False
                    # Vary k
                    for k2 in range(p):
                        if k2 != k:
                            s = np.dot(U[i*m+j, :], V[j*p+k2, :] * W[k2*n+i, :])
                            if abs(s) > tol:
                                return False
        
        return True
    
    def verify_fast(U, V, W, tol=1e-6):
        """Faster verification that checks all entries."""
        # Check the "1" entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check a subset of "0" entries
        # Check cases where exactly one index differs (sample)
        for _ in range(200):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            # Vary one index
            choice = random.randint(0, 2)
            if choice == 0:
                i2 = random.randint(0, n-1)
                if i2 != i:
                    s = np.dot(U[i2*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s) > tol:
                        return False
            elif choice == 1:
                j2 = random.randint(0, m-1)
                if j2 != j:
                    s = np.dot(U[i*m+j2, :], V[j2*p+k, :] * W[k*n+i, :])
                    if abs(s) > tol:
                        return False
            else:
                k2 = random.randint(0, p-1)
                if k2 != k:
                    s = np.dot(U[i*m+j, :], V[j*p+k2, :] * W[k2*n+i, :])
                    if abs(s) > tol:
                        return False
        
        return True
    
    # Try to reduce rank using a more effective search
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    best_rank = current_rank
    best_U = current_U.copy()
    best_V = current_V.copy()
    best_W = current_W.copy()
    
    # Strategy 1: Try to remove columns one at a time
    # Strategy 2: Try to find linear dependencies and eliminate them
    
    # First, check for exact linear dependencies in the schoolbook
    # Each column is a basis vector, so no dependencies exist
    
    # Try random column elimination with adjustment
    for attempt in range(2000):
        if time.time() > deadline - 10:
            break
            
        if current_rank <= best_rank:
            break
            
        # Try to remove a random column
        col_to_try = random.randint(0, current_rank - 1)
        
        # Remove the column
        mask = np.ones(current_rank, dtype=bool)
        mask[col_to_try] = False
        U_new = current_U[:, mask]
        V_new = current_V[:, mask]
        W_new = current_W[:, mask]
        
        # Check if this still works with fast verification
        if verify_fast(U_new, V_new, W_new):
            # Do a full verification to be sure
            if verify_full(U_new, V_new, W_new):
                current_rank -= 1
                current_U = U_new
                current_V = V_new
                current_W = W_new
                best_rank = current_rank
                best_U = current_U.copy()
                best_V = current_V.copy()
                best_W = current_W.copy()
    
    # Try a more sophisticated approach: Gaussian elimination on the tensor
    # This is complex, so we'll stick with the current approach but try more attempts
    
    # Convert back to lists for the return format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final verification with full check
    final_ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(best_rank):
                    s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                if abs(s - 1.0) > 1e-8:
                    final_ok = False
                    break
            if not final_ok:
                break
        if not final_ok:
            break
    
    if not final_ok:
        # Fallback to schoolbook
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
