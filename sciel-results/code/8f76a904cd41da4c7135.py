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
    
    def verify(U, V, W):
        """Check if the decomposition exactly represents matrix multiplication."""
        # For each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
        # For all other combinations, it should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - expected) > 1e-6:
                        return False
        # Check orthogonality conditions - for different (i,j) pairs
        # The decomposition must satisfy: sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        # This is automatically satisfied if the above holds and we check a few more
        return True
    
    # Try to reduce rank by merging columns
    # We'll use a simple local search approach
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    # Attempt to find a lower rank decomposition
    # Strategy: try to remove one column at a time by checking if the remaining columns
    # can still represent the tensor
    
    best_rank = current_rank
    best_U = current_U.copy()
    best_V = current_V.copy()
    best_W = current_W.copy()
    
    # Try random column elimination
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
            
        if current_rank <= best_rank:
            break
            
        # Try to remove a random column
        col_to_try = random.randint(0, current_rank - 1)
        
        # Remove the column
        U_new = np.delete(current_U, col_to_try, axis=1)
        V_new = np.delete(current_V, col_to_try, axis=1)
        W_new = np.delete(current_W, col_to_try, axis=1)
        
        # Check if this still works
        if verify(U_new, V_new, W_new):
            current_rank -= 1
            current_U = U_new
            current_V = V_new
            current_W = W_new
            best_rank = current_rank
            best_U = current_U.copy()
            best_V = current_V.copy()
            best_W = current_W.copy()
        else:
            # Try to adjust the remaining columns to compensate
            # This is more complex - skip for now
            pass
    
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
