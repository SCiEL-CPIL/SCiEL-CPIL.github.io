"""Matrix multiplication 3x4 x 4x4 tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]
DEADLINE_SECONDS = 140  # 20s safety margin before 160s kill (or 120s eval limit)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Must equal 1 if (i,j,k) is a valid index, else 0
    # Actually the matrix multiplication tensor is T[i,j,k] = delta[i,j,k] where
    # C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid i,j,k
    # Wait, the standard tensor for matrix mult: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    for r in range(rank):
        u_col = np.array(U[:, r])
        v_col = np.array(V[:, r])
        w_col = np.array(W[:, r])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # T[i,j,k] = 1
                    actual = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                    if abs(actual - expected) > 1e-9:
                        return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T from the decomposition
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_random_search(n, m, p, target_rank, time_budget):
    """Try to find a lower-rank decomposition via random search over small fields."""
    start_time = time.time()
    best_result = None
    
    # Try small prime fields
    for field in [2, 3, 5, 7]:
        if time.time() - start_time > time_budget:
            break
        for trial in range(50):
            if time.time() - start_time > time_budget:
                break
            # Random initialization
            U = np.random.randint(0, field, size=(n*m, target_rank))
            V = np.random.randint(0, field, size=(m*p, target_rank))
            W = np.random.randint(0, field, size=(p*n, target_rank))
            
            # Check if this works over the field
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            
            T = np.zeros((n, m, p), dtype=int)
            for r in range(target_rank):
                T += U3[:,:,r][:, :, np.newaxis] * V3[:,:,r][np.newaxis, :, :] * W3[:,:,r][np.newaxis, :, :]
            
            # Reduce mod field
            T = T % field
            
            # Check if T is all ones
            if np.all(T == 1):
                # Found a valid decomposition!
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return None

def _try_systematic_search(n, m, p, target_rank, time_budget):
    """Systematic search for rank reduction by perturbing schoolbook."""
    start_time = time.time()
    
    # Start with schoolbook
    school = _construct_schoolbook(n, m, p)
    U = [row[:] for row in school["U"]]
    V = [row[:] for row in school["V"]]
    W = [row[:] for row in school["W"]]
    rank = n * m * p
    
    # Try to zero out one column at a time and see if we can fix the others
    # This is a greedy approach - unlikely to work but worth trying
    
    # Alternative: try to find a linear dependency among the columns
    # If columns c1 and c2 are linearly dependent, we can merge them
    
    for col_to_remove in range(rank):
        if time.time() - start_time > time_budget:
            break
        
        # Remove column col_to_remove
        U_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
        V_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
        W_test = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
        
        new_rank = rank - 1
        if new_rank >= target_rank:
            # Verify
            if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
                return {
                    "U": U_test,
                    "V": V_test,
                    "W": W_test
                }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook (guaranteed valid)
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound for M_{3,4,4} is unknown to be less than 48
    # but let's try a few approaches
    
    # Try random search for rank 47
    time_left = DEADLINE_SECONDS - (time.time() - start_time)
    if time_left > 5:
        result = _try_random_search(n, m, p, 47, min(time_left - 2, 30))
        if result is not None:
            if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = 47
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        return best_result
    
    # Return the best result found
    return best_result
