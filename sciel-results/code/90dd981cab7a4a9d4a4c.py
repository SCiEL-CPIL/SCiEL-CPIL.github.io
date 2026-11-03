"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For matrix multiplication, T[i,j,k] should be 1 if the indices are valid
    # Actually, the tensor for matrix multiplication (C = A*B) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = delta (it's 1 for all valid i,j,k in the product)
    # Wait, let me reconsider. The matrix multiplication tensor is:
    # T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # The rank decomposition means:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # For the standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    # No wait, that's not right either.
    
    # Let me think more carefully. The matrix multiplication can be written as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a 3-tensor T with indices (i,j,k) where T[i,j,k] = 1 (it's just the product)
    # Actually no. The tensor is defined such that the bilinear form gives the product.
    
    # The standard way: we want U, V, W such that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # because the matrix multiplication tensor has all entries equal to 1
    # (since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]B[j,k] in C[i,k] is 1)
    
    # So we need: for all i in range(n), j in range(m), k in range(p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _to_int_half(val):
    """Convert to integer or half-integer if possible."""
    if abs(val - round(val)) < 1e-9:
        return int(round(val))
    elif abs(val * 2 - round(val * 2)) < 1e-9:
        return round(val * 2) / 2.0
    return val

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions using structured search
    # For 3x4x8, let's try various rank values below 96
    # Known lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n)
    # For 3x4x8: max(12, 32, 24) = 32
    
    # Try structured random search for lower ranks
    random.seed(42)
    
    for target_rank in range(50, best_rank):  # Try ranks from 50 up to best_rank-1
        if time.time() > deadline:
            break
        
        # Try a few random structured decompositions at this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Create random U, V, W with the target rank
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if this decomposition is exact
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                U_list = [[_to_int_half(U[i][r]) for r in range(target_rank)] for i in range(n*m)]
                V_list = [[_to_int_half(V[i][r]) for r in range(target_rank)] for i in range(m*p)]
                W_list = [[_to_int_half(W[i][r]) for r in range(target_rank)] for i in range(p*n)]
                
                # Verify with the list version
                if _verify(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break  # Found one at this rank, move to next lower rank
    
    return best
# EVOLVE-BLOCK-END
