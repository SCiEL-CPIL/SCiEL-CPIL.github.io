"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W uses trace convention: W[k*n+i, r]
    
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook decomposition as baseline
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # For matrix multiplication, we can try various rank values
    
    # Known: rank is at most n*m*p, and for small cases often much lower
    # Let's try ranks from low to high and see if we can find valid ones
    
    # The tensor equation is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (1 if valid mult)
    # Actually the matrix multiplication tensor T[i,j,k] = 1 for the standard basis
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    # where the multiplication is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Actually, the tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    # T[i,j,k] = 1 for all i,j,k in their ranges (it's the identity tensor in a sense)
    # No wait, it's: the multiplication map sends A (n×m), B (m×p) to C = A*B (n×p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k
    
    # We need to find U (n*m × r), V (m*p × r), W (p*n × r) such that:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try random search for lower ranks
    
    for rank in range(15, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Use gradient descent or alternating minimization
            # For simplicity, try a few iterations of a simple update rule
            
            converged = False
            for iteration in range(100):
                if time.time() > deadline - 15:
                    break
                
                # Check if we're close to solution
                # Compute the residual
                max_res = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                            res = abs(val - 1)
                            if res > max_res:
                                max_res = res
                
                if max_res < 1e-8:
                    converged = True
                    break
                
                # Simple alternating update: fix two, solve for third
                # This is complex, so let's just do a random perturbation approach
                # Actually, let's try a different strategy
                
            if converged:
                # Verify with exact rational check
                U_int = np.round(U * 1000) / 1000
                V_int = np.round(V * 1000) / 1000
                W_int = np.round(W * 1000) / 1000
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c] for c in range(rank))
                            if abs(val - 1) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    break  # Found rank, try to go lower
    
    # Convert to lists of lists with integer/half-integer entries
    def to_half_int_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                h = round(val * 2) / 2
                new_row.append(h)
            result.append(new_row)
        return result
    
    U_out = to_half_int_matrix(best_U)
    V_out = to_half_int_matrix(best_V)
    W_out = to_half_int_matrix(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
