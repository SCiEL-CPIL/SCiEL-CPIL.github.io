"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Build the multiplication tensor T[i,j,k] = 1 if this is a valid term
    # T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # The tensor has rank at most n*m*p (schoolbook)
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # and = 0 for all other combinations (but there are no other combinations
    # since every (i,j,k) triple corresponds to a matrix entry)
    
    # Actually, for matrix multiplication, the tensor is:
    # T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) triples
    # The rank is the minimum r such that T can be written as sum of r rank-1 tensors
    
    # Schoolbook: rank = n*m*p = 60
    # We want to find lower rank decompositions
    
    # Strategy: Try random structured decompositions at various ranks
    # Start with known constructions and search for improvements
    
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known bounds: for 2x5x6, the rank is between some lower and upper bound
    # Let's search systematically
    
    # First, try some structured approaches
    for rank in range(14, best_rank):  # Start from literature-known lower bound
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_rank = rank
            best_result = result
            # If we found a valid one, try to go lower
            if rank <= 14:
                break
    
    return best_result

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of matrix multiplication."""
    rank = len(U[0])
    # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using random search."""
    if time.time() > deadline:
        return None
    
    # For matrix multiplication, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # This is a 3-way tensor factorization problem.
    # Use a structured approach: initialize with random values and use gradient-free search
    
    max_iters = 200
    np.random.seed(42 + rank)
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Initialize with random half-integer values
        U = np.random.randint(0, 4, size=(n*m, rank)) * 0.5
        V = np.random.randint(0, 4, size=(m*p, rank)) * 0.5
        W = np.random.randint(0, 4, size=(p*n, rank)) * 0.5
        
        # Scale to help with convergence
        U = U / np.sqrt(rank) * np.sqrt(n*m)
        V = V / np.sqrt(rank) * np.sqrt(m*p)
        W = W / np.sqrt(rank) * np.sqrt(p*n)
        
        # Use iterative refinement: for each (i,j,k), adjust the columns
        # to better match the target value of 1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current error
            max_err = 0.0
            worst_ijk = None
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
                            worst_ijk = (i, j, k)
            
            if max_err < 1e-7:
                # Found a solution!
                # Convert to list of lists with rounded values
                U_list = [[float(round(U[i, c] * 2)) / 2 for c in range(rank)] for i in range(n*m)]
                V_list = [[float(round(V[i, c] * 2)) / 2 for c in range(rank)] for i in range(m*p)]
                W_list = [[float(round(W[i, c] * 2)) / 2 for c in range(rank)] for i in range(p*n)]
                
                # Verify the rounded version
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # Use unrounded if it works
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            
            # Gradient-free update: perturb the worst entry
            i, j, k = worst_ijk
            c = random.randint(0, rank - 1)
            step = 0.1 * random.choice([-1, 1])
            U[i*m+j, c] += step
            V[j*p+k, c] += step
            W[k*n+i, c] += step
        
        # After each attempt, check if we got close
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(s - 1.0)
                    max_err = max(max_err, err)
        
        if max_err < 1e-5:
            # Close enough, try to polish
            U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
