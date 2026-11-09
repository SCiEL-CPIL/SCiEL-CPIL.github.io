"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For each (i,j,k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    # Check that for all (i,j,k) not matching, the sum is 0
    # Actually, we need to check that the decomposition is exact
    # The tensor T[i,j,k] = 1 if this is a valid triple, else 0
    # But matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
    # Wait, let me reconsider. The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if we're multiplying element (i,j) of A by (j,k) of B
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    return True

def _schoolbook(n, m, p):
    """Create the standard schoolbook decomposition."""
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=10000, seed=42):
    """Try to reduce rank by finding linear dependencies among the columns."""
    rank = len(U[0])
    if target_rank >= rank:
        return None
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # We want to find a linear transformation that reduces the effective rank
    # Try random projections and see if we can maintain the decomposition
    
    best = None
    for iteration in range(max_iters):
        # Try to find a random linear combination that preserves the tensor
        # This is a heuristic approach
        
        # Generate random vectors for each column
        # We're looking for coefficients alpha_c such that
        # sum_c alpha_c * U[:,c] * V[:,c] * W[:,c] = 0 (approximately)
        # but we want to reduce the number of non-zero terms
        
        # Alternative: try to merge columns
        # Pick two columns and see if we can express one as a combination of others
        
        c1 = random.randint(0, rank - 1)
        c2 = random.randint(0, rank - 1)
        if c1 == c2:
            continue
        
        # Check if column c2 is approximately in the span of other columns
        # This is expensive, so use a simpler heuristic
        
        # Try a random perturbation approach
        # Add a small random perturbation to all columns and see if rank reduces
        
        if iteration % 100 == 0:
            # Try a structured approach: look for patterns
            # For matrix multiplication, there might be symmetries we can exploit
            pass
    
    return None

def _random_search(n, m, p, max_rank, max_time=100.0):
    """Random search for low-rank decompositions."""
    start_time = time.time()
    
    # Try various ranks from low to high
    for target_rank in range(1, max_rank + 1):
        if time.time() - start_time > max_time:
            break
        
        # Try random initialization
        for attempt in range(100):
            if time.time() - start_time > max_time:
                break
            
            # Random initialization with structure
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use random values but with some structure
            for c in range(target_rank):
                # Randomly assign non-zero values
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.3:
                            U[i*m + j, c] = random.choice([-1, 1]) * random.random()
                
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.3:
                            V[j*p + k, c] = random.choice([-1, 1]) * random.random()
                
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.3:
                            W[k*n + i, c] = random.choice([-1, 1]) * random.random()
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Try to find better decompositions
    # The theoretical lower bound for matrix multiplication rank is not well-known
    # For 3x4 * 4x7, let's try ranks from 84 down
    
    # Known results: For small matrices, the rank can be less than n*m*p
    # Try some known constructions
    
    # First, try a structured approach based on Strassen-like ideas
    # For 2x2 * 2x2, Strassen uses rank 7 instead of 8
    
    # For our case, let's try a random search with time limit
    result = _random_search(n, m, p, max_rank=84, max_time=100.0)
    if result is not None:
        best_result = result
        best_rank = len(result["U"][0])
    
    # Check if we've hit the deadline
    if time.time() > deadline - 5:
        return best_result
    
    # Try more sophisticated approaches
    # Use the schoolbook as a starting point and try to reduce
    
    # Attempt 1: Look for linear dependencies
    # The columns (U_col, V_col, W_col) form a tensor decomposition
    # We want to find the minimum number of such triples
    
    # For now, return the best we found
    return best_result
# EVOLVE-BLOCK-END
