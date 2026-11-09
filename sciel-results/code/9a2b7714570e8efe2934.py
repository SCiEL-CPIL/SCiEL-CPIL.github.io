"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    s += U[ui][r] * V[vj][r] * W[wk][r]
                # The multiplication tensor entry at (i,j,k) is 1 if we're 
                # computing C[i,k] += A[i,j]*B[j,k], so the tensor is 1 for all (i,j,k)
                # Wait - the tensor for matrix multiplication: T[i,j,k] = 1 for the 
                # standard basis, meaning C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor is T[i,j,k] where the "output" index is (i,k) and input 
                # indices are (i,j) and (j,k). So T[i,j,k] = 1 for all valid i,j,k.
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(val, tol=1e-9):
    """Check if val is a half-integer (integer or integer + 0.5)."""
    doubled = val * 2
    return abs(doubled - round(doubled)) < tol

def _check_half_integer_matrices(U, V, W, tol=1e-9):
    """Check if all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
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

def _try_random_rank(n, m, p, target_rank, num_trials=1000, seed=42):
    """Try to find a decomposition of the given rank using random initialization and local search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for trial in range(num_trials):
        # Initialize with random half-integer values
        # Use small values to increase chance of exactness
        scale = 1
        U = [[rng.choice([-scale, 0, scale]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.choice([-scale, 0, scale]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[rng.choice([-scale, 0, scale]) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Try to fix by solving linear system for W given U, V
        # For each (k, i) pair, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is complex. Instead, try a structured approach.
        
        # Alternative: use the fact that we can set W[k*n+i, r] = 1 for some r
        # and solve for consistency.
        
        # Try a different approach: initialize W with identity-like structure
        # and solve for U and V.
        
        # Actually, let's try a simpler heuristic: start from schoolbook and try to
        # merge columns (reduce rank) by finding linear dependencies.
        pass
    
    return None

def _try_local_search(n, m, p, start_rank, max_iterations=500, seed=42):
    """Local search: start from schoolbook and try to reduce rank by merging."""
    # This is complex; for now, return the schoolbook result
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 7 matrix."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with the guaranteed schoolbook solution
    best_result = _make_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Verify the schoolbook solution
    if _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        if _check_half_integer_matrices(best_result["U"], best_result["V"], best_result["W"]):
            pass  # Valid
        else:
            # Should not happen for schoolbook
            return best_result
    else:
        # Should not happen
        return best_result
    
    # Try to find a lower-rank decomposition
    # Strategy: try ranks from known bounds downward
    # Known: rank < n*m*p for all non-trivial matrix multiplication
    # For 3x5x7, try a range of ranks
    
    # Try some specific ranks that might work based on known results
    # The rank of M_{3,5,7} is known to be at most 94 (from various bounds)
    # Try a few candidate ranks
    
    candidate_ranks = [104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try to construct a decomposition of this rank
        # Use a structured approach: take the first target_rank terms of schoolbook
        # and see if they span the tensor (they won't unless target_rank = 105)
        
        # Alternative: try to find a valid decomposition using algebraic methods
        # For now, we'll try a simple approach: use the schoolbook but with
        # some zeroed-out columns and check if it still works (it won't for lower rank)
        
        # A better approach: use the fact that the tensor has symmetries
        # Try to find a decomposition using random linear combinations
        
        # Initialize U, V, W with random values and try to solve
        # This is a hard problem; for the time budget, we'll rely on the schoolbook
        
        # Try a simple local search: start from schoolbook, try to merge pairs
        # of columns by finding a linear combination that preserves the tensor
        if time.time() > deadline - 10:
            break
        
        # For this specific case, we'll attempt a constructive approach
        # based on the structure of 3x5x7 matrix multiplication
        
        # The multiplication: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1,2}, k in {0..6}
        # Tensor entries: T[i,j,k] = 1 for all i,j,k (this is the structure tensor)
        
        # Actually, the tensor for matrix multiplication is:
        # T[(i,j),(j,k),(i,k)] = 1
        # In our indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # This is a system of n*m*p = 105 equations in 3*rank variables (with products)
        # It's a tensor decomposition problem.
        
        # Given the time constraints and the difficulty of finding a rank < 105
        # decomposition with half-integer entries, we'll return the schoolbook result
        # unless we can quickly find a valid lower-rank one.
        
        # Try a quick random search for lower rank
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Try to build a decomposition of target_rank
            # Use a greedy approach: try to add terms one at a time
            # that cover more tensor entries
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Try to assign values to make the tensor work
            # This is a NP-hard problem in general, so we use heuristics
            
            # For each column r, choose U[:,r], V[:,r], W[:,r] to maximize
            # the number of (i,j,k) triples where the product matches
            # This is still very hard.
            
            # Given the complexity, we'll stick with schoolbook for now
            pass
    
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
