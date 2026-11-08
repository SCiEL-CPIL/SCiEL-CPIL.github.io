"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        For each r: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] should give the mult tensor.
        The tensor T[i,j,k] = 1 if and only if it's a valid (i,j,k) triple.
        Actually, the mult tensor has T[i,j,k] = 1 for all valid indices.
        We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Build the tensor
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions using random search
    # Work with integer entries scaled by 2 (so half-integers become integers)
    def random_search(max_rank, num_trials, time_limit):
        """Try random rank-r decompositions with small integer entries."""
        start = time.time()
        for trial in range(num_trials):
            if time.time() > start + time_limit:
                break
            if time.time() > deadline:
                break
            
            rank = max_rank
            # Generate random U, V, W with small integer entries
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for r in range(rank):
                # Random sparse vectors with small entries
                nnz = random.randint(1, 3)
                idx = random.sample(range(n * m), min(nnz, n * m))
                for idx_i in idx:
                    U[idx_i, r] = random.choice([-2, -1, 1, 2])
                
                nnz = random.randint(1, 3)
                idx = random.sample(range(m * p), min(nnz, m * p))
                for idx_j in idx:
                    V[idx_j, r] = random.choice([-2, -1, 1, 2])
                
                nnz = random.randint(1, 3)
                idx = random.sample(range(p * n), min(nnz, p * n))
                for idx_k in idx:
                    W[idx_k, r] = random.choice([-2, -1, 1, 2])
            
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        return None
    
    # Try structured approach: exploit the 3x5x5 structure
    # Idea: For each output position (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # We can group by i or k to share factors
    
    # Approach: Try rank = n*m + m*p + p*n - something
    # Or try known lower bounds
    
    # Let's try a systematic reduction from schoolbook
    # Start with schoolbook and try to find linear dependencies
    
    def try_reduce_from_schoolbook(start_rank, max_reductions):
        """Try to reduce rank by finding linear combinations."""
        nonlocal best, best_rank
        
        current_rank = start_rank
        # Build initial schoolbook
        U = np.zeros((n * m, current_rank))
        V = np.zeros((m * p, current_rank))
        W = np.zeros((p * n, current_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Try to find if any column is a linear combination of others
        # This is complex; instead try random perturbations
        
        return None
    
    # Try specific known structures
    # For 3x5x5, try rank around 60-70
    
    # Method: Use the fact that we can write the multiplication as
    # C = A * B, and try to decompose using intermediate sums
    
    # Let's try a different approach: use bilinear complexity results
    # The rank of M_{n,m,p} is at most n*m*p, and for small cases can be lower
    
    # Try random search for various ranks
    for target_rank in range(70, 75):
        if time.time() > deadline - 10:
            break
        result = random_search(target_rank, 500, 20)
        if result is not None:
            U, V, W = result
            # Convert to integer/half-integer format
            U_out = [[float(x)/2 for x in row] for row in U]
            V_out = [[float(x)/2 for x in row] for row in V]
            W_out = [[float(x)/2 for x in row] for row in W]
            if target_rank < best_rank:
                best = {"U": U_out, "V": V_out, "W": W_out}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
