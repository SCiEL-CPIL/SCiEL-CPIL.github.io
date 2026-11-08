"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin within 150s limit
    
    n, m, p = DIMENSIONS
    
    # Matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Trace convention: W[k*n + i, r]
    
    # Schoolbook decomposition: rank = n*m*p
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is valid (it should be)
    # For each (i,j,k), the contribution to C[i,k] from the tensor is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # In schoolbook, each (i,j,k) has exactly one r where all three are 1
    
    # Try to find a better decomposition using random search
    # The tensor T[i,j,k] = 1 can be decomposed as sum_r u_{i,j}^{(r)} v_{j,k}^{(r)} w_{k,i}^{(r)}
    # This is equivalent to finding a low-rank decomposition of the 3D tensor
    
    # One approach: use the fact that T[i,j,k] = 1 = (1)(1)(1) but with the constraint
    # that u depends on (i,j), v on (j,k), w on (k,i)
    
    # Let's try rank r with random initialization and see if we can solve
    # Actually, let's think about this more carefully.
    
    # The bilinear complexity (tensor rank) of matrix multiplication C = A*B where A is n x m, B is m x p
    # For n=2, m=3, p=8, this is a specific case.
    
    # Known lower bounds: rank >= max(n, p) * something, but more precisely:
    # The tensor rank of n x m * m x p multiplication is at least max(n, m, p) in some cases,
    # but actually the lower bound is more subtle.
    
    # Let's try a few candidate ranks and use a simple verification approach.
    # For a given rank r, we need to find U, V, W such that for all i,j,k:
    # sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of equations. Let's try to solve it for small r.
    
    # Actually, a simpler approach: use the structure of the problem.
    # Note that T[i,j,k] = 1 for all i,j,k.
    # If we can find rank r such that the tensor can be written as sum of r rank-1 tensors
    # with the specific index structure, we're done.
    
    # Let's try a greedy/iterative approach:
    # Start with r = some guess, randomly initialize, and try to fit.
    
    # First, let's establish a baseline with schoolbook and check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return schoolbook as list of lists
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    
    # Try to find a lower rank using random restarts
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n, p) = 8? Actually, the rank must be at least the number of distinct "rows" in some sense.
    # A known lower bound is that the tensor rank is at least max(n, m, p) for general matrices, but this isn't tight.
    # Let's try ranks: 8, 9, 10, ..., up to min(rank_schoolbook, some limit)
    
    # For each candidate rank, try random initialization and verify
    # Verification: check that for all i,j,k, the sum equals 1 (within tolerance)
    
    # To make this efficient, let's precompute and use vectorized operations
    
    # For a given U, V, W of rank r:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need C[i,k] = 1 for all i,k, and also the individual terms must be consistent.
    # Actually, the tensor condition is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is n*m*p equations.
    
    # Let's try a different approach: since T[i,j,k] = 1, we can use the fact that
    # 1 = sum_{a=1}^{n} delta_{ia} * sum_{b=1}^{p} delta_{kb} * 1 / (n*p) ... no, that doesn't work directly.
    
    # Actually, let's just do a numerical search for small ranks.
    
    # Candidate ranks to try: start from a lower bound
    # The tensor rank of n x m * m x p is known to be at most n*m*p and at least... 
    # For 2x3 * 3x8, let's try ranks from 8 to 20 or so.
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is valid."""
        nrm = n * m
        mpp = m * p
        pnn = p * n
        rank = U.shape[1]
        
        # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try random searches for lower ranks
    np.random.seed(42)
    
    for target_rank in range(8, min(rank_schoolbook + 1, 30)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for each rank
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Quick check: compute all (i,j,k) sums
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i * m + j, :]
                        v_val = V[j * p + k, :]
                        w_val = W[k * n + i, :]
                        s = np.dot(u_val, v_val * w_val)
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Found a valid decomposition!
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
        
        if best_rank < rank_schoolbook:
            break
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return what we have
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
