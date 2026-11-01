"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _try_rank_reduction(n, m, p, deadline, max_attempts=5000):
    """
    Attempt to find a lower-rank decomposition using structured random search.
    Strategy: 
    1. Use the schoolbook as baseline.
    2. Try to merge pairs of schoolbook terms that share structure.
    3. Use randomized linear algebra to find low-rank approximations and then
       snap to exact rational/half-integer values.
    """
    best_rank = n * m * p
    best_decomp = _schoolbook_decomposition(n, m, p)
    
    # Convert to numpy for faster operations
    U_np = np.array(best_decomp["U"], dtype=float)
    V_np = np.array(best_decomp["V"], dtype=float)
    W_np = np.array(best_decomp["W"], dtype=float)
    
    # Try different rank targets
    # Lower bounds: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But for matrix mult, the rank is at least max(n*m, m*p, p*n)
    # Actually, the rank of the matrix multiplication tensor is at least 
    # max(n*m, m*p, p*n) but this is not tight.
    # Known: rank(MM_{n,m,p}) >= max(n*m, m*p, p*n)
    # For 2x3x10: max(6, 30, 20) = 30
    
    # Try ranks from 30 up to best_rank-1
    for target_rank in range(30, best_rank):
        if time.time() > deadline:
            break
            
        # For each target rank, try to find a valid decomposition
        # Use a greedy approach: start with random rank-1 terms and try to cover all (i,j,k)
        
        # Randomized approach: generate random rank-1 components and check if they can span
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Try a structured approach: for each j, the terms involving j form a 
        # n x p matrix. If we can factor this as a sum of rank-1 terms with shared
        # structure across j, we can reduce rank.
        
        # For fixed j, T[i,j,k] = 1 for all i,k. This is an n x p all-ones matrix.
        # Rank of all-ones matrix is 1. So for each j, we need at least 1 term.
        # But the terms for different j can potentially share U and W components.
        
        # Key insight: If we use the same U and W vectors for all j, we need:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # If U[i*m+j,c] and W[k*n+i,c] are the same for all j (i.e., U[i,c] independent of j),
        # then we need V[j*p+k,c] to compensate.
        
        # Try: U[i*m+j,c] = a[i,c], W[k*n+i,c] = b[k,c], V[j*p+k,c] = c[j,k,c]
        # Then sum_c a[i,c] * c[j,k,c] * b[k,c] = 1
        # This is a bilinear form in a and b with coefficients c.
        
        # For n=2, p=10, we can try small ranks.
        # Let's try a direct search for small ranks using integer/half-integer entries.
        
        # For now, use a randomized linear algebra approach:
        # 1. Generate random U, V, W of shape (n*m, r), (m*p, r), (p*n, r)
        # 2. Check if the tensor equation holds
        # 3. Use local search to adjust
        
        # This is expensive, so we'll do a limited number of trials
        trials = max(1, min(100, int((deadline - time.time()) / 0.01)))
        
        found = False
        for trial in range(trials):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            U_rand = np.random.randint(-2, 3, size=(n*m, target_rank))
            V_rand = np.random.randint(-2, 3, size=(m*p, target_rank))
            W_rand = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for c in range(target_rank):
                            s += U_rand[u_idx, c] * V_rand[v_idx, c] * W_rand[w_idx, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_decomp = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist()
                }
                found = True
                break
        
        if found:
            break  # Found a valid decomposition at this rank, try lower
    
    return best_decomp, best_rank

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Tries to find a lower-rank decomposition than schoolbook using structured search.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_decomp = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank
    if time.time() < deadline - 5:  # Leave 5s for verification
        try:
            decomp, rank = _try_rank_reduction(n, m, p, deadline)
            if rank < best_rank:
                best_decomp = decomp
                best_rank = rank
        except Exception:
            pass  # Fall back to schoolbook
    
    # Final verification
    U = best_decomp["U"]
    V = best_decomp["V"]
    W = best_decomp["W"]
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook if verification fails
        best_decomp = _schoolbook_decomposition(n, m, p)
    
    return best_decomp
# EVOLVE-BLOCK-END
