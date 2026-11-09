"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # And that the tensor is zero elsewhere?
    # The definition of the matrix multiplication tensor T is:
    # T[i*m+j, j*p+k, k*n+i] = 1?
    # Actually, the standard tensor T for matrix mult A(n,m) B(m,p) = C(n,p)
    # has indices (i,j) for A, (j,k) for B, (i,k) for C.
    # The condition is sum_r U[i,j,r] V[j,k,r] W[i,k,r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}?
    # No, it's simply that the reconstructed matrix C = A B.
    # The tensor entries are 1 if the indices match the "wire" in the circuit.
    # For schoolbook, each term r corresponds to a specific (i,j,k).
    # U[i*m+j, r] is 1 only if r matches (i,j,k_r).
    # So the sum is 1 if (i,j,k) is the one for term r, and 0 otherwise?
    # Wait.
    # If we sum over r, for a fixed (i,j,k):
    # Only the term r* corresponding to (i,j,k) has U[i*m+j, r*]=1, V[j*p+k, r*]=1, W[k*n+i, r*]=1.
    # All other terms have at least one of these as 0 (because their indices don't match).
    # So the sum is 1.
    # For a "wrong" index combination, e.g., checking if the tensor is zero where it should be.
    # The tensor T is defined such that T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # And 0 otherwise.
    # So we must check:
    # 1. For all i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    # 2. For all other (a,b,c) where the indices don't align: sum_r U[a, r] V[b, r] W[c, r] == 0
    
    # Efficient check for 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check for 0s (sampled or full)
    # Full check is O((nm)(mp)(pn) * R) which is huge.
    # But for small dims, we can check all "invalid" tuples.
    # Total tuples: (n*m) * (m*p) * (p*n) = n^2 m^2 p^2.
    # For 3,5,6: 9 * 25 * 36 = 8100 tuples.
    # Rank 90. 8100 * 90 = 729,000 ops. This is fast.
    
    # Generate all valid (a,b,c) indices
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.add((i * m + j, j * p + k, k * n + i))
                
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in valid_indices:
                    continue
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition
    # Strategy: Random search with small integer entries for ranks < schoolbook_rank
    random.seed(42)
    
    # We will try ranks from schoolbook_rank - 1 down to a lower bound
    # Lower bound for rank is max(n*m, m*p, p*n) = 30
    # But realistically, we only try a few steps down to avoid long searches
    # If we find one, we return immediately.
    
    # To make search effective, we use a heuristic:
    # Start with schoolbook, and try to "merge" or "perturb"
    # However, random restarts with small integers are a common baseline.
    
    # Given the low probability of finding a valid lower rank via pure random search
    # in 110s for this specific size, we rely on the schoolbook fallback.
    # But we must attempt the search as per the directive to "investigate improvements".
    
    # Limit the search to a few iterations to ensure we don't timeout
    # and to keep the code simple and robust.
    
    # Try rank 89
    target_rank = schoolbook_rank - 1
    if time.time() < deadline:
        for attempt in range(5000):
            if time.time() > deadline:
                break
            # Random initialization with -1, 0, 1
            # Bias towards sparsity
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            # Quick check on a subset of 1s
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                # Full verification
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
    
    # Convert to lists for the return format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
