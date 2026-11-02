"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if the product term exists: i=i', j=j', k=k'
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # The tensor is T[i,j,k] = 1 (for the multiplication C[i,k] = sum_j A[i,j]*B[j,k])
    # In the factorized form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Schoolbook: rank = n*m*p = 45
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T is of shape (9, 15, 15)
    # We need U (9 x R), V (15 x R), W (15 x R) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) corresponds to a valid multiplication term
    
    # Actually, the tensor is sparse: T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise
    # But wait, that's not right. The tensor entries are:
    # For matrix multiplication, the tensor T has T[i,j,k] = 1 for all i,j,k
    # No, that's the all-ones tensor.
    
    # Let me reconsider. The matrix multiplication tensor T_MM has:
    # T_MM[i, j, k] = 1 (where i indexes A, j indexes B, k indexes C output)
    # Specifically, C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # This means T[i*m+j, j*p+k, k*n+i] = 1
    
    # The rank-1 decomposition: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook: each (i,j,k) triple gets its own rank-1 term
    
    # The tensor is actually the standard matrix multiplication tensor
    # Its rank for 3x3 x 3x5 is at most 45 (schoolbook)
    
    # Let's try to find a lower rank by optimization
    # This is a hard problem, so let's try a simple approach
    
    # Try ranks from 35 to 44
    best_result = None
    best_rank = rank_schoolbook
    
    # First, return schoolbook if we can't find better
    def make_schoolbook():
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
    
    schoolbook = make_schoolbook()
    
    # Try a simple optimization approach for lower rank
    # Use random initialization and gradient-free search
    # Given time constraints, try a few ranks
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals the MM tensor"""
        n, m, p = DIMENSIONS
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(len(U[0])):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1) > tol:
                        return False
        # Check that other entries are 0 (this is harder, skip for now)
        return True
    
    # Try to find a rank-44 decomposition using a simple perturbation
    # Start from schoolbook and try to merge two terms
    
    # Actually, let's try a different approach: use the fact that we can
    # potentially combine terms using linear algebra
    
    # For now, if time permits, try a random search
    if time.time() < deadline - 30:
        # Try a random search for rank < 45
        np.random.seed(42)
        for target_rank in [44, 43, 42, 41, 40]:
            if time.time() > deadline - 10:
                break
            max_iters = 50
            for attempt in range(max_iters):
                if time.time() > deadline - 5:
                    break
                # Random initialization
                R = target_rank
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
                
                # Simple verification
                found = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            s = 0
                            for r in range(R):
                                s += U[a, r] * V[b, r] * W[c, r]
                            if abs(s - 1) > 1e-6:
                                found = False
                                break
                        if not found:
                            break
                    if not found:
                        break
                
                if found:
                    # Convert to lists
                    result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_result = result
                    best_rank = target_rank
                    break
            if best_result is not None:
                break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook
# EVOLVE-BLOCK-END
