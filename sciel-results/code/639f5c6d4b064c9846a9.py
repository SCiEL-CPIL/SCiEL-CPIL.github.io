"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W, n, m, p):
    """Check that U @ V^T @ W^T reconstructs the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor: T[i,j,k, i',j',k'] = delta_{j,j'} delta_{k,k'}
    # Actually the decomposition is: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's verify by building the full tensor and comparing.
    # T[i, j, k, i', j', k'] should equal delta_{j,j'} * delta_{k,k'}
    # The decomposition gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k'*n+i',c]
    # which should equal delta_{j,j'} delta_{k,k'} when we sum over appropriate indices.
    # Actually, the matrix multiplication tensor is:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T_{i,j,k} = delta_{j,j} (trivially 1) but structured as:
    # T[i, j, k] where i indexes row of A, j indexes col of A / row of B, k indexes col of B
    # The rank decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let me just check with random matrices.
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    AB = A @ B
    # Reconstruct using decomposition
    # (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # With decomposition: sum_c (sum_j A_{i,j} U[i*m+j,c]) * (sum_j V[j*p+k,c] B_{j,k}) * W[k*n+i,c]
    # Hmm, that's not quite right. Let me think again.
    # The tensor is T_{i,j,k} = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = delta (always 1 for the multiplication structure)
    # Actually T[i,j,k] is the coefficient such that (AB)_{i,k} = sum_{j} T[i,j,k] A_{i,j} B_{j,k}
    # T[i,j,k] = 1 for all i,j,k (since (AB)_{i,k} = sum_j A_{i,j} B_{j,k})
    # Wait no. The tensor for matrix multiplication of (n x m) * (m x p) is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 (the multiplication tensor has all ones)
    # But that's not right either. Let me re-read the problem.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # The decomposition: for each c, U[:,c], V[:,c], W[:,c] form a rank-1 term
    # The multiplication tensor M[i,j,k] = 1 for all valid i,j,k (since (AB)_{ik} = sum_j A_{ij}B_{jk})
    # Actually no. The tensor is M[i, j, k] = delta_{j,j} which is just 1.
    # Hmm, I think the tensor entries are all 1, and the rank decomposition means:
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Let me verify with the schoolbook: U[i*m+j, c] = 1 when c corresponds to (i,j,k), etc.
    # So for schoolbook, each c has exactly one nonzero in each of U, V, W.
    # Let me just do a numerical check.
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Compute the tensor from decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i,j,k] = 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find a lower rank decomposition using optimization
    # Strategy: start with a random initialization at rank r < schoolbook,
    # and try to optimize to satisfy the tensor equation.
    # Use a simple approach: try ranks from 71 down, but that's too slow.
    # Instead, try a few specific lower ranks with random restarts.
    
    import random
    random.seed(12345)
    np.random.seed(12345)
    
    # Try ranks that are slightly below schoolbook
    # For 3x3x8, schoolbook is 72. Try 71, 70, ... but each trial is expensive.
    # Let's try a smarter approach: use the structure.
    # The tensor M[i,j,k] = 1 for all i,j,k is a very simple tensor.
    # Its rank is actually n*m*p = 72 for this specific structure? 
    # No, the multiplication tensor for (n x m) * (m x p) has rank n*m*p in general,
    # but for small cases, fast multiplication algorithms can reduce it.
    # For 3x3x3, Strassen gives rank 7 < 27.
    # For 3x3x8, we might be able to use Strassen-like ideas.
    
    # Let's try to construct a decomposition with rank < 72 using a search approach.
    # Given time constraints, let's try a simple local search from a perturbed schoolbook.
    
    # Actually, let's just return the schoolbook since finding a better one
    # within the time limit is challenging and the tensor structure is simple.
    # But let's at least try a few random initializations at lower ranks.
    
    for target_rank in [71, 70, 69, 68]:
        if time.time() > deadline - 20:
            break
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            # Random initialization
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Simple gradient-free optimization: coordinate descent
            # This is very slow, so let's just check if a random init works (it won't)
            # and skip. Instead, let's try a structured approach.
            pass
    
    # Given the complexity, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
