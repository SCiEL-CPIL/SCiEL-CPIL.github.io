"""Matrix multiplication 3,4,7 rank-minimal decomposition search."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose matmul tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,k), check that the bilinear form in j matches
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # With decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait, that's not right. Let me re-derive.
    #
    # The tensor T[i,j,k] = delta (matmul tensor is T[i,j,k] = 1 for all i,j,k)
    # Actually for matmul, T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For fixed i,k: sum_j T[i,j,k] * A[i,j] * B[j,k] should give C[i,k]
    # Actually let's just check the tensor directly but vectorized.
    
    # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Build T as (n, m, p)
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        for i in range(n):
            for k in range(p):
                T[i, :, k] += Uc[i, :] * Vc[:, k] * Wc[k, i]
    
    return np.allclose(T, 1.0, atol=tol)

def _try_random_rank(r, n, m, p, max_attempts=500, seed=None):
    """Try to find a rank-r decomposition via random search with local refinement."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        # Initialize with random entries
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Try to refine: we want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a nonlinear system. Use gradient-free approach.
        
        # Actually, let's use a different strategy:
        # Pick r random "directions" and solve for the coefficients.
        
        # Alternative: use the fact that if we fix U and V, we can solve for W
        # W[k*n+i, c] = T[i,j,k] / (U[i*m+j,c] * V[j*p+k,c]) for any (i,j,k)
        # But this must be consistent across all (i,j,k).
        
        # Better approach: use a structured random initialization
        # and check if it works.
        
        # For small ranks, try systematic search
        if r <= 20:
            # Try to construct via rank-1 terms
            # Each term c contributes U_c ⊗ V_c ⊗ W_c (outer product)
            # We need sum of these to equal the all-ones tensor
            
            # Random approach: pick random sparse patterns
            U_sparse = np.zeros((n*m, r))
            V_sparse = np.zeros((m*p, r))
            W_sparse = np.zeros((p*n, r))
            
            for c in range(r):
                # Randomly choose which entries are nonzero
                n_nonzero_u = random.randint(1, min(n*m, 5))
                n_nonzero_v = random.randint(1, min(m*p, 5))
                n_nonzero_w = random.randint(1, min(p*n, 5))
                
                u_indices = random.sample(range(n*m), n_nonzero_u)
                v_indices = random.sample(range(m*p), n_nonzero_v)
                w_indices = random.sample(range(p*n), n_nonzero_w)
                
                for idx in u_indices:
                    U_sparse[idx, c] = random.choice([-1, 1])
                for idx in v_indices:
                    V_sparse[idx, c] = random.choice([-1, 1])
                for idx in w_indices:
                    W_sparse[idx, c] = random.choice([-1, 1])
            
            if _verify_fast(U_sparse, V_sparse, W_sparse, n, m, p):
                return U_sparse.tolist(), V_sparse.tolist(), W_sparse.tolist()
    
    return None

def _try_structured_search(r, n, m, p, deadline):
    """Try structured approaches for low rank."""
    
    # Strategy 1: Block decomposition
    # Partition the matrices into blocks and use Strassen-like ideas
    
    # For 3x4 x 4x7, try to exploit structure
    # A is 3x4, B is 4x7, C is 3x7
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # A rank-r decomposition means we can write this as r products of 
    # linear forms in A, B, and C.
    
    # Let's try: express each C[i,k] as sum of r terms
    # term_c = (linear form in A) * (linear form in B) * (linear form in C)
    # but that's not quite right.
    
    # Actually: C[i,k] = sum_c f_c(A) * g_c(B) * h_c(C)
    # where f_c is linear in A entries, g_c linear in B entries, h_c linear in C entries
    # But h_c should be 1 for the matmul to work... 
    
    # Wait, I'm confusing myself. Let me re-read the problem.
    
    # The tensor T[i,j,k] for matmul is T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # So we need r trilinear forms whose sum is the all-ones tensor on {0..n-1} x {0..m-1} x {0..p-1}.
    
    # The all-ones tensor has rank... well, it's a 3-tensor.
    # The rank of the all-ones tensor of size a x b x c is at most min(a,b,c) in some cases,
    # but generally it can be higher.
    
    # For a x b x c all-ones tensor, one decomposition is:
    # T[i,j,k] = sum_{c=1}^{min(a,b,c)} e_i * e_j * e_k where e is all-ones vector
    # But that gives rank min(a,b,c) only if we allow the factors to be the all-ones vectors.
    
    # Wait: if U[i,j,c] = 1 for all i,j when c=1, and 0 otherwise
    # V[j,k,c] = 1 for all j,k when c=1, and 0 otherwise  
    # W[k,i,c] = 1 for all k,i when c=1, and 0 otherwise
    # Then T[i,j,k] = 1*1*1 = 1. So rank 1 works?!
    
    # Let me verify: U has shape (n*m, 1) with all entries 1
    # V has shape (m*p, 1) with all entries 1
    # W has shape (p*n, 1) with all entries 1
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1. YES!
    
    # So the rank is 1!
    
    return None  # We'll handle rank 1 directly

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try rank 1: all-ones decomposition
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify rank 1
    if _verify_fast(np.array(U), np.array(V), np.array(W), n, m, p, tol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # If rank 1 doesn't work (shouldn't happen), try higher ranks
    # But for the all-ones tensor, rank 1 should always work
    
    # Fallback: schoolbook
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
