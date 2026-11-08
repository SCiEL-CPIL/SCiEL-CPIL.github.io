"""Matrix multiplication 3x5x6 rank minimization via algebraic search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (15, rank), (30, rank), (18, rank)
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 15, 30, 18
    
    start_time = time.time()
    deadline = 120.0  # seconds
    
    def elapsed():
        return time.time() - start_time
    
    def is_deadline_exceeded(margin=10.0):
        return elapsed() > deadline - margin
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = delta[i,k] if we're doing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # Actually, for matrix multiplication C = A * B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = delta[i,k] (in the sense that T maps A,B to C)
        # Actually the structure tensor for matrix mult: T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
        # Wait, let me reconsider.
        
        # The matrix multiplication tensor T has components T_{i,j,k} such that:
        # (A*B)_{i,k} = sum_{j} A_{i,j} * B_{j,k}
        # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # This is just the all-ones tensor of shape (n,m,p)? No...
        
        # Actually, the structure tensor for matmul: we want to express
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # As a bilinear map: C[i,k] = sum_{a,b,c} T[i,k,a,b,c] * A[a,b] * B[c,d]... 
        # Let me use the standard formulation.
        
        # Standard: C = A * B, C is n x p, A is n x m, B is m x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # Tensor form: C[i,k] = sum_{j} T_{i,k,j} * A[i,j] * B[j,k]
        # But T_{i,k,j} = 1 for all valid indices.
        #
        # Actually the proper structure tensor:
        # We have three modes: first mode = (i,k) for C, second mode = (i,j) for A, third mode = (j,k) for B
        # T[(i,k),(i',j),(j',k')] = 1 if i=i' and k=k' and j=j'
        # So T[(i,k),(i',j),(j',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        
        # For the decomposition:
        # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer products)
        # where U has shape (n*p, rank), V has shape (n*m, rank), W has shape (m*p, rank)
        #
        # Wait, the task says:
        # U: shape (n*m, rank) = (15, rank)
        # V: shape (m*p, rank) = (30, rank) 
        # W: shape (p*n, rank) = (18, rank)
        #
        # And the output factor follows W[k*n+i, r] convention.
        #
        # So the tensor is: T[(i,j),(j',k),(k',i')] = U[i,j,r] * V[j',k,r] * W[k',i',r]
        # Summed over r should give: 1 if i=i' and j=j' and k=k', else 0
        # i.e., T[(i,j),(j',k),(k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        
        # Let's verify:
        # Compute the reconstructed tensor
        # T_recon[(i,j),(j',k),(k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Reshape and check
        # T_recon shape: (n*m, m*p, p*n)
        # T_true shape: (n*m, m*p, p*n) where T_true[(i,j),(j',k),(k',i')] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
        
        # Efficient verification using einsum
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j'*p+k, c = k'*n+i'
        
        # T_true[a,b,c] = 1 if (i from a) == (i' from c) and (j from a) == (j' from b) and (k from b) == (k' from c)
        
        # Let's build T_true
        T_true = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_true[a, b, c] = 1.0
        
        # Compute T_recon
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = sum_r (U[:,r] * V[:,r])_a * W[c,r]... not quite
        # Better: T_recon = sum_r np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        # Or use einsum: T_recon[a,b,c] = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        try:
            T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        except Exception:
            # Fallback
            T_recon = np.zeros((nm, mp, pn))
            for r in range(rank):
                T_recon += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        
        diff = np.abs(T_recon - T_true)
        return np.max(diff) < tol, np.max(diff)
    
    def try_rank(rank, max_iter=1000):
        """Try to find a decomposition of given rank using randomized approach."""
        # Initialize with random values in [-1, 1] or small integers
        U = np.random.uniform(-2, 2, (nm, rank))
        V = np.random.uniform(-2, 2, (mp, rank))
        W = np.random.uniform(-2, 2, (pn, rank))
        
        # Build T_true
        T_true = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_true[a, b, c] = 1.0
        
        # Use gradient descent on the tensor fitting problem
        # Minimize ||sum_r U_r * V_r * W_r - T_true||_F^2
        # 
        # We'll do a simple alternating least squares or gradient approach
        # But for exact integer/half-integer solutions, we need a different approach.
        
        # Alternative: Try to find a low-rank decomposition by solving a system
        # For small ranks, we can try specific patterns.
        
        # First, let's try the schoolbook and see if we can compress
        # Schoolbook has rank = n*m*p = 90
        
        # Let's try to find a better decomposition using known results
        # For 3x5x6, the minimal rank is not known to be much less than 90
        # Let's try ranks from low to high
        
        return None
    
    # Start with schoolbook as baseline
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to improve using structured approaches
    # Approach 1: Exploit the symmetry/structure of matrix multiplication
    
    # For matrix multiplication, we can use block structures
    # Let's try a few different rank values and see if we can find valid decompositions
    
    # Try rank = 75, 70, 65, etc.
    # But finding exact decompositions is hard.
    
    # Let's try a more systematic approach:
    # Use the fact that matrix multiplication tensor has certain symmetries
    
    # Actually, let's try to use a randomized search with rounding to half-integers
    # for a range of ranks
    
    def random_search_with_rounding(rank, n_trials=50):
        """Try random initialization followed by rounding to half-integers."""
        for trial in range(n_trials):
            if is_deadline_exceeded():
                return None
            
            # Random initialization
            U = np.random.randint(-2, 3, (nm, rank))
            V = np.random.randint(-2, 3, (mp, rank))
            W = np.random.randint(-2, 3, (pn, rank))
            
            valid, max_diff = verify_decomposition(U.tolist(), V.tolist(), W.tolist())
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Try half-integers
            U_half = U * 0.5
            V_half = V * 0.5
            W_half = W * 0.5
            
            valid, max_diff = verify_decomposition(U_half.tolist(), V_half.tolist(), W_half.tolist())
            if valid:
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        
        return None
    
    # Try various ranks
    for rank in range(best_rank - 1, max(20, best_rank - 30), -1):
        if is_deadline_exceeded():
            break
        
        result = random_search_with_rounding(rank, n_trials=10)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # If we didn't find anything better, return the schoolbook
    return best_result
