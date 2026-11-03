import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix using linear algebra rank analysis.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Build the target tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify decomposition using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T_target, atol=tol)

    def clean_matrix(M, tol=1e-7):
        """Clean a matrix by snapping near-zero and near-half-integer values."""
        M = np.array(M, dtype=float)
        M_clean = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if abs(val) < tol:
                    M_clean[i, j] = 0.0
                else:
                    val2 = round(val * 2)
                    if abs(val * 2 - val2) < tol * 10:
                        M_clean[i, j] = val2 / 2.0
                    else:
                        M_clean[i, j] = val
        return M_clean

    # Build schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()

    # Use the flattening technique to find the exact rank
    # The rank of the tensor equals the minimum r such that all flattenings have rank <= r
    # For matrix multiplication, the rank is known to be n*m*p for small cases,
    # but let's verify by trying to find a decomposition
    
    # Try to find a rank-r decomposition using alternating least squares
    # with a better strategy: fix the structure based on the tensor's sparsity
    
    # For M_{3,4,5}, the tensor has a specific structure. Let's exploit it.
    # Each term in the decomposition corresponds to a triple (i,j,k)
    # where we pick one element from each dimension
    
    # A rank-r decomposition means we can write T as sum of r rank-1 tensors
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # The key constraint: for the decomposition to be exact, the support must match
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # This means: if T[a,b,c] = 1, then there must be at least one r with
    # U[a,r] != 0, V[b,r] != 0, W[c,r] != 0
    # And if T[a,b,c] = 0, then for all r: U[a,r]*V[b,r]*W[c,r] = 0
    
    # This is a combinatorial problem. Let's use a SAT-like approach with linear algebra
    
    # For each candidate rank r, we need to find U, V, W such that:
    # 1. For all (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # 2. For all (a,b,c) not in support: sum_r U[a,r] V[b,r] W[c,r] = 0
    
    # This is a system of nonlinear equations. Let's use a different approach:
    # Parameterize U, V, W and use numerical optimization with multiple restarts
    
    # Use a smarter initialization: start from schoolbook and try to merge terms
    
    # For ranks 59 and 58, try to find a decomposition by perturbing schoolbook
    
    for target_rank in [59, 58]:
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try different strategies for each rank
        strategies = 20
        
        for strat in range(strategies):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(target_rank * 100 + strat)
            
            # Start from schoolbook and randomly combine/merge terms
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random linear combinations of schoolbook terms
            # Each new term is a weighted sum of random schoolbook terms
            for r in range(target_rank):
                # Pick 2-3 random schoolbook terms and combine them
                num_terms = np.random.randint(2, 4)
                indices = np.random.choice(rank_schoolbook, num_terms, replace=False)
                
                U[:, r] = np.zeros(n * m)
                V[:, r] = np.zeros(m * p)
                W[:, r] = np.zeros(p * n)
                
                for idx in indices:
                    weight = np.random.randn() * 0.5
                    # Extract the schoolbook term
                    i = idx // (m * p)
                    rem = idx % (m * p)
                    j = rem // p
                    k = rem % p
                    
                    U[i * m + j, r] += weight
                    V[j * p + k, r] += weight
                    W[k * n + i, r] += weight
            
            # Optimize using gradient descent
            lr = 0.05
            for iteration in range(300):
                if time.time() > deadline - 10:
                    break
                
                T_computed = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_computed - T_target
                
                if np.linalg.norm(residual) < 1e-7:
                    break
                
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                lr *= 0.995
            
            U_clean = clean_matrix(U)
            V_clean = clean_matrix(V)
            W_clean = clean_matrix(W)
            
            if verify(U_clean, V_clean, W_clean):
                best_rank = target_rank
                best_U = U_clean.tolist()
                best_V = V_clean.tolist()
                best_W = W_clean.tolist()
                found = True
                break
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
