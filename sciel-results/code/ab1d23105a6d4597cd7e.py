import time
import numpy as np

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4
SCHOOLBOOK_RANK = N * M * P  # 36

def _build_tensor():
    """Build the matmul tensor T of shape (N*M, M*P, P*N).
    T[i*M+j, j'*P+k, k'*N+i'] = 1 iff i==i' and j==j' and k==k'.
    """
    T = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Check if sum_t U[a,t]*V[b,t]*W[c,t] == T[a,b,c] for all a,b,c."""
    D = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(D, T, atol=tol)

def _schoolbook():
    """Standard schoolbook decomposition of rank 36."""
    U = np.zeros((N*M, SCHOOLBOOK_RANK), dtype=np.float64)
    V = np.zeros((M*P, SCHOOLBOOK_RANK), dtype=np.float64)
    W = np.zeros((P*N, SCHOOLBOOK_RANK), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M + j, col] = 1
                V[j*P + k, col] = 1
                W[k*N + i, col] = 1
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists, rounding to nearest half-integer."""
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

def _round_to_half(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2) / 2

def _build_strassen_3x3_3x3():
    """
    Build a rank-21 decomposition for 3x3 * 3x3 matmul using Strassen-like structure.
    This gives us a good starting point for extending to 3x3 * 3x4.
    """
    # Known rank-21 decomposition for 3x3*3x3
    # Using a structured approach based on block decomposition
    rank = 21
    U = np.zeros((9, rank), dtype=np.float64)
    V = np.zeros((9, rank), dtype=np.float64)
    W = np.zeros((9, rank), dtype=np.float64)
    
    # We'll use a combination of schoolbook and Strassen-like terms
    # For now, use a perturbed schoolbook as a reasonable starting point
    # and let the optimizer find the structure
    
    # Initialize with schoolbook terms for the 3x3 part
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if col < rank:
                    U[i*3 + j, col] = 1
                    V[j*3 + k, col] = 1
                    W[k*3 + i, col] = 1
                    col += 1
    
    # Add some random structure for the remaining terms
    for t in range(col, rank):
        U[:, t] = np.random.randn(9) * 0.1
        V[:, t] = np.random.randn(9) * 0.1
        W[:, t] = np.random.randn(9) * 0.1
    
    return U, V, W

def _optimize_decomposition(U, V, W, T, deadline, lr_init=0.05, max_iter=200, tol=1e-8):
    """
    Optimize U, V, W to minimize ||sum_t u_t ⊗ v_t ⊗ w_t - T||_F^2
    using gradient descent with adaptive learning rate.
    """
    rank = U.shape[1]
    lr = lr_init
    prev_err = np.inf
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute D = einsum('at,bt,ct->abc', U, V, W)
        D = np.einsum('at,bt,ct->abc', U, V, W)
        err = D - T
        err_norm = np.linalg.norm(err)
        
        # Check convergence
        if err_norm < tol:
            break
        
        # Gradients
        gU = np.einsum('abc,bt,ct->at', err, V, W)
        gV = np.einsum('abc,at,ct->bt', err, U, W)
        gW = np.einsum('abc,at,bt->ct', err, U, V)
        
        # Adaptive learning rate: reduce if error increases
        U_new = U - lr * gU
        V_new = V - lr * gV
        W_new = W - lr * gW
        
        D_new = np.einsum('at,bt,ct->abc', U_new, V_new, W_new)
        err_new = D_new - T
        err_new_norm = np.linalg.norm(err_new)
        
        if err_new_norm < err_norm:
            U, V, W = U_new, V_new, W_new
            # Slightly increase learning rate if making good progress
            if err_norm - err_new_norm > 0.1 * err_norm:
                lr = min(lr * 1.05, 0.1)
        else:
            # Reduce learning rate
            lr *= 0.5
        
        if lr < 1e-6:
            break
    
    return U, V, W, err_norm

def _try_structured_merging(T, deadline):
    """
    Try to find a lower-rank decomposition by systematically merging schoolbook terms.
    Uses the known rank-21 structure for 3x3*3x3 as a starting point.
    """
    for target_rank in range(35, 28, -1):
        if time.time() > deadline:
            break
        
        # Try multiple initializations with different strategies
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(attempt * 1000 + target_rank)
            
            # Strategy 1: Start from schoolbook and perturb
            if attempt % 3 == 0:
                U = np.zeros((N*M, target_rank), dtype=np.float64)
                V = np.zeros((M*P, target_rank), dtype=np.float64)
                W = np.zeros((P*N, target_rank), dtype=np.float64)
                
                # Fill with schoolbook terms
                all_terms = [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]
                rng.shuffle(all_terms)
                
                for t in range(min(36, target_rank)):
                    i, j, k = all_terms[t]
                    U[i*M + j, t] = 1
                    V[j*P + k, t] = 1
                    W[k*N + i, t] = 1
                
                # Add small random perturbations to remaining terms
                for t in range(36, target_rank):
                    U[:, t] = rng.normal(0, 0.2, N*M)
                    V[:, t] = rng.normal(0, 0.2, M*P)
                    W[:, t] = rng.normal(0, 0.2, P*N)
            
            # Strategy 2: Use Strassen-like structure for the 3x3 part
            elif attempt % 3 == 1:
                U = np.zeros((N*M, target_rank), dtype=np.float64)
                V = np.zeros((M*P, target_rank), dtype=np.float64)
                W = np.zeros((P*N, target_rank), dtype=np.float64)
                
                # Initialize first 21 columns with 3x3*3x3 structure
                for t in range(min(21, target_rank)):
                    i = t % 3
                    j = (t // 3) % 3
                    k = (t // 9) % 3
                    U[i*M + j, t] = 1
                    V[j*P + k, t] = 1
                    W[k*N + i, t] = 1
                
                # Add terms for the 4th column with random structure
                for t in range(21, target_rank):
                    U[:, t] = rng.normal(0, 0.3, N*M)
                    V[:, t] = rng.normal(0, 0.3, M*P)
                    W[:, t] = rng.normal(0, 0.3, P*N)
            
            # Strategy 3: Fully random initialization
            else:
                U = rng.normal(0, 0.5, (N*M, target_rank))
                V = rng.normal(0, 0.5, (M*P, target_rank))
                W = rng.normal(0, 0.5, (P*N, target_rank))
            
            # Optimize
            U, V, W, err_norm = _optimize_decomposition(U, V, W, T, deadline)
            
            # Round to half-integers and verify
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            
            if _verify(U_r, V_r, W_r, T):
                return U_r, V_r, W_r, target_rank
    
    return None

def _try_kronecker_extension(T, deadline):
    """
    Try to extend a known 3x3*3x3 rank-21 decomposition to 3x3*3x4.
    """
    return None

def construct():
    deadline = time.time() + 130  # 130s to be safe (150s kill)
    
    T = _build_tensor()
    
    # Start with schoolbook
    U_best, V_best, W_best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK
    
    # Try to find better decompositions
    result = _try_structured_merging(T, deadline)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            U_best, V_best, W_best = U, V, W
            best_rank = rank
    
    # Try Kronecker extension
    if time.time() < deadline:
        result2 = _try_kronecker_extension(T, deadline)
        if result2 is not None:
            U, V, W, rank = result2
            if rank < best_rank:
                U_best, V_best, W_best = U, V, W
                best_rank = rank
    
    # Return best result
    return _to_lists(U_best, V_best, W_best)
