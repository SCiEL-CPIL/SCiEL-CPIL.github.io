"""Matrix multiplication 2x4x4 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]
n, m, p = DIMENSIONS
nm, mp, pn = n * m, m * p, p * n


def build_tensor():
    """Build the flattened matrix multiplication tensor T[a, b, c] = T_{a,b,c}."""
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def check_decomposition(U, V, W, T, tol=1e-6):
    """Check if U, V, W exactly decompose T."""
    rank = U.shape[1]
    # Reconstruct: T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_rec, T, atol=tol)


def random_decomposition(rank, seed=None, structure='generic'):
    """Generate a random rank-`rank` decomposition with some structure."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # Apply some structure: make entries from {-1, 0, 1} or small integers/halves
    # This helps with exactness
    def discretize(mat):
        mat = mat * 2
        mat = np.round(mat)
        mat = np.clip(mat, -4, 4) / 2.0
        return mat
    
    U = discretize(U)
    V = discretize(V)
    W = discretize(W)
    
    return U, V, W


def local_search(U, V, W, T, max_iter=5000, tol=1e-6):
    """Local search to find an exact decomposition."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Compute initial error
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_rec - T) ** 2)
    
    best_error = error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if error < tol:
        return U, V, W, error
    
    # Try to improve by adjusting one entry at a time
    # Use a simple hill-climbing approach
    step_sizes = [0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 0.25, -0.25, 0.75, -0.75]
    
    for _ in range(max_iter):
        if error < tol:
            break
        
        # Randomly pick a matrix and a position
        mat_choice = random.randint(0, 2)
        if mat_choice == 0:
            mat = U
            r, c = random.randint(0, nm - 1), random.randint(0, rank - 1)
        elif mat_choice == 1:
            mat = V
            r, c = random.randint(0, mp - 1), random.randint(0, rank - 1)
        else:
            mat = W
            r, c = random.randint(0, pn - 1), random.randint(0, rank - 1)
        
        old_val = mat[r, c]
        step = random.choice(step_sizes)
        new_val = old_val + step
        
        # Limit magnitude
        new_val = np.clip(new_val, -4, 4)
        # Round to nearest 0.25
        new_val = round(new_val * 4) / 4
        
        if new_val == old_val:
            continue
        
        mat[r, c] = new_val
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        new_error = np.sum((T_rec - T) ** 2)
        
        if new_error < error:
            error = new_error
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            mat[r, c] = old_val
    
    if best_error < error:
        U, V, W, error = best_U, best_W, best_W, best_error
    
    return U, V, W, error


def try_rank(R, time_budget, T):
    """Try to find a rank-R decomposition within time_budget seconds."""
    start = time.time()
    deadline = start + time_budget
    
    # Try a few random initializations with local search
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        U, V, W = random_decomposition(R, seed=attempt)
        U, V, W, error = local_search(U, V, W, T, max_iter=2000)
        
        if error < 1e-6:
            # Verify with exact check
            if check_decomposition(U, V, W, T, tol=1e-8):
                return U, V, W
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication."""
    start = time.time()
    deadline = start + 115  # Leave 5s safety margin
    
    T = build_tensor()
    
    # First, try the known rank-16 construction
    # For each (j, k) pair, create one term
    # U[i*m+j, c(j,k)] = 1 if c's j-part is j
    # V[j*p+k, c(j,k)] = 1 if c's j-part is j AND k-part is k  
    # W[k*n+i, c(j,k)] = 1 if c's k-part is k
    
    # Let's index c by (j, k) where j in 0..3, k in 0..3
    R16 = m * p  # 16
    U16 = np.zeros((nm, R16), dtype=float)
    V16 = np.zeros((mp, R16), dtype=float)
    W16 = np.zeros((pn, R16), dtype=float)
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U: depends on i and j; for this term, j is fixed
            for i in range(n):
                U16[i * m + j, c] = 1.0
            # V: depends on j and k; both fixed for this term
            V16[j * p + k, c] = 1.0
            # W: depends on k and i; k is fixed, i varies
            for i in range(n):
                W16[k * n + i, c] = 1.0
    
    if check_decomposition(U16, V16, W16, T, tol=1e-8):
        best_rank = R16
        best_U, best_V, best_W = U16, V16, W16
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U, best_V, best_W = None, None, None
    
    # Try to find lower ranks
    for R in range(8, best_rank):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
        
        # Allocate some time for this rank
        alloc = min(time_left * 0.3, 15.0)
        result = try_rank(R, alloc, T)
        
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W, T, tol=1e-8):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                # Found a lower rank, continue searching for even lower
                continue
    
    # If no improvement, use schoolbook
    if best_U is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert numpy arrays to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
