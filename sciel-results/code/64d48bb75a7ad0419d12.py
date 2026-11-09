"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # For each (i,j,k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # and for all other combinations it should be 0.
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Efficient check: for each column c, we have a rank-1 tensor contribution
    # U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # Let's verify by checking a few random matrices
    # Actually, let's do an exact check using the tensor definition
    # The matrix multiplication tensor T has T[i,j,k] = 1 if this corresponds to (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The tensor for matrix multiplication is:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has indices (i,j,k) and T[i,j,k] = 1 for all i,j,k in range
    # because each term A_{i,j} * B_{j,k} contributes to (A*B)_{i,k}
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # We want: (A*B)_{i,k} = sum_c U[i,j] * V[j,k] * W[k,i] for the appropriate j
    # This is tricky. Let me think again.
    
    # The matrix multiplication can be written as:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a 3-way tensor T_{i,j,k} where T_{i,j,k} = 1 for all i,j,k
    # and the decomposition is: T_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # where U_{i,j}^{(c)} corresponds to U[i*m+j, c], etc.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # And there are no other constraints because the tensor is all ones.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
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

def _try_random_rank(n, m, p, target_rank, max_iter=5000, deadline=None):
    """Try to find a decomposition of the given rank using random initialization."""
    start_time = time.time()
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Random initialization with small values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Check if this works
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_local_search(n, m, p, start_rank, max_iter=10000, deadline=None):
    """Try to reduce the rank by local search from the schoolbook solution."""
    start_time = time.time()
    
    # Start from schoolbook
    result = _make_schoolbook(n, m, p)
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    current_rank = n * m * p
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Try to remove one random column
        if current_rank <= 1:
            break
        
        col_to_remove = np.random.randint(current_rank)
        
        # Check if removing this column still gives a valid decomposition
        # This is expensive, so let's do a quick check
        # Actually, let's try a different approach: perturb and see if we can merge
        
        # Alternative: try to find if any column is redundant
        # For now, let's just try random perturbations
        pass
    
    return result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook
    best_result = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But the actual rank might be higher.
    
    # Try random search for lower ranks
    for target_rank in range(max(n*m, m*p, p*n), best_rank):
        if time.time() > deadline:
            break
        
        result = _try_random_rank(n, m, p, target_rank, max_iter=1000, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
