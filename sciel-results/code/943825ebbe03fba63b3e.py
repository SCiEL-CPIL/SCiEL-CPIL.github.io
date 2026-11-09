import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form an exact decomposition of the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    
    The decomposition must satisfy:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    For all (a, b, c) not corresponding to valid (i, j, k): sum_c U[a, c] * V[b, c] * W[c, c] = 0
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Check 1s: for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a] * V[b], W[c])
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check 0s: for all (a, b, c) that are NOT valid
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))
                
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_indices:
                    val = np.dot(U[a] * V[b], W[c])
                    if abs(val) > tol:
                        return False
                        
    return True

def _try_rank_r_structured(n, m, p, r, max_attempts=200, seed=42):
    """
    Try to find a rank-r decomposition using structured patterns.
    """
    np.random.seed(seed)
    
    for attempt in range(max_attempts):
        # Generate random sparse U and V with small integer entries
        # Use {-1, 0, 1} entries to increase chance of integer W
        density = 0.3 + 0.4 * (attempt % 10) / 10.0  # Vary density
        U = np.random.randint(-1, 2, size=(n * m, r))
        V = np.random.randint(-1, 2, size=(m * p, r))
        
        # Sparsify
        mask_u = np.random.rand(n * m, r) < density
        mask_v = np.random.rand(m * p, r) < density
        U = U * mask_u
        V = V * mask_v
        
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Construct linear system for W[k*n+i, :]
                # Variables: w_0, ..., w_{r-1}
                # Equations: for j in 0..m-1:
                #   sum_c U[i*m+j, c] * V[j*p+k, c] * w_c = 1
                
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A_mat * w_vec = ones(m)
                try:
                    w_vec, residuals, rank, s = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to integers
                w_int = np.round(w_vec)
                if np.linalg.norm(A_mat @ w_int - np.ones(m)) > 1e-9:
                    valid = False
                    break
                
                W[k * n + i, :] = w_int
        
        if not valid:
            continue
            
        # Final verification including zeros
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def _try_rank_r_kronecker(n, m, p, r, max_attempts=50, seed=100):
    """
    Try to find a rank-r decomposition using Kronecker product structure.
    For small matrices, we can try to exploit the structure.
    """
    np.random.seed(seed)
    
    for attempt in range(max_attempts):
        # Try different factorizations of r
        found = False
        
        # Generate random structured U and V
        # Use block patterns based on the dimensions
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        # Try to assign columns to different (i, j) or (j, k) pairs
        # Random permutation of columns
        col_perm = np.random.permutation(r)
        
        for c in range(r):
            # Assign this column to a random (i, j) for U and (j, k) for V
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            
            # Set U[i*m + j, c] and V[j*p + k, c] to random small integers
            U[i * m + j, c] = np.random.randint(-2, 3)
            V[j * p + k, c] = np.random.randint(-2, 3)
            
            # Also set some additional entries to create overlap
            if np.random.rand() < 0.3:
                i2 = np.random.randint(n)
                j2 = np.random.randint(m)
                U[i2 * m + j2, c] = np.random.randint(-1, 2)
            if np.random.rand() < 0.3:
                j3 = np.random.randint(m)
                k3 = np.random.randint(p)
                V[j3 * p + k3, c] = np.random.randint(-1, 2)
        
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                try:
                    w_vec, residuals, rank, s = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                w_int = np.round(w_vec)
                if np.linalg.norm(A_mat @ w_int - np.ones(m)) > 1e-9:
                    valid = False
                    break
                
                W[k * n + i, :] = w_int
        
        if not valid:
            continue
            
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def _try_rank_r_fast(n, m, p, r, max_attempts=300, seed=200):
    """
    Fast approach: use random binary patterns and verify quickly.
    """
    np.random.seed(seed)
    
    for attempt in range(max_attempts):
        # Use binary patterns for U and V
        U = np.random.randint(0, 2, size=(n * m, r))
        V = np.random.randint(0, 2, size=(m * p, r))
        
        # Ensure some columns have both U and V non-zero for the same (i, j, k)
        # to allow W to be solved
        
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                try:
                    w_vec, residuals, rank, s = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # For binary U, V, W should be integer
                w_int = np.round(w_vec)
                if np.linalg.norm(A_mat @ w_int - np.ones(m)) > 1e-9:
                    valid = False
                    break
                
                W[k * n + i, :] = w_int
        
        if not valid:
            continue
            
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Trivial solution fallback
    rank_trivial = n * m * p
    U_triv = np.zeros((n * m, rank_trivial))
    V_triv = np.zeros((m * p, rank_trivial))
    W_triv = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1.0
                V_triv[j * p + k, col] = 1.0
                W_triv[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_trivial
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Lower bound
    lb = max(n * m, m * p, p * n)
    
    # Search from lb upwards
    for r in range(lb, rank_trivial + 1):
        if time.time() > deadline:
            break
            
        # Try multiple approaches
        found = False
        
        # Approach 1: Structured random
        if not found:
            res = _try_rank_r_structured(n, m, p, r, max_attempts=50, seed=r)
            if res is not None:
                best_rank = r
                best_U, best_V, best_W = res
                found = True
        
        # Approach 2: Kronecker-like
        if not found and time.time() < deadline - 5:
            res = _try_rank_r_kronecker(n, m, p, r, max_attempts=30, seed=r + 1000)
            if res is not None:
                best_rank = r
                best_U, best_V, best_W = res
                found = True
        
        # Approach 3: Fast binary
        if not found and time.time() < deadline - 3:
            res = _try_rank_r_fast(n, m, p, r, max_attempts=100, seed=r + 2000)
            if res is not None:
                best_rank = r
                best_U, best_V, best_W = res
                found = True
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
