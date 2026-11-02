<<<<<<< SEARCH
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]
VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                nz_indices.append((u, v, w))
    return T, nz_indices

def _compute_error(U, V, W, T, nz_indices):
    # Compute reconstruction
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # T: (nm, mp, pn)
    # We want sum_r U[:,r] V[:,r] W[:,r] to match T
    # Using einsum
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    # Error is sum of squares of differences
    # To speed up, we can just check nz and a sample of zeros?
    # For optimization, full error is better.
    # T is sparse, but recon is dense.
    # Error = ||recon - T||^2
    return np.sum((recon - T) ** 2)

def _check_exact(U, V, W, T, nz_indices, n, m, p):
    # Check non-zero positions
    for u, v, w in nz_indices:
        val = np.sum(U[:, u] * V[:, v] * W[:, w]) # Wait, U is (nm, R), so U[u, :]
        # U shape is (n*m, R). So U[u, r].
        val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
        if abs(val - 1.0) > 1e-9:
            return False
    
    # Check zero positions
    # To be safe, check all zero positions if fast enough, else sample
    # Total positions: 6*12*8 = 576
    # NZ: 24
    # Zeros: 552
    # 552 checks * R (up to 23) * 3 mults = ~38k ops. Fast.
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) in nz_indices:
                    continue
                val = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                if abs(val) > 1e-9:
                    return False
    return True

def _get_schoolbook(n, m, p):
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

def _search_rank(R, n, m, p, T, nz_indices, time_limit):
    start_time = time.time()
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Number of restarts
    num_restarts = 100
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = np.random.choice(VALUES, size=(n * m, R))
        V = np.random.choice(VALUES, size=(m * p, R))
        W = np.random.choice(VALUES, size=(p * n, R))
        
        current_error = _compute_error(U, V, W, T, nz_indices)
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if best_error < 1e-9:
                return best_U, best_V, best_W
                
        # Local Search
        # We use a simple hill climbing with occasional 2-variable moves
        max_iter = 1000
        temp = 1.0
        cooling = 0.99
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if current_error < 1e-9:
                break
                
            # Pick move type: 1 variable (80%) or 2 variables (20%)
            if random.random() < 0.8:
                # 1 variable
                total_vars = U.size + V.size + W.size
                var_idx = random.randint(0, total_vars - 1)
                
                if var_idx < U.size:
                    mat = U
                    idx = var_idx
                elif var_idx < U.size + V.size:
                    mat = V
                    idx = var_idx - U.size
                else:
                    mat = W
                    idx = var_idx - U.size - V.size
                    
                old_val = mat.flat[idx]
                new_val = random.choice(VALUES)
                if new_val == old_val:
                    continue
                    
                mat.flat[idx] = new_val
                new_error = _compute_error(U, V, W, T, nz_indices)
                
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_error < 1e-9:
                            return best_U, best_V, best_W
                else:
                    mat.flat[idx] = old_val
            else:
                # 2 variables
                # Pick two distinct variables
                total_vars = U.size + V.size + W.size
                var1 = random.randint(0, total_vars - 1)
                var2 = random.randint(0, total_vars - 1)
                if var1 == var2:
                    continue
                
                def get_mat_idx(idx):
                    if idx < U.size:
                        return U, idx
                    elif idx < U.size + V.size:
                        return V, idx - U.size
                    else:
                        return W, idx - U.size - V.size
                
                mat1, idx1 = get_mat_idx(var1)
                mat2, idx2 = get_mat_idx(var2)
                
                old_val1 = mat1.flat[idx1]
                old_val2 = mat2.flat[idx2]
                
                new_val1 = random.choice(VALUES)
                new_val2 = random.choice(VALUES)
                
                if new_val1 == old_val1 and new_val2 == old_val2:
                    continue
                    
                mat1.flat[idx1] = new_val1
                mat2.flat[idx2] = new_val2
                
                new_error = _compute_error(U, V, W, T, nz_indices)
                
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_error < 1e-9:
                            return best_U, best_V, best_W
                else:
                    mat1.flat[idx1] = old_val1
                    mat2.flat[idx2] = old_val2
                    
            temp *= cooling
            
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T, nz_indices = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Try ranks from low to high (or high to low? Low to high is better for finding minimal)
    # But if 24 is the answer, we want to fail fast for lower ranks.
    # Let's try 12 to 23.
    for R in range(12, best_rank):
        if time.time() - start_time > deadline:
            break
            
        # Allocate more time to lower ranks? Or equal?
        # Lower ranks are harder to find if they exist.
        # Let's give 10s per rank.
        time_budget = min(10.0, (deadline - (time.time() - start_time)) / max(best_rank - R, 1))
        if time_budget < 1.0:
            break
            
        result = _search_rank(R, n, m, p, T, nz_indices, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify exactness again just in case
            if _check_exact(U, V, W, T, nz_indices, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
=======
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]
VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                nz_indices.append((u, v, w))
    return T, nz_indices

def _compute_error_fast(U, V, W, T, nz_indices, n, m, p):
    """Compute ||recon - T||^2 efficiently using the structure of T."""
    # recon[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # We need to compute this for all u,v,w and compare to T.
    # Use einsum for the full reconstruction
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.sum((recon - T) ** 2)

def _compute_error_sparse(U, V, W, nz_indices, n, m, p, sample_zeros=True):
    """Compute error only at nonzero positions and optionally sample zeros.
    This is much faster for large R."""
    nm, mp, pn = n * m, m * p, p * n
    error = 0.0
    
    # Check all nonzero positions
    for u, v, w in nz_indices:
        val = np.dot(U[u, :], V[v, :] * W[w, :])
        error += (val - 1.0) ** 2
    
    # Sample some zero positions to get a sense of the error there
    if sample_zeros:
        # Pre-compute which positions are nonzero
        nz_set = set(nz_indices)
        # Sample up to 100 zero positions
        num_samples = min(100, nm * mp * pn - len(nz_indices))
        for _ in range(num_samples):
            u = random.randint(0, nm - 1)
            v = random.randint(0, mp - 1)
            w = random.randint(0, pn - 1)
            if (u, v, w) in nz_set:
                continue
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            error += val ** 2
        # Scale up the zero error estimate
        if num_samples > 0:
            error += error * (nm * mp * pn - len(nz_indices) - num_samples) / num_samples
    
    return error

def _check_exact(U, V, W, T, nz_indices, n, m, p):
    # Check non-zero positions
    for u, v, w in nz_indices:
        val = np.dot(U[u, :], V[v, :] * W[w, :])
        if abs(val - 1.0) > 1e-9:
            return False
    
    # Check zero positions
    nz_set = set(nz_indices)
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) in nz_set:
                    continue
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(val) > 1e-9:
                    return False
    return True

def _get_schoolbook(n, m, p):
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

def _structured_init(R, n, m, p, nz_indices):
    """Initialize with structure: some columns follow the schoolbook pattern,
    others are random."""
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For the first min(R, 24) columns, use schoolbook-like structure
    num_structured = min(R, n * m * p)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= num_structured:
                    break
                U[i * m + j, col] = random.choice([1.0, -1.0])
                V[j * p + k, col] = random.choice([1.0, -1.0])
                W[k * n + i, col] = random.choice([1.0, -1.0])
                col += 1
    
    # Fill remaining columns randomly
    for r in range(num_structured, R):
        U[:, r] = np.random.choice(VALUES, size=n * m)
        V[:, r] = np.random.choice(VALUES, size=m * p)
        W[:, r] = np.random.choice(VALUES, size=p * n)
    
    return U, V, V, W

def _random_init(R, n, m, p):
    """Fully random initialization."""
    U = np.random.choice(VALUES, size=(n * m, R))
    V = np.random.choice(VALUES, size=(m * p, R))
    W = np.random.choice(VALUES, size=(p * n, R))
    return U, V, W

def _search_rank(R, n, m, p, T, nz_indices, time_limit):
    start_time = time.time()
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Number of restarts - increase for faster error computation
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Use structured init for first half, random for second half
        if restart < num_restarts // 2:
            U, V, W = _structured_init(R, n, m, p, nz_indices)
        else:
            U, V, W = _random_init(R, n, m, p)
        
        current_error = _compute_error_fast(U, V, W, T, nz_indices, n, m, p)
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if best_error < 1e-9:
                return best_U, best_V, best_W
                
        # Local Search with adaptive iteration count
        max_iter = 2000
        temp = 1.0
        cooling = 0.995
        no_improve_count = 0
        max_no_improve = 500
        
        for iter_num in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if current_error < 1e-9:
                break
            if no_improve_count > max_no_improve:
                break
                
            # Pick move type: 1 variable (70%) or 2 variables (30%)
            if random.random() < 0.7:
                # 1 variable
                total_vars = U.size + V.size + W.size
                var_idx = random.randint(0, total_vars - 1)
                
                if var_idx < U.size:
                    mat = U
                    idx = var_idx
                elif var_idx < U.size + V.size:
                    mat = V
                    idx = var_idx - U.size
                else:
                    mat = W
                    idx = var_idx - U.size - V.size
                    
                old_val = mat.flat[idx]
                new_val = random.choice(VALUES)
                if new_val == old_val:
                    continue
                    
                mat.flat[idx] = new_val
                new_error = _compute_error_fast(U, V, W, T, nz_indices, n, m, p)
                
                delta = new_error - current_error
                if delta < -1e-12 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_error = new_error
                    no_improve_count = 0
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_error < 1e-9:
                            return best_U, best_V, best_W
                else:
                    mat.flat[idx] = old_val
                    no_improve_count += 1
            else:
                # 2 variables
                total_vars = U.size + V.size + W.size
                var1 = random.randint(0, total_vars - 1)
                var2 = random.randint(0, total_vars - 1)
                if var1 == var2:
                    continue
                
                def get_mat_idx(idx):
                    if idx < U.size:
                        return U, idx
                    elif idx < U.size + V.size:
                        return V, idx - U.size
                    else:
                        return W, idx - U.size - V.size
                
                mat1, idx1 = get_mat_idx(var1)
                mat2, idx2 = get_mat_idx(var2)
                
                old_val1 = mat1.flat[idx1]
                old_val2 = mat2.flat[idx2]
                
                new_val1 = random.choice(VALUES)
                new_val2 = random.choice(VALUES)
                
                if new_val1 == old_val1 and new_val2 == old_val2:
                    continue
                    
                mat1.flat[idx1] = new_val1
                mat2.flat[idx2] = new_val2
                
                new_error = _compute_error_fast(U, V, W, T, nz_indices, n, m, p)
                
                delta = new_error - current_error
                if delta < -1e-12 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_error = new_error
                    no_improve_count = 0
                    if current_error < best_error:
                        best_error = current_error
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_error < 1e-9:
                            return best_U, best_V, best_W
                else:
                    mat1.flat[idx1] = old_val1
                    mat2.flat[idx2] = old_val2
                    no_improve_count += 1
                    
            temp *= cooling
            
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    T, nz_indices = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But matrix multiplication rank is at least n*m*p / something...
    # For 2x3x4, known results suggest rank might be around 18-24
    for R in range(14, best_rank):
        if time.time() - start_time > deadline:
            break
            
        # Allocate time based on remaining ranks
        ranks_remaining = best_rank - R
        time_budget = min(15.0, (deadline - (time.time() - start_time)) / max(ranks_remaining, 1))
        if time_budget < 2.0:
            break
            
        result = _search_rank(R, n, m, p, T, nz_indices, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify exactness again just in case
            if _check_exact(U, V, W, T, nz_indices, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
>>>>>>> REPLACE
