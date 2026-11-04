"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions, starting from lower ranks
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 20:
            break
        result = _find_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result

def _schoolbook(n, m, p):
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

def _compute_error(U_np, V_np, W_np, n, m, p):
    """Compute total squared error for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                err += (val - 1.0) ** 2
    return err

def _is_valid(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is valid."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    err = _compute_error(U_np, V_np, W_np, n, m, p)
    return err < tol * tol

def _find_decomposition(n, m, p, target_rank, deadline):
    """Find a decomposition of given rank using enhanced random search with local optimization."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    max_attempts = 200
    attempt = 0
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    while attempt < max_attempts and time.time() < deadline - 15:
        attempt += 1
        
        # Generate random initialization
        U = _rand_mat(n_m, target_rank, scale=1.0)
        V = _rand_mat(m_p, target_rank, scale=1.0)
        W = _rand_mat(p_n, target_rank, scale=1.0)
        
        # Compute error
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        err = _compute_error(U_np, V_np, W_np, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U, V, W
        
        if err < 1e-12:
            return _to_output(U, V, W)
        
        # Local optimization from this starting point
        U_opt, V_opt, W_opt, err_opt = _local_opt(U, V, W, n, m, p, deadline, max_steps=500)
        
        if err_opt < best_err:
            best_err = err_opt
            best_U, best_V, best_W = U_opt, V_opt, W_opt
        
        if err_opt < 1e-12:
            return _to_output(U_opt, V_opt, W_opt)
        
        # Every few attempts, try a structured initialization
        if attempt % 5 == 0:
            U_s, V_s, W_s = _structured_init(n, m, p, target_rank)
            U_s_np = np.array(U_s, dtype=np.float64)
            V_s_np = np.array(V_s, dtype=np.float64)
            W_s_np = np.array(W_s, dtype=np.float64)
            err_s = _compute_error(U_s_np, V_s_np, W_s_np, n, m, p)
            
            if err_s < best_err:
                best_err = err_s
                best_U, best_V, best_W = U_s, V_s, W_s
            
            U_so, V_so, W_so, err_so = _local_opt(U_s, V_s, W_s, n, m, p, deadline, max_steps=500)
            
            if err_so < best_err:
                best_err = err_so
                best_U, best_V, best_W = U_so, V_so, W_so
            
            if err_so < 1e-12:
                return _to_output(U_so, V_so, W_so)
    
    # If we found a very close solution, try to polish it
    if best_U is not None and best_err < 1.0:
        U_pol, V_pol, W_pol, err_pol = _polish(best_U, best_V, best_W, n, m, p, deadline)
        if err_pol < 1e-10:
            return _to_output(U_pol, V_pol, W_pol)
    
    return None

def _to_output(U, V, W):
    """Convert to output format, rounding near-integer values."""
    U_out = []
    for row in U:
        U_out.append([_round_entry(x) for x in row])
    V_out = []
    for row in V:
        V_out.append([_round_entry(x) for x in row])
    W_out = []
    for row in W:
        W_out.append([_round_entry(x) for x in row])
    return {"U": U_out, "V": V_out, "W": W_out}

def _round_entry(x):
    """Round to nearest 0.5 if close, otherwise keep as is."""
    r = round(x * 2) / 2
    if abs(x - r) < 1e-8:
        return r
    return x

def _rand_mat(rows, cols, scale=1.0):
    """Generate random matrix with continuous values in [-scale, scale]."""
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

def _structured_init(n, m, p, rank):
    """Initialize with structure inspired by matrix multiplication patterns."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U = [[0.0] * rank for _ in range(n_m)]
    V = [[0.0] * rank for _ in range(m_p)]
    W = [[0.0] * rank for _ in range(p_n)]
    
    # Create rank patterns based on (i,j,k) tuples
    patterns = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                patterns.append((i, j, k))
    
    # Shuffle and select
    random.shuffle(patterns)
    for col in range(rank):
        if col < len(patterns):
            i, j, k = patterns[col]
        else:
            i = random.randrange(n)
            j = random.randrange(m)
            k = random.randrange(p)
        
        # Assign values with some structure
        u_val = random.choice([1.0, -1.0, 0.5, -0.5, 1.5, -1.5])
        v_val = random.choice([1.0, -1.0, 0.5, -0.5, 1.5, -1.5])
        w_val = random.choice([1.0, -1.0, 0.5, -0.5, 1.5, -1.5])
        
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
    
    return U, V, W

def _local_opt(U, V, W, n, m, p, deadline, max_steps=500):
    """Local optimization using gradient-like steps."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_err = _compute_error(U_np, V_np, W_np, n, m, p)
    
    step_size = 0.5
    for step in range(max_steps):
        if time.time() > deadline - 15:
            break
        
        # Pick a random matrix and entry
        mat_idx = random.choice([0, 1, 2])
        
        if mat_idx == 0:
            r = random.randrange(len(U))
            c = random.randrange(rank)
            old = U[r][c]
            delta = random.choice([-1, 1]) * step_size * random.uniform(0.5, 1.5)
            new = old + delta
            U[r][c] = new
            U_np[r][c] = new
            
            new_err = _compute_error(U_np, V_np, W_np, n, m, p)
            if new_err >= current_err:
                U[r][c] = old
                U_np[r][c] = old
            else:
                current_err = new_err
        elif mat_idx == 1:
            r = random.randrange(len(V))
            c = random.randrange(rank)
            old = V[r][c]
            delta = random.choice([-1, 1]) * step_size * random.uniform(0.5, 1.5)
            new = old + delta
            V[r][c] = new
            V_np[r][c] = new
            
            new_err = _compute_error(U_np, V_np, W_np, n, m, p)
            if new_err >= current_err:
                V[r][c] = old
                V_np[r][c] = old
            else:
                current_err = new_err
        else:
            r = random.randrange(len(W))
            c = random.randrange(rank)
            old = W[r][c]
            delta = random.choice([-1, 1]) * step_size * random.uniform(0.5, 1.5)
            new = old + delta
            W[r][c] = new
            W_np[r][c] = new
            
            new_err = _compute_error(U_np, V_np, W_np, n, m, p)
            if new_err >= current_err:
                W[r][c] = old
                W_np[r][c] = old
            else:
                current_err = new_err
        
        # Occasionally reduce step size
        if step % 100 == 99:
            step_size *= 0.8
    
    return U, V, W, current_err

def _polish(U, V, W, n, m, p, deadline):
    """High-precision polish pass to get exact solution."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_err = _compute_error(U_np, V_np, W_np, n, m, p)
    
    # Very small steps for precision
    for step in range(1000):
        if time.time() > deadline - 15:
            break
        if current_err < 1e-12:
            break
        
        mat_idx = random.choice([0, 1, 2])
        
        if mat_idx == 0:
            r = random.randrange(len(U))
            c = random.randrange(rank)
            old = U[r][c]
            delta = random.choice([-1, 1]) * 0.05 * random.uniform(0.5, 1.5)
            new = old + delta
            U[r][c] = new
            U_np[r][c] = new
            
            new_err = _compute_error(U_np, V_np, W_np, n, m, p)
            if new_err >= current_err:
                U[r][c] = old
                U_np[r][c] = old
            else:
                current_err = new_err
        elif mat_idx == 1:
            r = random.randrange(len(V))
            c = random.randrange(rank)
            old = V[r][c]
            delta = random.choice([-1, 1]) * 0.05 * random.uniform(0.5, 1.5)
            new = old + delta
            V[r][c] = new
            V_np[r][c] = new
            
            new_err = _compute_error(U_np, V_np, W_np, n, m, p)
            if new_err >= current_err:
                V[r][c] = old
                V_np[r][c] = old
            else:
                current_err = new_err
        else:
            r = random.randrange(len(W))
            c = random.randrange(rank)
            old = W[r][c]
            delta = random.choice([-1, 1]) * 0.05 * random.uniform(0.5, 1.5)
            new = old + delta
            W[r][c] = new
            W_np[r][c] = new
            
            new_err = _compute_error(U_np, V_np, W_np, n, m, p)
            if new_err >= current_err:
                W[r][c] = old
                W_np[r][c] = old
            else:
                current_err = new_err
    
    return U, V, W, current_err
# EVOLVE-BLOCK-END
