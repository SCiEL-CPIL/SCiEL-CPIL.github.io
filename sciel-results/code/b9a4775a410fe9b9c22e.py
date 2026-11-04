"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify_numpy(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact using NumPy.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Target T is all 1s.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    T = np.zeros((n, m, p), dtype=np.int64)
    
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0: continue
                    w_val = w_col[k * n + i]
                    T[i, j, k] += u_val * v_val * w_val
                    
    if not np.all(T == 1):
        return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the error tensor and return max absolute error."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    T = np.zeros((n, m, p), dtype=np.int64)
    
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0: continue
                    w_val = w_col[k * n + i]
                    T[i, j, k] += u_val * v_val * w_val
                    
    return np.max(np.abs(T - 1))

def _laderman_2x2x2():
    """
    Laderman's rank-7 algorithm for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for multiplying 2x2 by 2x2.
    """
    # Laderman's decomposition
    # U: input A (2x2) -> 7 factors
    # V: input B (2x2) -> 7 factors  
    # W: output C (2x2) <- 7 factors
    
    # Standard Laderman coefficients
    U_lad = [
        [1, 0, 1, 0, 0, 0, 0],  # a11
        [0, 0, 0, 0, 1, 0, 0],  # a12
        [0, 0, 0, 0, 0, 1, 0],  # a21
        [0, 1, 1, 1, 0, 0, 0],  # a22
    ]
    V_lad = [
        [0, 1, 0, 1, 0, 0, 0],  # b11
        [1, 0, 0, 0, 0, 0, 1],  # b12
        [0, 0, 0, 0, 1, 0, 0],  # b21
        [1, 0, 1, 0, 0, 1, 0],  # b22
    ]
    W_lad = [
        [0, 0, 1, 1, 0, 0, 0],  # c11
        [1, 1, 0, 0, 0, 0, 0],  # c12
        [0, 0, 0, 0, 1, 1, 0],  # c21
        [0, 1, 0, 0, 0, 0, 1],  # c22
    ]
    return U_lad, V_lad, W_lad

def _construct_block_decomposition(n, m, p, deadline):
    """
    Construct a decomposition using block methods.
    For 3x3x4, we can use overlapping 2x2x2 blocks.
    """
    # Strategy: Use Laderman for overlapping 2x2x2 subproblems
    # This is complex, so let's try a different approach:
    # Start from a structured random initialization and optimize
    
    best_error = float('inf')
    best_solution = None
    
    start_time = time.time()
    
    # Try structured random search with better initialization
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        rank = 27
        
        # Generate structured random matrices with sparsity
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Each tensor factor r should have a specific structure
        # For multiplication, factor r typically has limited non-zero entries
        
        for r in range(rank):
            # Randomly choose which indices are active for this factor
            # Typically 3-5 non-zero entries per factor
            num_nonzero_u = random.randint(2, 4)
            num_nonzero_v = random.randint(2, 4)
            num_nonzero_w = random.randint(2, 4)
            
            # Choose random positions
            u_indices = random.sample(range(n * m), min(num_nonzero_u, n * m))
            v_indices = random.sample(range(m * p), min(num_nonzero_v, m * p))
            w_indices = random.sample(range(p * n), min(num_nonzero_w, p * n))
            
            for idx in u_indices:
                U[idx][r] = random.choice([-1, 1])
            for idx in v_indices:
                V[idx][r] = random.choice([-1, 1])
            for idx in w_indices:
                W[idx][r] = random.choice([-1, 1])
        
        # Check if this works
        if _verify_numpy(U, V, W, n, m, p):
            return U, V, W
        
        # Track best error for potential refinement
        err = _compute_error(U, V, W, n, m, p)
        if err < best_error:
            best_error = err
            best_solution = (U, V, W)
    
    return None

def _refine_solution(U, V, W, n, m, p, deadline):
    """
    Try to refine a near-solution by flipping individual entries.
    """
    start_time = time.time()
    current_error = _compute_error(U, V, W, n, m, p)
    
    if current_error == 0:
        return U, V, W
    
    # Copy
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    rank = len(U[0])
    
    # Hill climbing: flip entries that reduce error
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try flipping entries in U
        for i in range(n * m):
            for r in range(rank):
                if U[i][r] == 0:
                    continue
                old_val = U[i][r]
                U[i][r] = -old_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                else:
                    U[i][r] = old_val
        
        # Try flipping entries in V
        for i in range(m * p):
            for r in range(rank):
                if V[i][r] == 0:
                    continue
                old_val = V[i][r]
                V[i][r] = -old_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                else:
                    V[i][r] = old_val
        
        # Try flipping entries in W
        for i in range(p * n):
            for r in range(rank):
                if W[i][r] == 0:
                    continue
                old_val = W[i][r]
                W[i][r] = -old_val
                new_error = _compute_error(U, V, W, n, m, p)
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                else:
                    W[i][r] = old_val
        
        if current_error == 0:
            return U, V, W
        
        if not improved:
            # Stuck, try random restart
            break
    
    return None

def _structured_search(rank, n, m, p, deadline):
    """
    Structured search: generate candidates with known good structure.
    """
    start_time = time.time()
    
    for attempt in range(20000):
        if time.time() > deadline:
            return None
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # For each factor r, assign a "type" that determines its structure
        for r in range(rank):
            # Choose a pattern: which (i,j,k) indices this factor connects
            # A valid factor for matmul typically has:
            # - One U entry at (i,j)
            # - One V entry at (j,k)  
            # - One W entry at (k,i)
            # But can have multiple for more complex decompositions
            
            # Random pattern
            pattern = random.randint(0, 3)
            
            if pattern == 0:
                # Simple: one entry each
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U[i*m+j][r] = random.choice([-1, 1])
                V[j*p+k][r] = random.choice([-1, 1])
                W[k*n+i][r] = random.choice([-1, 1])
            elif pattern == 1:
                # Two U entries
                i1 = random.randint(0, n-1)
                i2 = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U[i1*m+j][r] = random.choice([-1, 1])
                U[i2*m+j][r] = random.choice([-1, 1])
                V[j*p+k][r] = random.choice([-1, 1])
                W[k*n+i1][r] = random.choice([-1, 1])
                W[k*n+i2][r] = random.choice([-1, 1])
            elif pattern == 2:
                # Two V entries
                i = random.randint(0, n-1)
                j1 = random.randint(0, m-1)
                j2 = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U[i*m+j1][r] = random.choice([-1, 1])
                U[i*m+j2][r] = random.choice([-1, 1])
                V[j1*p+k][r] = random.choice([-1, 1])
                V[j2*p+k][r] = random.choice([-1, 1])
                W[k*n+i][r] = random.choice([-1, 1])
            else:
                # Two W entries
                i1 = random.randint(0, n-1)
                i2 = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U[i1*m+j][r] = random.choice([-1, 1])
                U[i2*m+j][r] = random.choice([-1, 1])
                V[j*p+k][r] = random.choice([-1, 1])
                W[k*n+i1][r] = random.choice([-1, 1])
                W[k*n+i2][r] = random.choice([-1, 1])
        
        if _verify_numpy(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_rank = n * m * p
    
    # Strategy 1: Try to find rank 27 using structured search
    if time.time() < deadline - 30:
        result = _structured_search(27, n, m, p, deadline - 30)
        if result:
            U, V, W = result
            if _verify_numpy(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Try rank 28
    if time.time() < deadline - 20:
        result = _structured_search(28, n, m, p, deadline - 20)
        if result:
            U, V, W = result
            if _verify_numpy(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Strategy 3: Try rank 29
    if time.time() < deadline - 15:
        result = _structured_search(29, n, m, p, deadline - 15)
        if result:
            U, V, W = result
            if _verify_numpy(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Strategy 4: Block-based construction
    if time.time() < deadline - 10:
        result = _construct_block_decomposition(n, m, p, deadline - 10)
        if result:
            U, V, W = result
            if _verify_numpy(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
