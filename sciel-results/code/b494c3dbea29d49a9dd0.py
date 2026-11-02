"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _find_2x3x3_decomp(max_attempts=5000, deadline=None):
    """Find a rank-r decomposition for 2x3 x 3x3 matrix multiplication.
    Returns (U, V, W) as lists of lists, or None if not found.
    U shape: (6, r), V shape: (9, r), W shape: (6, r)
    Condition: for all i in {0,1}, j in {0,1,2}, k in {0,1,2}:
        sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c] == 1
    """
    n, m, p = 2, 3, 3
    # Try ranks from 6 upward
    for r in range(6, 12):
        for attempt in range(max_attempts):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Random sparse initialization with values in {-1, 0, 1}
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            for c in range(r):
                # Each term should have a few non-zero entries
                num_nonzero = random.randint(2, 5)
                # Pick distinct positions for U
                u_pos = random.sample(range(n*m), min(num_nonzero, n*m))
                v_pos = random.sample(range(m*p), min(num_nonzero, m*p))
                w_pos = random.sample(range(p*n), min(num_nonzero, p*n))
                for pos in u_pos:
                    U[pos, c] = random.choice([-1, 1])
                for pos in v_pos:
                    V[pos, c] = random.choice([-1, 1])
                for pos in w_pos:
                    W[pos, c] = random.choice([-1, 1])
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(r):
                            total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _find_block_decomp(n, m, p, max_attempts=5000, deadline=None):
    """Find a rank-r decomposition for n x m x p matrix multiplication.
    Returns (U, V, W) as lists of lists, or None if not found.
    U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    """
    for r in range(n*m*p // 2, n*m*p + 1):
        if r < 4:
            r = 4
        for attempt in range(max_attempts):
            if deadline is not None and time.time() > deadline:
                return None
            
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            for c in range(r):
                num_nonzero = random.randint(2, 5)
                u_pos = random.sample(range(n*m), min(num_nonzero, n*m))
                v_pos = random.sample(range(m*p), min(num_nonzero, m*p))
                w_pos = random.sample(range(p*n), min(num_nonzero, p*n))
                for pos in u_pos:
                    U[pos, c] = random.choice([-1, 1])
                for pos in v_pos:
                    V[pos, c] = random.choice([-1, 1])
                for pos in w_pos:
                    W[pos, c] = random.choice([-1, 1])
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(r):
                            total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Strategy: decompose into blocks of 2x3x3
    # Each 2x3x3 block has rank 6, total rank = 18
    block_size = 3  # p = 9 = 3 * 3
    num_blocks = p // block_size
    
    # Find a 2x3x3 decomposition
    decomp = _find_2x3x3_decomp(max_attempts=3000, deadline=deadline - 30)
    
    if decomp is not None:
        U_block, V_block, W_block = decomp
        block_rank = len(U_block[0])
        total_rank = num_blocks * block_rank
        
        # Build full U, V, W
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for g in range(num_blocks):
            # For block g, k ranges from g*block_size to (g+1)*block_size - 1
            for c in range(block_rank):
                col = g * block_rank + c
                # Copy U block (same for all g since A is shared)
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][col] = U_block[i*m+j][c]
                
                # Copy V block for this g
                for j in range(m):
                    for k_local in range(block_size):
                        k_global = g * block_size + k_local
                        V[j*p + k_global][col] = V_block[j*block_size + k_local][c]
                
                # Copy W block for this g
                for k_local in range(block_size):
                    k_global = g * block_size + k_local
                    for i in range(n):
                        W[k_global*n + i][col] = W_block[k_local*n + i][c]
        
        # Verify the full decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
