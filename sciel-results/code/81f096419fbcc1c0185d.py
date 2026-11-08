"""Matrix multiplication (2,3,9) rank search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    # U:(n*m,r) V:(m*p,r) W:(p*n,r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    # exact check: entries should be exactly 1 (within tiny fp error for halves)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _kron_decomposition():
    """
    Kronecker-product based decomposition.
    Models 2x3 * 3x9 as three independent 2x3 * 3x3 block multiplications.
    Each 2x3*3x3 block has schoolbook rank 18.
    Total rank = 3 * 18 = 54.
    """
    n, m, p = DIMENSIONS
    block_size = 3
    num_blocks = p // block_size  # 3
    rank = num_blocks * n * m * block_size  # 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for kp in range(block_size):
                    U[i * m + j, col] = 1
                    V[j * p + (b * block_size + kp), col] = 1
                    W[(b * block_size + kp) * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _structured_search(n, m, p, r, deadline, seed_offset=0):
    """
    Structured search that exploits the block structure of the 3x9 matrix.
    We view the multiplication as A(2x3) * B(3x9) = C(2x9).
    B can be seen as 3 blocks of 3x3: B = [B0 | B1 | B2].
    C = A * B = [A*B0 | A*B1 | A*B2].
    
    Strategy: Try to find low-rank decompositions where each block
    multiplication shares structure.
    """
    random.seed(42 + seed_offset)
    
    # Value sets for entries
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5, 3, -3],
    ]
    
    # Different sparsity patterns
    sparsity_levels = [0.3, 0.5, 0.7]
    
    trials = 2000
    for t in range(trials):
        if time.time() > deadline:
            return None
        
        vs = random.choice(val_sets)
        sp = random.choice(sparsity_levels)
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Fill with structured randomness
        for c in range(r):
            # For U: entries indexed by (i, j) where i in [0,n), j in [0,m)
            for i in range(n):
                for j in range(m):
                    if random.random() < sp:
                        U[i * m + j, c] = random.choice(vs)
            
            # For V: entries indexed by (j, k) where j in [0,m), k in [0,p)
            # Structure: group k into blocks of 3
            for j in range(m):
                for b in range(p // 3):
                    for kp in range(3):
                        k = b * 3 + kp
                        if random.random() < sp:
                            V[j * p + k, c] = random.choice(vs)
            
            # For W: entries indexed by (k, i) where k in [0,p), i in [0,n)
            for b in range(p // 3):
                for kp in range(3):
                    k = b * 3 + kp
                    for i in range(n):
                        if random.random() < sp:
                            W[k * n + i, c] = random.choice(vs)
        
        if _verify(U, V, W, n, m, p, r):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _blockwise_search(n, m, p, r, deadline, seed_offset=0):
    """
    Blockwise search: decompose the 3x9 multiplication into 3 blocks of 3x3.
    For each block, use a low-rank approximation of the 2x3*3x3 multiplication.
    """
    random.seed(99 + seed_offset)
    
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
    ]
    
    num_blocks = p // 3
    rank_per_block = r // num_blocks
    remainder = r % num_blocks
    
    trials = 1500
    for t in range(trials):
        if time.time() > deadline:
            return None
        
        vs = random.choice(val_sets)
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        col = 0
        for b in range(num_blocks):
            rb = rank_per_block + (1 if b < remainder else 0)
            for c in range(rb):
                # For this block, we're doing A(2x3) * B_b(3x3) = C_b(2x3)
                # U entries: A[i,j] for i in [0,2), j in [0,3)
                # V entries: B_b[j,kp] for j in [0,3), kp in [0,3)
                # W entries: C_b[i,kp] for i in [0,2), kp in [0,3)
                
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.6:
                            U[i * m + j, col] = random.choice(vs)
                
                for j in range(m):
                    for kp in range(3):
                        if random.random() < 0.6:
                            V[j * p + (b * 3 + kp), col] = random.choice(vs)
                
                for kp in range(3):
                    for i in range(n):
                        if random.random() < 0.6:
                            W[(b * 3 + kp) * n + i, col] = random.choice(vs)
                
                col += 1
        
        if _verify(U, V, W, n, m, p, r):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_fixed_patterns(n, m, p, r):
    """
    Try some hand-crafted patterns based on known low-rank constructions.
    For 2x3 * 3x9, we can try:
    1. Tensor product structure
    2. Block diagonal with shared factors
    """
    patterns = []
    
    # Pattern 1: Pure Kronecker with shared A factors
    # A is 2x3, B is 3x9 = [B0|B1|B2] each 3x3
    # C = A*B = [A*B0 | A*B1 | A*B2]
    # If we use the same rank-r0 decomposition for each A*Bb, total rank = 3*r0
    
    # Pattern 2: Exploit that n=2 is small
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    # We can factor out some structure
    
    return patterns

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s kill
    
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try Kronecker decomposition first
    kron_result = _kron_decomposition()
    if _verify(np.array(kron_result['U']), np.array(kron_result['V']), 
               np.array(kron_result['W']), n, m, p, len(kron_result['U'][0])):
        kron_rank = len(kron_result['U'][0])
        if kron_rank < best_rank:
            best = kron_result
            best_rank = kron_rank
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    low_bound = max(n * m, m * p, p * n)
    rmax = best_rank
    
    # Search from low_bound up to rmax
    # Use multiple search strategies in parallel (sequentially)
    search_strategies = [
        ("structured", _structured_search),
        ("blockwise", _blockwise_search),
    ]
    
    for r in range(low_bound, rmax + 1):
        if time.time() > deadline:
            break
        
        # Allocate time budget for this rank
        remaining = deadline - time.time()
        time_per_strategy = max(5.0, remaining / (rmax - r + 1) / len(search_strategies))
        
        found = False
        for strategy_name, strategy_func in search_strategies:
            if time.time() > deadline:
                break
            
            # Give each strategy a time slice
            strategy_deadline = min(deadline, time.time() + time_per_strategy)
            result = strategy_func(n, m, p, r, strategy_deadline, seed_offset=r)
            
            if result is not None:
                # Verify and update best
                if _verify(np.array(result['U']), np.array(result['V']), 
                          np.array(result['W']), n, m, p, r):
                    best = result
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found valid decomposition at this rank
            # Since we're iterating ascending, this is the minimum
            return best
    
    return best
# EVOLVE-BLOCK-END
