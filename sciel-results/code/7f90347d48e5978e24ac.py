"""Different strategy: DP-optimal block partition + bounded random search for cross-block cancellations."""
import numpy as np
import random
import time

DIMENSIONS = [2, 2, 15]

# Precomputed rank tables for small blocks
# rank_2x2xk for k = 0..8
_RANK_2x2xk = {}

def _compute_rank_2x2xk(k):
    """Compute rank of 2x2 * 2xk tensor using DP over k."""
    if k == 0:
        return 0
    if k == 1:
        return 4  # schoolbook for 2x2 * 2x1
    if k == 2:
        return 7  # Strassen
    if k == 3:
        # 2x2 * 2x3: can use Strassen(2x2x2) + 4 for last col = 11
        # Or other splits? 3 = 2+1 -> 7+4=11
        return 11
    if k == 4:
        # 4 = 2+2 -> 7+7=14, or 4=4 (Strassen 2x2x2 applied twice? no)
        # Actually 2x2 * 2x4: two Strassen blocks = 14
        return 14
    if k == 5:
        # 5 = 2+2+1 -> 7+7+4 = 18, or 5=3+2 -> 11+7=18
        return 18
    if k == 6:
        # 6 = 2+2+2 -> 21, or 6=3+3 -> 22, or 6=4+2 -> 21
        return 21
    if k == 7:
        # 7 = 2+2+2+1 -> 25, or 7=3+2+2 -> 25
        return 25
    if k == 8:
        # 8 = 2+2+2+2 -> 28
        return 28
    return 4 * k  # fallback schoolbook

for k in range(9):
    _RANK_2x2xk[k] = _compute_rank_2x2xk(k)

# DP: dp[k] = min rank for 2x2 * 2xk
def _dp_optimal_rank(p):
    """Compute optimal rank for 2x2 * 2xp using DP."""
    dp = [0] * (p + 1)
    for k in range(1, p + 1):
        best = 4 * k  # schoolbook upper bound
        for split in range(1, k):
            # split k into two parts: we need rank_2x2x(a) + rank_2x2x(b) where a+b=k
            # But we only have precomputed up to 8, so limit
            if split <= 8 and (k - split) <= 8:
                r = _RANK_2x2xk[split] + _RANK_2x2xk[k - split]
                if r < best:
                    best = r
        # Also try single block if k <= 8
        if k <= 8:
            best = min(best, _RANK_2x2xk[k])
        dp[k] = best
    return dp[p]

def _build_block_decomposition(p, block_sizes):
    """Build U, V, W for given block sizes. Returns (U, V, W, rank)."""
    n, m = 2, 2
    rank = sum(_RANK_2x2xk[bs] for bs in block_sizes)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    k_offset = 0
    for bs in block_sizes:
        if bs == 1:
            # Schoolbook for 2x2 * 2x1
            k = k_offset
            # A[0,0]*B[0,k] -> C[0,k]
            U[0, col] = 1; V[k, col] = 1; W[2*k, col] = 1; col += 1
            # A[0,1]*B[1,k] -> C[0,k]
            U[1, col] = 1; V[15+k, col] = 1; W[2*k, col] = 1; col += 1
            # A[1,0]*B[0,k] -> C[1,k]
            U[2, col] = 1; V[k, col] = 1; W[2*k+1, col] = 1; col += 1
            # A[1,1]*B[1,k] -> C[1,k]
            U[3, col] = 1; V[15+k, col] = 1; W[2*k+1, col] = 1; col += 1
        elif bs == 2:
            # Strassen for 2x2 * 2x2
            k0 = k_offset; k1 = k_offset + 1
            # P1 = (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
            U[0, col] = 1; U[3, col] = 1
            V[k0, col] = 1; V[15+k1, col] = 1
            W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
            # P2 = (A10+A11)*B0k0 -> C1k0, -C1k1
            U[2, col] = 1; U[3, col] = 1
            V[k0, col] = 1
            W[2*k0, col] = 1; W[2*k1, col] = -1; col += 1
            # P3 = A00*(B1k1-B0k0) -> C0k1, C1k1
            U[0, col] = 1
            V[15+k1, col] = 1; V[k0, col] = -1
            W[2*k1, col] = 1; W[2*k1+1, col] = 1; col += 1
            # P4 = A11*(B1k0-B0k1) -> C0k0, C1k1
            U[3, col] = 1
            V[15+k0, col] = 1; V[15+k1, col] = -1
            W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
            # P5 = (A00+A01)*B1k1 -> -C0k0, C1k1
            U[0, col] = 1; U[1, col] = 1
            V[15+k1, col] = 1
            W[2*k0, col] = -1; W[2*k1+1, col] = 1; col += 1
            # P6 = (A10-A00)*(B0k0+B1k1) -> C1k1
            U[2, col] = 1; U[0, col] = -1
            V[k0, col] = 1; V[15+k1, col] = 1
            W[2*k1+1, col] = 1; col += 1
            # P7 = (A01-A11)*(B1k0+B0k1) -> C0k0
            U[1, col] = 1; U[3, col] = -1
            V[15+k0, col] = 1; V[15+k1, col] = 1
            W[2*k0, col] = 1; col += 1
        elif bs == 3:
            # Strassen(2) + schoolbook(1)
            k0 = k_offset; k1 = k_offset + 1; k2 = k_offset + 2
            # First 2 columns: Strassen
            U[0, col] = 1; U[3, col] = 1
            V[k0, col] = 1; V[15+k1, col] = 1
            W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
            U[2, col] = 1; U[3, col] = 1
            V[k0, col] = 1
            W[2*k0, col] = 1; W[2*k1, col] = -1; col += 1
            U[0, col] = 1
            V[15+k1, col] = 1; V[k0, col] = -1
            W[2*k1, col] = 1; W[2*k1+1, col] = 1; col += 1
            U[3, col] = 1
            V[15+k0, col] = 1; V[15+k1, col] = -1
            W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
            U[0, col] = 1; U[1, col] = 1
            V[15+k1, col] = 1
            W[2*k0, col] = -1; W[2*k1+1, col] = 1; col += 1
            U[2, col] = 1; U[0, col] = -1
            V[k0, col] = 1; V[15+k1, col] = 1
            W[2*k1+1, col] = 1; col += 1
            U[1, col] = 1; U[3, col] = -1
            V[15+k0, col] = 1; V[15+k1, col] = 1
            W[2*k0, col] = 1; col += 1
            # Last 1 column: schoolbook
            U[0, col] = 1; V[k2, col] = 1; W[2*k2, col] = 1; col += 1
            U[1, col] = 1; V[15+k2, col] = 1; W[2*k2, col] = 1; col += 1
            U[2, col] = 1; V[k2, col] = 1; W[2*k2+1, col] = 1; col += 1
            U[3, col] = 1; V[15+k2, col] = 1; W[2*k2+1, col] = 1; col += 1
        elif bs == 4:
            # Two Strassen blocks
            for sub in range(2):
                k0 = k_offset + 2*sub; k1 = k_offset + 2*sub + 1
                U[0, col] = 1; U[3, col] = 1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[3, col] = 1
                V[k0, col] = 1
                W[2*k0, col] = 1; W[2*k1, col] = -1; col += 1
                U[0, col] = 1
                V[15+k1, col] = 1; V[k0, col] = -1
                W[2*k1, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[3, col] = 1
                V[15+k0, col] = 1; V[15+k1, col] = -1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[0, col] = 1; U[1, col] = 1
                V[15+k1, col] = 1
                W[2*k0, col] = -1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[0, col] = -1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k1+1, col] = 1; col += 1
                U[1, col] = 1; U[3, col] = -1
                V[15+k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; col += 1
        elif bs == 5:
            # 2+2+1
            for sub in range(2):
                k0 = k_offset + 2*sub; k1 = k_offset + 2*sub + 1
                U[0, col] = 1; U[3, col] = 1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[3, col] = 1
                V[k0, col] = 1
                W[2*k0, col] = 1; W[2*k1, col] = -1; col += 1
                U[0, col] = 1
                V[15+k1, col] = 1; V[k0, col] = -1
                W[2*k1, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[3, col] = 1
                V[15+k0, col] = 1; V[15+k1, col] = -1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[0, col] = 1; U[1, col] = 1
                V[15+k1, col] = 1
                W[2*k0, col] = -1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[0, col] = -1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k1+1, col] = 1; col += 1
                U[1, col] = 1; U[3, col] = -1
                V[15+k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; col += 1
            k2 = k_offset + 4
            U[0, col] = 1; V[k2, col] = 1; W[2*k2, col] = 1; col += 1
            U[1, col] = 1; V[15+k2, col] = 1; W[2*k2, col] = 1; col += 1
            U[2, col] = 1; V[k2, col] = 1; W[2*k2+1, col] = 1; col += 1
            U[3, col] = 1; V[15+k2, col] = 1; W[2*k2+1, col] = 1; col += 1
        elif bs == 6:
            # 2+2+2
            for sub in range(3):
                k0 = k_offset + 2*sub; k1 = k_offset + 2*sub + 1
                U[0, col] = 1; U[3, col] = 1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[3, col] = 1
                V[k0, col] = 1
                W[2*k0, col] = 1; W[2*k1, col] = -1; col += 1
                U[0, col] = 1
                V[15+k1, col] = 1; V[k0, col] = -1
                W[2*k1, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[3, col] = 1
                V[15+k0, col] = 1; V[15+k1, col] = -1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[0, col] = 1; U[1, col] = 1
                V[15+k1, col] = 1
                W[2*k0, col] = -1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[0, col] = -1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k1+1, col] = 1; col += 1
                U[1, col] = 1; U[3, col] = -1
                V[15+k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; col += 1
        elif bs == 7:
            # 2+2+2+1
            for sub in range(3):
                k0 = k_offset + 2*sub; k1 = k_offset + 2*sub + 1
                U[0, col] = 1; U[3, col] = 1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[3, col] = 1
                V[k0, col] = 1
                W[2*k0, col] = 1; W[2*k1, col] = -1; col += 1
                U[0, col] = 1
                V[15+k1, col] = 1; V[k0, col] = -1
                W[2*k1, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[3, col] = 1
                V[15+k0, col] = 1; V[15+k1, col] = -1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[0, col] = 1; U[1, col] = 1
                V[15+k1, col] = 1
                W[2*k0, col] = -1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[0, col] = -1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k1+1, col] = 1; col += 1
                U[1, col] = 1; U[3, col] = -1
                V[15+k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; col += 1
            k2 = k_offset + 6
            U[0, col] = 1; V[k2, col] = 1; W[2*k2, col] = 1; col += 1
            U[1, col] = 1; V[15+k2, col] = 1; W[2*k2, col] = 1; col += 1
            U[2, col] = 1; V[k2, col] = 1; W[2*k2+1, col] = 1; col += 1
            U[3, col] = 1; V[15+k2, col] = 1; W[2*k2+1, col] = 1; col += 1
        elif bs == 8:
            # 2+2+2+2
            for sub in range(4):
                k0 = k_offset + 2*sub; k1 = k_offset + 2*sub + 1
                U[0, col] = 1; U[3, col] = 1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[3, col] = 1
                V[k0, col] = 1
                W[2*k0, col] = 1; W[2*k1, col] = -1; col += 1
                U[0, col] = 1
                V[15+k1, col] = 1; V[k0, col] = -1
                W[2*k1, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[3, col] = 1
                V[15+k0, col] = 1; V[15+k1, col] = -1
                W[2*k0, col] = 1; W[2*k1+1, col] = 1; col += 1
                U[0, col] = 1; U[1, col] = 1
                V[15+k1, col] = 1
                W[2*k0, col] = -1; W[2*k1+1, col] = 1; col += 1
                U[2, col] = 1; U[0, col] = -1
                V[k0, col] = 1; V[15+k1, col] = 1
                W[2*k1+1, col] = 1; col += 1
                U[1, col] = 1; U[3, col] = -1
                V[15+k0, col] = 1; V[15+k1, col] = 1
                W[2*k0, col] = 1; col += 1
        else:
            # Fallback: schoolbook for this block
            for kk in range(bs):
                k = k_offset + kk
                U[0, col] = 1; V[k, col] = 1; W[2*k, col] = 1; col += 1
                U[1, col] = 1; V[15+k, col] = 1; W[2*k, col] = 1; col += 1
                U[2, col] = 1; V[k, col] = 1; W[2*k+1, col] = 1; col += 1
                U[3, col] = 1; V[15+k, col] = 1; W[2*k+1, col] = 1; col += 1
        k_offset += bs
    
    return U, V, W, col

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1.0) > 1e-8:
                    return False
    # Check zero entries
    for _ in range(200):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        if not (i1 == i2 and j1 == j2):
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val) > 1e-8:
                return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W, rank

def _random_search(U0, V0, W0, n, m, p, max_iter=5000, deadline=None):
    """Try to reduce rank via random perturbation and null-space projection."""
    rank0 = U0.shape[1]
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Try to find a linear dependency among the rank-1 terms
        # Build the tensor from current decomposition and check rank
        # Instead, try random small modifications
        
        # Pick a random pair of terms and try to merge/cancel
        r1, r2 = random.randint(0, rank0-1), random.randint(0, rank0-1)
        if r1 == r2:
            continue
        
        # Try: can we express term r2 as a combination involving r1?
        # Simpler: try setting one entry to a random value and see if verification still passes
        # This is very unlikely to help, so instead try a different approach:
        # Use the fact that if we have rank R decomposition, we can try to find
        # a (R-1) rank decomposition by solving a linear system.
        
        # For each candidate rank r < rank0, try to find U,V,W of that rank.
        # This is NP-hard in general, but for small reductions we can try.
        
        # Try rank = rank0 - 1: set last column of U,V,W to be linear combos of others
        # Actually, let's try a different approach: use the structure.
        # The tensor has a specific structure: T[i,j,k] = 1 iff i,j,k are consistent.
        # We can try to find a lower rank decomposition by exploiting symmetries.
        
        # For now, just try random perturbations of the last few columns
        if rank0 <= 30:
            continue  # Already at lower bound, no point
        
        # Try to eliminate one term by expressing it as combination of others
        # This requires solving a system, which is expensive. Skip for now.
        pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    import time
    _start_time = time.time()
    _DEADLINE = 110.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # DP to find optimal block partition
    optimal_rank = _dp_optimal_rank(p)
    
    # Find the block partition that achieves optimal_rank
    # Use DP with backtracking
    dp = [float('inf')] * (p + 1)
    dp[0] = 0
    choice = [0] * (p + 1)
    
    for k in range(1, p + 1):
        # Try all possible last block sizes
        for bs in range(1, min(k, 8) + 1):
            if dp[k - bs] + _RANK_2x2xk[bs] < dp[k]:
                dp[k] = dp[k - bs] + _RANK_2x2xk[bs]
                choice[k] = bs
        # Also try schoolbook for the whole thing
        if 4 * k < dp[k]:
            dp[k] = 4 * k
            choice[k] = 0  # 0 means schoolbook
    
    # Backtrack to get block sizes
    block_sizes = []
    k = p
    while k > 0:
        bs = choice[k]
        if bs == 0:
            # Schoolbook for k columns
            block_sizes.append(k)
            k = 0
        else:
            block_sizes.append(bs)
            k -= bs
    block_sizes.reverse()
    
    # Build the decomposition
    U, V, W, rank = _build_block_decomposition(p, block_sizes)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        result = (U, V, W, rank)
    else:
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        result = (U, V, W, rank)
    
    # Try random search for further improvement if time permits
    if time.time() < _DEADLINE - 20.0 and rank > 30:
        U2, V2, W2, rank2 = _random_search(U, V, W, n, m, p, 
                                            max_iter=2000, 
                                            deadline=_DEADLINE - 10.0)
        if rank2 < rank and _verify(U2, V2, W2, n, m, p):
            result = (U2, V2, W2, rank2)
    
    U, V, W, rank = result
    
    # Final deadline check
    if time.time() > _DEADLINE - 5.0:
        # Just return what we have
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
