"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105  # Safety margin before 120s limit

    def time_remaining():
        return deadline - time.time()

    # Build schoolbook decomposition as fallback (rank 72)
    schoolbook_rank = n * m * p  # 72
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank

    def verify_decomposition_fast(U, V, W):
        """Fast vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        
        # Build the target tensor T[a,b,c] = 1 if (a//m == c%n and a%m == b//p and b%p == c//n)
        # a = i*m + j, b = j*p + k, c = k*n + i
        # Condition: j == j2 (a%m == b//p), k == k2 (b%p == c//n), i == i2 (a//m == c//n)
        
        # Compute the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3rd order tensor contraction
        
        # Efficient way: for each r, add outer product
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
        
        # Build target
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        return np.allclose(T_recon, T_target, atol=1e-9)

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        return verify_decomposition_fast(U, V, W)

    def random_decomposition(target_rank, seed=None):
        """Generate a random sparse decomposition."""
        rng = random.Random(seed)
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        for r in range(target_rank):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            U[i * m + j][r] = 1
            V[j * p + k][r] = 1
            W[k * n + i][r] = 1
        
        return U, V, W

    def structured_decomposition(target_rank, seed=None):
        """
        Generate a structured decomposition that respects the tensor structure.
        Each term (i,j,k) must have U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1 for the
        schoolbook term. We try to find combinations that cancel out.
        """
        rng = random.Random(seed)
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Start with schoolbook terms for all (i,j,k) triples, then randomly
        # assign signs and try to find cancellations
        all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Shuffle and pick target_rank terms with possible coefficient variations
        rng.shuffle(all_triples)
        
        for r in range(min(target_rank, len(all_triples))):
            i, j, k = all_triples[r]
            # Use coefficients from {-1, 0, 1} or {-2, -1, 0, 1, 2}
            coeff = rng.choice([-2, -1, 1, 2])
            U[i * m + j][r] = coeff
            V[j * p + k][r] = coeff
            W[k * n + i][r] = coeff
        
        # Fill remaining terms with random assignments
        for r in range(len(all_triples), target_rank):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            coeff = rng.choice([-1, 1])
            U[i * m + j][r] = coeff
            V[j * p + k][r] = coeff
            W[k * n + i][r] = coeff
        
        return U, V, W

    def local_search_improve(U, V, W, max_iter=500, seed=None):
        """
        Try to improve the decomposition by adjusting coefficients.
        Uses a gradient-free approach: try small perturbations and keep if valid.
        """
        rng = random.Random(seed)
        rank = len(U[0])
        current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Try modifying a random entry
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randint(0, n * m - 1)
                col = rng.randint(0, rank - 1)
                old_val = current_U[row][col]
                delta = rng.choice([-1, 1])
                current_U[row][col] = old_val + delta
            elif mat == 'V':
                row = rng.randint(0, m * p - 1)
                col = rng.randint(0, rank - 1)
                old_val = current_V[row][col]
                delta = rng.choice([-1, 1])
                current_V[row][col] = old_val + delta
            else:
                row = rng.randint(0, p * n - 1)
                col = rng.randint(0, rank - 1)
                old_val = current_W[row][col]
                delta = rng.choice([-1, 1])
                current_W[row][col] = old_val + delta
            
            # Check if still valid
            if verify_decomposition_fast(current_U, current_V, current_W):
                # Keep the change
                pass
            else:
                # Revert
                if mat == 'U':
                    current_U[row][col] = old_val
                elif mat == 'V':
                    current_V[row][col] = old_val
                else:
                    current_W[row][col] = old_val
        
        return current_U, current_V, current_W

    # Try various approaches to find a low-rank decomposition
    
    # Approach 1: Try structured decompositions with different ranks
    for target_rank in range(32, 73, 4):
        if time_remaining() < 15:
            break
        
        for attempt in range(50):
            if time_remaining() < 10:
                break
            
            seed = attempt * 10000 + target_rank
            U, V, W = structured_decomposition(target_rank, seed)
            
            if verify_decomposition_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Approach 2: Try random decompositions
    if best_rank == schoolbook_rank:
        for target_rank in range(48, 73, 4):
            if time_remaining() < 15:
                break
            
            for attempt in range(100):
                if time_remaining() < 10:
                    break
                
                seed = attempt * 10000 + target_rank + 7777
                U, V, W = random_decomposition(target_rank, seed)
                
                if verify_decomposition_fast(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
            
            if best_rank < schoolbook_rank:
                break
    
    # Approach 3: Local search refinement from schoolbook
    if best_rank == schoolbook_rank and time_remaining() > 20:
        U_cur, V_cur, W_cur = [row[:] for row in U_school], [row[:] for row in V_school], [row[:] for row in W_school]
        
        # Try to zero out some terms and compensate
        for r_remove in range(schoolbook_rank - 1, -1, -1):
            if time_remaining() < 10:
                break
            
            # Try removing term r_remove
            U_test = [row[:] for row in U_cur]
            V_test = [row[:] for row in V_cur]
            W_test = [row[:] for row in W_cur]
            U_test[r_remove if r_remove < len(U_test) else -1] = [0] * best_rank
            
            # This is getting complex; skip for now
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
