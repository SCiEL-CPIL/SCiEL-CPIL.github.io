"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    def to_list(mat):
        return mat.tolist()
    
    def verify_exact(U, V, W):
        """Verify that U, V, W correctly decompose the tensor using exact arithmetic."""
        rank = len(U[0]) if U else 0
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check all n*m*p required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = Fraction(0)
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if actual != Fraction(1):
                        return False
        
        # Check that all other entries are zero
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                if j1 != j2:
                    actual = Fraction(0)
                    for r in range(rank):
                        actual += U[a][r] * V[b][r] * W[k1 * n + i1][r]
                    if actual != Fraction(0):
                        return False
                else:
                    for k2 in range(p):
                        c = k2 * n + i1
                        if k1 != k2:
                            actual = Fraction(0)
                            for r in range(rank):
                                actual += U[a][r] * V[b][r] * W[c][r]
                            if actual != Fraction(0):
                                return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using float, for initial screening."""
        rank = len(U[0]) if U else 0
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j1 == j2 and i1 == i2 and k1 == k2):
                actual = 0.0
                for r in range(rank):
                    actual += U[a][r] * V[b][r] * W[c][r]
                if abs(actual) > 1e-9:
                    return False
        return True
    
    # Build schoolbook decomposition
    U_sb = [[Fraction(0)] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[Fraction(0)] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[Fraction(0)] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = Fraction(1)
                V_sb[j * p + k][col] = Fraction(1)
                W_sb[k * n + i][col] = Fraction(1)
                col += 1
    
    # Try to find rank reduction using structured search
    # Key insight: For fixed j, we have n*p = 14 terms
    # We can try to find linear dependencies
    
    def try_rank_reduction():
        """Try to find a rank-41 or lower decomposition."""
        if time.time() > deadline:
            return None
        
        # Strategy: Try to merge pairs of terms with the same j
        # For terms (i1, j, k1) and (i2, j, k2), we can potentially combine them
        # if we can find a rank-1 tensor that covers both
        
        # Let's try a different approach: use the structure more carefully
        # The tensor is sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
        
        # For m=3, we can try to use the fact that there are only 3 values of j
        # Group terms by j:
        # j=0: 14 terms (i=0,1; k=0..6)
        # j=1: 14 terms
        # j=2: 14 terms
        
        # Within each j-group, can we reduce the rank?
        # For fixed j, the terms are (i, k) pairs
        # A rank-1 term in this group would need to cover multiple (i,k) pairs
        
        # Let's try: for fixed j, can we cover all 14 (i,k) pairs with fewer than 14 rank-1 terms?
        # This is equivalent to finding the rank of a 2×7 matrix (for fixed j)
        # But the constraint is that the rank-1 terms must also be consistent with other j groups
        
        # Actually, let's try a more direct approach:
        # Start with schoolbook, and try to perturb it to find a lower rank decomposition
        
        # Use random search with half-integer entries
        half_ints = [Fraction(x, 2) for x in range(-6, 7)]  # -3, -2.5, ..., 3
        
        def random_half_int():
            return np.random.choice(half_ints)
        
        # Try rank 41, 40, 39, 38
        for target_rank in [41, 40, 39, 38]:
            if time.time() > deadline:
                break
            
            max_attempts = 500 if target_rank >= 40 else 200
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                U = [[Fraction(0)] * target_rank for _ in range(n * m)]
                V = [[Fraction(0)] * target_rank for _ in range(m * p)]
                W = [[Fraction(0)] * target_rank for _ in range(p * n)]
                
                # Initialize with sparse structure
                for r_idx in range(target_rank):
                    num_u = np.random.randint(1, 4)
                    num_v = np.random.randint(1, 4)
                    num_w = np.random.randint(1, 4)
                    
                    u_pos = np.random.choice(n * m, size=num_u, replace=False)
                    v_pos = np.random.choice(m * p, size=num_v, replace=False)
                    w_pos = np.random.choice(p * n, size=num_w, replace=False)
                    
                    for pos in u_pos:
                        U[pos][r_idx] = random_half_int()
                    for pos in v_pos:
                        V[pos][r_idx] = random_half_int()
                    for pos in w_pos:
                        W[pos][r_idx] = random_half_int()
                
                # Quick float check
                if verify_fast(U, V, W):
                    # Exact check
                    if verify_exact(U, V, W):
                        return U, V, W
        
        return None
    
    result = try_rank_reduction()
    if result is not None:
        U, V, W = result
        # Convert to lists of floats
        U_out = [[float(x) for x in row] for row in U]
        V_out = [[float(x) for x in row] for row in V]
        W_out = [[float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Return schoolbook
    U_out = [[float(x) for x in row] for row in U_sb]
    V_out = [[float(x) for x in row] for row in V_sb]
    W_out = [[float(x) for x in row] for row in W_sb]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
