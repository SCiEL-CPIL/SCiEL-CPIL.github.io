"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # For each (i,j,k), check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(r):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
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
    
    def try_decomposition(r, U_pattern, W_pattern):
        """
        Try to find V given structured U and W patterns.
        U_pattern and W_pattern are functions that generate the matrices.
        Returns (U, V, W) if valid, None otherwise.
        """
        U = U_pattern(r)
        W = W_pattern(r)
        
        # Build linear system: A * V_flat = b
        # Unknowns: V[j*p+k, t] for j in [0,m), k in [0,p), t in [0,r)
        # Equations: n*m*p = 112
        
        A = np.zeros((n * m * p, m * p * r))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return U, V, W
        except:
            pass
        
        return None
    
    def make_mod_pattern(mod_dim):
        """Create a pattern where entry (idx, t) = 1 if idx == t % mod_dim"""
        def pattern(r):
            mat = np.zeros((mod_dim, r))
            for t in range(r):
                idx = t % mod_dim
                mat[idx, t] = 1.0
            return mat
        return pattern
    
    def make_random_pattern(rows, seed=42):
        """Create a random sparse pattern with ~1 non-zero per column"""
        def pattern(r):
            mat = np.zeros((rows, r))
            rng = np.random.RandomState(seed)
            for t in range(r):
                idx = rng.randint(0, rows)
                mat[idx, t] = 1.0
            return mat
        return pattern
    
    def make_kronecker_pattern(n1, n2, r1, r2):
        """Kronecker product pattern for structured decomposition"""
        def pattern(r):
            # Total rows = n1 * n2, columns = r
            # Split r into r1 * r2 if possible
            mat = np.zeros((n1 * n2, r))
            for t in range(r):
                if r % (r1 * r2) != 0:
                    # Fallback to simple mod
                    idx = t % (n1 * n2)
                    mat[idx, t] = 1.0
                else:
                    t1 = (t // r2) % r1
                    t2 = t % r2
                    idx1 = t1 % n1
                    idx2 = t2 % n2
                    idx = idx1 * n2 + idx2
                    mat[idx, t] = 1.0
            return mat
        return pattern
    
    # Try various ranks and patterns
    # Start from higher ranks and work down
    
    # Pattern 1: Simple modular patterns
    U_mod = make_mod_pattern(n * m)  # 16 rows
    W_mod = make_mod_pattern(p * n)  # 28 rows
    
    # Pattern 2: Random sparse patterns
    U_rand = make_random_pattern(n * m, seed=1)
    W_rand = make_random_pattern(p * n, seed=2)
    
    # Pattern 3: Kronecker-like patterns
    U_kron = make_kronecker_pattern(n, m, n, m)
    W_kron = make_kronecker_pattern(p, n, p, n)
    
    # Try different rank values
    best_result = None
    best_rank = n * m * p
    
    for r in range(112, 70, -1):
        # Try modular patterns
        result = try_decomposition(r, U_mod, W_mod)
        if result:
            U, V, W = result
            if r < best_rank:
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try random patterns
        result = try_decomposition(r, U_rand, W_rand)
        if result:
            U, V, W = result
            if r < best_rank:
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try Kronecker patterns
        result = try_decomposition(r, U_kron, W_kron)
        if result:
            U, V, W = result
            if r < best_rank:
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Early stop if we found a good solution
        if best_rank <= 80:
            break
    
    if best_result:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
