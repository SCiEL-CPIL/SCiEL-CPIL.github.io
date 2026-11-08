"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 135  # safety margin before 150s kill
    
    best_rank = n * m * p  # start with schoolbook
    best_result = None
    
    # Schoolbook as fallback
    schoolbook_U = np.zeros((nm, best_rank))
    schoolbook_V = np.zeros((mp, best_rank))
    schoolbook_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j, col] = 1
                schoolbook_V[j * p + k, col] = 1
                schoolbook_W[k * n + i, col] = 1
                col += 1
    best_result = {
        "U": schoolbook_U.tolist(),
        "V": schoolbook_V.tolist(),
        "W": schoolbook_W.tolist()
    }
    
    np.random.seed(42)
    
    def check_solution(U, V, W, tol=1e-9):
        """Check if U, V, W form an exact decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err < tol
    
    def get_max_error(U, V, W):
        """Get the maximum error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def try_exact_recovery(U, V, W, rank):
        """Try to recover an exact half-integer solution from a floating-point solution."""
        # Try direct rounding at various scales
        for scale in [2, 4, 8, 16, 32]:
            U_round = np.round(U * scale) / scale
            V_round = np.round(V * scale) / scale
            W_round = np.round(W * scale) / scale
            if check_solution(U_round, V_round, W_round, 1e-10):
                return U_round, V_round, W_round
        
        # Try rational reconstruction with small denominators
        for denom in [2, 4, 8]:
            U_frac = np.zeros_like(U)
            V_frac = np.zeros_like(V)
            W_frac = np.zeros_like(W)
            for idx in range(U.size):
                U_frac.flat[idx] = round(U.flat[idx] * denom) / denom
            for idx in range(V.size):
                V_frac.flat[idx] = round(V.flat[idx] * denom) / denom
            for idx in range(W.size):
                W_frac.flat[idx] = round(W.flat[idx] * denom) / denom
            
            if check_solution(U_frac, V_frac, W_frac, 1e-10):
                return U_frac, V_frac, W_frac
        
        # Try local search around the solution
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            noise = 0.05 * (trial + 1)
            U_t = np.round((U + np.random.randn(nm, rank) * noise) * 2) / 2
            V_t = np.round((V + np.random.randn(mp, rank) * noise) * 2) / 2
            W_t = np.round((W + np.random.randn(pn, rank) * noise) * 2) / 2
            if check_solution(U_t, V_t, W_t, 1e-10):
                return U_t, V_t, W_t
        
        return None
    
    def als_solve(U, V, W, rank, max_iters=150, tol=1e-12):
        """Run ALS iterations to converge."""
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 8:
                break
            
            # Fix V and W, solve for U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = x
                    except:
                        pass
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = x
                    except:
                        pass
            
            err = get_max_error(U, V, W)
            if err < tol:
                return True
            if abs(prev_err - err) < 1e-14:
                # Stalled, try a small perturbation
                U += np.random.randn(nm, rank) * 0.01
                V += np.random.randn(mp, rank) * 0.01
                W += np.random.randn(pn, rank) * 0.01
            prev_err = err
        
        return False
    
    def kronecker_init(rank):
        """Initialize using Kronecker product structure."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Try to use the structure: decompose indices into sub-indices
        # For 4x5x6, think of 4=2*2, 5=5, 6=2*3
        # Or use binary decomposition: 4=2^2, 6=2*3
        
        # Strategy: for each combination of "sub-indices", create a rank-1 term
        # This is heuristic - try various factorizations
        
        # Simple approach: fill first rank columns with structured patterns
        for c in range(min(rank, 120)):
            # Map column index to (i, j, k) pattern
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            # Add structured entries
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            
            # Add some cross-terms for structure
            if c < rank // 2:
                i2 = (i + 1) % n
                j2 = (j + 1) % m
                k2 = (k + 1) % p
                U[i2 * m + j2, c] += 0.5
                V[j2 * p + k2, c] += 0.5
                W[k2 * n + i2, c] += 0.5
        
        # Add random perturbation to escape local minima
        U += np.random.randn(nm, rank) * 0.1
        V += np.random.randn(mp, rank) * 0.1
        W += np.random.randn(pn, rank) * 0.1
        
        return U, V, W
    
    # Try ranks from lower values upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But matrix multiplication rank is typically much higher
    # For 4x5x6, try from 60 up to 119
    ranks_to_try = list(range(60, 120))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 40:
            break
        
        # Multiple initialization strategies
        inits = []
        
        # 1. Small random
        inits.append(('small_random', 
                      np.random.randn(nm, rank) * 0.3,
                      np.random.randn(mp, rank) * 0.3,
                      np.random.randn(pn, rank) * 0.3))
        
        # 2. Medium random
        inits.append(('med_random',
                      np.random.randn(nm, rank) * 0.6,
                      np.random.randn(mp, rank) * 0.6,
                      np.random.randn(pn, rank) * 0.6))
        
        # 3. Large random
        inits.append(('large_random',
                      np.random.randn(nm, rank) * 1.0,
                      np.random.randn(mp, rank) * 1.0,
                      np.random.randn(pn, rank) * 1.0))
        
        # 4. Kronecker structured
        U_k, V_k, W_k = kronecker_init(rank)
        inits.append(('kronecker', U_k, V_k, W_k))
        
        # 5. Sparse random
        U_s = np.zeros((nm, rank))
        V_s = np.zeros((mp, rank))
        W_s = np.zeros((pn, rank))
        for c in range(rank):
            idx_u = np.random.randint(0, nm)
            idx_v = np.random.randint(0, mp)
            idx_w = np.random.randint(0, pn)
            U_s[idx_u, c] = np.random.choice([-1, 1])
            V_s[idx_v, c] = np.random.choice([-1, 1])
            W_s[idx_w, c] = np.random.choice([-1, 1])
        inits.append(('sparse', U_s, V_s, W_s))
        
        # 6. Binary structure (4=2*2, 6=2*3)
        U_b = np.zeros((nm, rank))
        V_b = np.zeros((mp, rank))
        W_b = np.zeros((pn, rank))
        # Use binary decomposition of indices
        for c in range(min(rank, 120)):
            # Decompose c into binary-like indices
            i1, i2 = divmod(c, 2)
            i1 = i1 % 2
            i = i1 * 2 + i2
            j = (c // 4) % 5
            k1, k2 = divmod(c // 20, 3)
            k = k1 * 3 + k2
            
            if i < n and j < m and k < p:
                U_b[i * m + j, c] = 1.0
                V_b[j * p + k, c] = 1.0
                W_b[k * n + i, c] = 1.0
        
        U_b += np.random.randn(nm, rank) * 0.15
        V_b += np.random.randn(mp, rank) * 0.15
        W_b += np.random.randn(pn, rank) * 0.15
        inits.append(('binary', U_b, V_b, W_b))
        
        for init_name, U_init, V_init, W_init in inits:
            if time.time() > deadline - 10:
                break
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            if als_solve(U, V, W, rank):
                # Found a converging solution!
                result = try_exact_recovery(U, V, W, rank)
                if result is not None:
                    U_round, V_round, W_round = result
                    best_rank = rank
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break  # found for this rank, go to lower ranks
        
        if best_rank < n * m * p:
            break  # found a better rank, stop
    
    if time.time() > deadline - 2:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
