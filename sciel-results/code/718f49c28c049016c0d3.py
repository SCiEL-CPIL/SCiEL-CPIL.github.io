import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 7]

def _matmul_check(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if total != 1:
                    return False
    return True

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

def _try_rank_linear_solve(R, n, m, p, time_limit, seed=42):
    """
    Try to find a rank-R decomposition by fixing U, V randomly and solving for W.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Indices
    # U: (i, j) -> i*m+j
    # V: (j, k) -> j*p+k
    # W: (k, i) -> k*n+i
    
    # Number of equations: n*m*p
    # Number of unknowns: p*n*R
    
    num_eqs = n * m * p
    num_unk = p * n * R
    
    # We will try several random initializations of U and V
    max_attempts = 10000
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate random U and V
        # Use entries in {-1, 0, 1}
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        
        # Ensure each column r has some non-zeros
        for r in range(R):
            # Pick random non-zero positions
            # For U, pick 1-2 entries
            # For V, pick 1-2 entries
            u_nonzeros = rng.randint(1, 2)
            u_indices = rng.sample(range(n*m), u_nonzeros)
            for idx in u_indices:
                U[idx][r] = rng.choice([-1, 1])
                
            v_nonzeros = rng.randint(1, 2)
            v_indices = rng.sample(range(m*p), v_nonzeros)
            for idx in v_indices:
                V[idx][r] = rng.choice([-1, 1])
                
        # Build the linear system A x = b
        # x is vector of W entries: x[k*n+i, r]
        # A has shape (num_eqs, num_unk)
        # To save memory and time, we can use sparse representation or just dense since 42 x (14*R) is small.
        # Max R is 42, so 42 x 588. Very small.
        
        A = np.zeros((num_eqs, num_unk), dtype=float)
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(R):
                        unk_idx = (k * n + i) * R + r
                        coeff = U[i*m+j][r] * V[j*p+k][r]
                        if coeff != 0:
                            A[eq_idx, unk_idx] = coeff
        
        # Solve A x = b
        try:
            # Use least squares or direct solve
            # Since A is likely rank deficient or inconsistent, we should check.
            # But we want an exact solution.
            # If the system is consistent, we can use np.linalg.lstsq or solve.
            # However, we need exact rational solution.
            # Let's use Fraction-based solver if possible, or check if float solution is close to rational.
            
            # Since the system is small, we can use sympy? No, standard library + numpy + scipy.
            # We can use np.linalg.solve if square, but it's underdetermined (42 eqs, 14R unks).
            # If R=42, 42 eqs, 588 unks.
            # We want a solution with small integer/rational entries.
            
            # Let's find the minimum norm solution or a particular solution.
            # np.linalg.lstsq gives minimum norm solution.
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if residuals are small
            if np.linalg.norm(b - A.dot(x)) > 1e-9:
                continue
                
            # Check if x has "nice" values
            # Convert to Fractions and check denominators
            valid = True
            W_frac = [[Fraction(0) for _ in range(R)] for _ in range(p*n)]
            
            for idx in range(num_unk):
                val = x[idx]
                # Check if val is close to a fraction with small denominator
                # Try denominators 1, 2, 3, 4, 6
                found = False
                for d in [1, 2, 3, 4, 6, 12]:
                    num = round(val * d)
                    if abs(val - num/d) < 1e-6:
                        W_frac[idx // R][idx % R] = Fraction(num, d)
                        found = True
                        break
                if not found:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Verify the solution exactly
            if _matmul_check(U, V, W_frac, n, m, p):
                return {"U": U, "V": V, "W": W_frac}
                
        except np.linalg.LinAlgError:
            continue
            
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 42
    
    # Try to find lower ranks using the linear solve method
    # Start from 41 down to 30
    for r in range(41, 29, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        # Use different seeds for each rank
        result = _try_rank_linear_solve(r, n, m, p, time_limit=remaining, seed=r*1000 + 1)
        if result is not None:
            best = result
            best_rank = r
            break  # Found lower rank, stop
    
    return best
# EVOLVE-BLOCK-END
