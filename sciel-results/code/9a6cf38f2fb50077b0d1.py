"""Combined program for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 5, 6]

def _check_exact(U, V, W, n, m, p, timeout=30):
    """Check if the decomposition is exact using random integer tests."""
    start = time.time()
    for trial in range(200):
        if time.time() - start > timeout:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # True product
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Decomposed product
        C_dec = [[0] * p for _ in range(n)]
        for c in range(len(U[0])):
            a_col = [U[i * m + j][c] for i in range(n) for j in range(m)]
            b_col = [V[j * p + k][c] for j in range(m) for k in range(p)]
            w_col = [W[k * n + i][c] for i in range(n) for k in range(p)]
            # Sum over c
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_dec[i][k] += a_col[i * m + j] * b_col[j * p + k] * w_col[k * n + i]
        for i in range(n):
            for k in range(p):
                if C_dec[i][k] != C_true[i][k]:
                    return False
    return True

def _verify_rank_numpy(U, V, W, n=2, m=5, p=6, tol=1e-6):
    """Check if given factor matrices produce correct matmul tensor using numpy."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check on-diagonal terms: T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some off-diagonal terms to ensure they are 0
    checked = 0
    for a in range(n*m):
        for b in range(m*p):
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k != k2 or i != i2:
                    val = 0.0
                    for cc in range(rank):
                        val += U_np[a, cc] * V_np[b, cc] * W_np[c, cc]
                    if abs(val) > tol:
                        return False
                    checked += 1
                    if checked > 5000:
                        break
            if checked > 5000:
                break
        if checked > 5000:
            break
    return True

def _schoolbook():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _rationalize(val, max_denom=8):
    """Try to convert a float to a Fraction with small denominator."""
    if isinstance(val, Fraction):
        return val
    if abs(val) < 1e-10:
        return Fraction(0)
    # Check if it's close to a simple fraction
    for d in range(1, max_denom + 1):
        num = round(val * d)
        if abs(val - num / d) < 1e-8:
            return Fraction(num, d)
    return Fraction(val)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s out of 120s budget
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Strategy: Use numpy for fast verification, try structured random search
    # Focus on ranks near the lower bound
    
    def try_rank_structured(r, max_attempts=50):
        """Try to construct a rank-r decomposition via structured random search."""
        if r >= best_rank:
            return None
        
        # Basic lower bound
        lower_bound = max(n*m, m*p, n*p)
        if r < lower_bound:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random small integers/half-integers
            # Use a mix of 0, ±1, ±1/2
            choices = [0, 1, -1, Fraction(1,2), Fraction(-1,2), Fraction(1,4), Fraction(-1,4)]
            
            U = [[random.choice(choices) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(choices) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(choices) for _ in range(r)] for _ in range(p*n)]
            
            # Quick numpy check
            if _verify_rank_numpy(U, V, W, n, m, p, tol=1e-4):
                # Full exact check
                if _check_exact(U, V, W, n, m, p, timeout=10):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_rank_perturbed(r, base_result, max_attempts=30):
        """Try to construct a rank-r decomposition by perturbing a base decomposition."""
        if r >= best_rank:
            return None
        
        # Start from a schoolbook-like base and try to merge columns
        U_base = [[0] * (n*m*p) for _ in range(n*m)]
        V_base = [[0] * (n*m*p) for _ in range(m*p)]
        W_base = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_base[i*m+j][col] = 1
                    V_base[j*p+k][col] = 1
                    W_base[k*n+i][col] = 1
                    col += 1
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Try to create a rank-r decomposition by combining columns
            # This is a heuristic approach
            U = [[0] * r for _ in range(n*m)]
            V = [[0] * r for _ in range(m*p)]
            W = [[0] * r for _ in range(p*n)]
            
            # Fill with random fractions
            choices = [0, 1, -1, Fraction(1,2), Fraction(-1,2)]
            for row in range(n*m):
                for col_idx in range(r):
                    U[row][col_idx] = random.choice(choices)
            for row in range(m*p):
                for col_idx in range(r):
                    V[row][col_idx] = random.choice(choices)
            for row in range(p*n):
                for col_idx in range(r):
                    W[row][col_idx] = random.choice(choices)
            
            # Quick numpy check
            if _verify_rank_numpy(U, V, W, n, m, p, tol=1e-4):
                # Full exact check
                if _check_exact(U, V, W, n, m, p, timeout=10):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks in a strategic order
    # Focus on ranks that might be achievable
    ranks_to_try = []
    
    # Try near the lower bound first, then work up
    lower_bound = max(n*m, m*p, n*p)  # 30
    for r in range(lower_bound, min(schoolbook_rank, lower_bound + 20)):
        ranks_to_try.append(r)
    
    # Also try some ranks near schoolbook
    for r in range(schoolbook_rank - 1, max(lower_bound, schoolbook_rank - 10), -1):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    
    # Sort by likelihood of success (try lower ranks first)
    ranks_to_try.sort()
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try structured search
        result = try_rank_structured(r, max_attempts=20)
        if result is not None:
            best_rank = r
            best_result = result
            break
        
        # Try perturbed search
        result = try_rank_perturbed(r, best_result, max_attempts=10)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
