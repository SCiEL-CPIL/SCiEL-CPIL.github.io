import time
import random
import numpy as np
from fractions import Fraction

def _build_schoolbook(n, m, p):
    """Build the standard rank-n*m*p decomposition."""
    U = [[0]*(n*m*p) for _ in range(n*m)]
    V = [[0]*(n*m*p) for _ in range(m*p)]
    W = [[0]*(n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n=2, m=2, p=7, tol=1e-9):
    """Verify tensor identity: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all (i,j,k)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _residual(U, V, W, n=2, m=2, p=7):
    """Compute sum of squared residuals of the tensor identity."""
    rank = len(U[0])
    sse = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                sse += (s - 1.0)**2
    return sse

def _try_small_ints(rank, n, m, p, deadline, rng, max_time=30.0, max_range=3):
    """
    Try to find an exact decomposition with entries in small integers.
    Uses structured search: fix one matrix, solve for others via constraints.
    """
    start = time.time()
    target = rank
    
    # Strategy: Use the fact that for small n,m,p, we can try to find
    # decompositions with entries in {-1, 0, 1, 2, -2}
    # We use a randomized approach with local search.
    
    for attempt in range(200):
        if time.time() > deadline - 15:
            break
        if time.time() - start > max_time:
            break
        
        # Initialize with random small integers
        low, high = -max_range, max_range
        U = [[rng.randint(low, high) for _ in range(target)] for _ in range(n*m)]
        V = [[rng.randint(low, high) for _ in range(target)] for _ in range(m*p)]
        W = [[rng.randint(low, high) for _ in range(target)] for _ in range(p*n)]
        
        # Local improvement: try to fix residuals by adjusting entries
        for _ in range(100):
            if time.time() > deadline - 15:
                break
            sse = _residual(U, V, W, n, m, p)
            if sse < 1e-10:
                # Verify exactly
                if _verify(U, V, W, n, m, p):
                    return U, V, W
            
            # Pick a random equation (i,j,k) with largest residual
            residuals = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        residuals.append((abs(s-1.0), i, j, k, s))
            residuals.sort(reverse=True)
            
            if not residuals:
                break
            _, i, j, k, s = residuals[0]
            
            # Try to adjust a random entry in U[i*m+j] to reduce residual
            r = rng.randint(0, target-1)
            old_val = U[i*m+j][r]
            # Compute partial derivative-like: changing U[i*m+j][r] by delta changes s by delta * V[j*p+k][r] * W[k*n+i][r]
            coeff = V[j*p+k][r] * W[k*n+i][r]
            if coeff != 0:
                # We want s + delta * coeff = 1, so delta = (1 - s) / coeff
                delta = (1.0 - s) / coeff
                # Try nearby integer values
                for d in [int(delta), int(delta)+1, int(delta)-1, 0, 1, -1]:
                    new_val = old_val + d
                    U[i*m+j][r] = new_val
                    new_sse = _residual(U, V, W, n, m, p)
                    if new_sse < sse:
                        sse = new_sse
                        break
                    else:
                        U[i*m+j][r] = old_val
                else:
                    U[i*m+j][r] = old_val
    
    return None

def _try_fractional(rank, n, m, p, deadline, rng, max_time=40.0):
    """
    Try to find a decomposition with fractional (half-integer) entries.
    Uses a more structured approach: build from schoolbook and try to merge.
    """
    start = time.time()
    
    # Start with schoolbook decomposition
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    sb_rank = n * m * p  # 28
    
    # Try to merge pairs of terms to reduce rank by 1
    # For each pair (a, b), check if we can represent their sum as a single term
    # This requires solving for u, v, w such that:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] = u_a[i*m+j]*v_a[j*p+k]*w_a[k*n+i] + u_b[i*m+j]*v_b[j*p+k]*w_b[k*n+i]
    # for all (i,j,k)
    
    # A simpler approach: use random initialization with fractional values
    # and use Newton-like iterations
    
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
        if time.time() - start > max_time:
            break
        
        # Initialize with values in {-2, -1, -0.5, 0, 0.5, 1, 2}
        vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
        U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)]
        
        # Gradient descent with discrete steps
        best_sse = _residual(U, V, W, n, m, p)
        
        for _ in range(200):
            if time.time() > deadline - 20:
                break
            
            sse = _residual(U, V, W, n, m, p)
            if sse < 1e-10:
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    # Round to half-integers
                    U = [[round(x * 2) / 2 for x in row] for row in U]
                    V = [[round(x * 2) / 2 for x in row] for row in V]
                    W = [[round(x * 2) / 2 for x in row] for row in W]
                    if _verify(U, V, W, n, m, p, tol=1e-8):
                        return U, V, W
                break
            
            # Pick random matrix and entry to modify
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = rng.randint(0, n*m-1)
                row = rng.randint(0, rank-1)
                old = U[idx][row]
            elif mat == 'V':
                idx = rng.randint(0, m*p-1)
                row = rng.randint(0, rank-1)
                old = V[idx][row]
            else:
                idx = rng.randint(0, p*n-1)
                row = rng.randint(0, rank-1)
                old = W[idx][row]
            
            # Try changing to a nearby half-integer
            new_val = old + rng.choice([-1.0, -0.5, 0.5, 1.0])
            
            if mat == 'U':
                U[idx][row] = new_val
            elif mat == 'V':
                V[idx][row] = new_val
            else:
                W[idx][row] = new_val
            
            new_sse = _residual(U, V, W, n, m, p)
            if new_sse < sse:
                pass  # Keep the change
            else:
                # Revert
                if mat == 'U':
                    U[idx][row] = old
                elif mat == 'V':
                    V[idx][row] = old
                else:
                    W[idx][row] = old
    
    return None

def _try_strassen_like(rank, n, m, p, deadline, rng, max_time=30.0):
    """
    Try to find a decomposition inspired by Strassen's algorithm.
    For 2x2 * 2x7, we can think of it as 7 separate 2x2 * 2x1 products,
    but share some computations.
    """
    start = time.time()
    
    # For 2x2 * 2x1, Strassen gives rank 7 instead of 8.
    # For 2x2 * 2x7, if we could share across the 7 columns,
    # we might get rank < 28.
    
    # Let's try a structured approach:
    # A = [[a, b], [c, d]], B = [[e1, f1, e2, f2, ..., e7, f7]]
    # C = A * B = [[a*e1+b*e2+...+a*e7+b*f7? No...
    # Actually C[i][k] = sum_j A[i][j] * B[j][k]
    # So C[0][k] = a*B[0][k] + b*B[1][k]
    #    C[1][k] = c*B[0][k] + d*B[1][k]
    
    # This is just 2*2*7 = 28 independent multiplications in the schoolbook sense,
    # but we can share bilinear forms.
    
    # One approach: use the identity (a+b)(c+d) = ac + ad + bc + bd
    # to share computations.
    
    # For now, let's try a random search with a bias toward structured solutions.
    for attempt in range(100):
        if time.time() > deadline - 20:
            break
        if time.time() - start > max_time:
            break
        
        # Initialize with a structured pattern
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # Fill with random values but ensure some structure
        for r in range(rank):
            # Each term should contribute to a subset of entries
            # Randomly assign which (i,j,k) this term covers
            num_terms = rng.randint(1, 4)
            for _ in range(num_terms):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j][r] = rng.choice([-1, 0, 1])
                V[j*p+k][r] = rng.choice([-1, 0, 1])
                W[k*n+i][r] = rng.choice([-1, 0, 1])
        
        # Local search
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            sse = _residual(U, V, W, n, m, p)
            if sse < 1e-10:
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    return U, V, W
                break
            
            # Modify a random entry
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = rng.randint(0, n*m-1)
                row = rng.randint(0, rank-1)
                old = U[idx][row]
                U[idx][row] = rng.choice([-2, -1, 0, 1, 2])
                new_sse = _residual(U, V, W, n, m, p)
                if new_sse >= sse:
                    U[idx][row] = old
            elif mat == 'V':
                idx = rng.randint(0, m*p-1)
                row = rng.randint(0, rank-1)
                old = V[idx][row]
                V[idx][row] = rng.choice([-2, -1, 0, 1, 2])
                new_sse = _residual(U, V, W, n, m, p)
                if new_sse >= sse:
                    V[idx][row] = old
            else:
                idx = rng.randint(0, p*n-1)
                row = rng.randint(0, rank-1)
                old = W[idx][row]
                W[idx][row] = rng.choice([-2, -1, 0, 1, 2])
                new_sse = _residual(U, V, W, n, m, p)
                if new_sse >= sse:
                    W[idx][row] = old
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for the 150s limit
    
    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28
    
    # 1. Construct Schoolbook (Rank 28) as guaranteed fallback
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    rng = random.Random(42)
    
    # 2. Try rank 27 with various methods
    for target_rank in [27, 26, 25]:
        if best_rank <= target_rank:
            continue
        if time.time() > deadline - 30:
            break
        
        # Method 1: Small integer search
        result = _try_small_ints(target_rank, n, m, p, deadline, rng, max_time=15.0)
        if result:
            best_U, best_V, best_W = result
            best_rank = target_rank
            continue
        
        # Method 2: Fractional search
        if time.time() > deadline - 25:
            break
        result = _try_fractional(target_rank, n, m, p, deadline, rng, max_time=15.0)
        if result:
            best_U, best_V, best_W = result
            best_rank = target_rank
            continue
        
        # Method 3: Structured search
        if time.time() > deadline - 20:
            break
        result = _try_strassen_like(target_rank, n, m, p, deadline, rng, max_time=15.0)
        if result:
            best_U, best_V, best_W = result
            best_rank = target_rank
    
    # Final verification
    if best_rank < schoolbook_rank:
        if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
            # Fallback to schoolbook if verification fails
            best_U, best_V, best_W = _build_schoolbook(n, m, p)
            best_rank = schoolbook_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
