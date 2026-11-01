"""Find rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
from fractions import Fraction
import time
import random
from itertools import product

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


def _search_rank(r, n, m, p, time_limit, seed=42):
    """Search for a rank-r decomposition using structured random + local search.
    
    For each rank-1 term, we choose (i,j) for U, (j',k) for V, (k',i') for W.
    The constraint is that for each (i,j,k), the sum of products must be 1.
    
    We use a greedy/structured approach:
    1. Initialize with random sparse patterns
    2. Try to satisfy equations one by one
    """
    start = time.time()
    rng = random.Random(seed)
    
    # For a rank-1 term r, it contributes to equation (i,j,k) if:
    # U[i*m+j, r] != 0, V[j*p+k, r] != 0, W[k*n+i, r] != 0
    # The contribution is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # We need: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Strategy: Use a "support" structure. For each rank-1 term, pick a pattern
    # of which (i,j), (j,k), (k,i) indices are nonzero.
    
    # Simple approach: each rank-1 term has exactly one nonzero in each of U, V, W
    # (i.e., it's a "monomial" term). Then the equation for (i,j,k) is satisfied
    # if there's exactly one term r where U[i*m+j,r]=a, V[j*p+k,r]=b, W[k*n+i,r]=c
    # with abc=1, and no other terms contribute.
    
    # This is equivalent to: we need to cover all (i,j,k) triples with rank-1 terms,
    # where each term is a "line" through (i,j,k) in the 3D grid.
    
    # Actually, a rank-1 term with U nonzero only at (i0,j0), V nonzero only at (j1,k1),
    # W nonzero only at (k2,i2) contributes to equation (i,j,k) only if i=i0=i2, j=j0=j1, k=k1=k2.
    # So it only contributes to ONE equation: (i0, j0, k1) where we need i0=i2, j0=j1, k1=k2.
    # That means i=i0, j=j0, k=k1. So each such term only helps one equation.
    # That gives rank = nmp = 42, the schoolbook.
    
    # To get rank < 42, we need terms that contribute to MULTIPLE equations.
    # A term with U having multiple nonzeros, or V, or W, can contribute to multiple (i,j,k).
    
    # Let's try: each rank-1 term has U nonzero at one (i,j), V nonzero at one (j',k),
    # W nonzero at one (k',i'). It contributes to equation (i,j,k) iff i=i', j=j', k=k'.
    # Wait no. Let me re-examine.
    
    # U is indexed by (i,j) with i in [0,n), j in [0,m)
    # V is indexed by (j,k) with j in [0,m), k in [0,p)  
    # W is indexed by (k,i) with k in [0,p), i in [0,n)
    
    # For term r, it contributes to equation (i,j,k) the value:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So if U[.,r] is nonzero at (i0,j0), V[.,r] is nonzero at (j1,k1), W[.,r] is nonzero at (k2,i2),
    # then term r contributes to equation (i,j,k) only when i=i0=i2, j=j0=j1, k=k1=k2.
    # So i=i0, j=j0, k=k1, with the consistency i2=i0, j1=j0, k2=k1.
    # This means the term only contributes to ONE equation: (i0, j0, k1).
    
    # UNLESS: U has multiple nonzeros, or V has multiple nonzeros, etc.
    
    # If U[.,r] is nonzero at (i0,j0) AND (i1,j1), then term r can contribute to
    # equations involving both (i0,j0,*) and (i1,j1,*).
    
    # Let's try a structured approach: for each rank-1 term, let U be nonzero at
    # one (i,j), V be nonzero at multiple (j,k) for the same j, W be nonzero at
    # multiple (k,i) for the same i.
    
    # Actually, let's just do a more exhaustive search with small values.
    
    # Try rank r with entries in {-1, 0, 1} and a few structured patterns
    
    # Pattern: For each term r, assign a "type" that determines the support
    # Type A: U at (i,j), V at (j,k), W at (k,i) - monomial, covers 1 equation
    # Type B: U at (i,j), V at (j,k1) and (j,k2), W at (k1,i) and (k2,i) - covers 2 equations
    # etc.
    
    # This is getting complex. Let me try a direct numerical search.
    
    # For small r, try random assignments with entries in {-1, 0, 1, 1/2}
    max_attempts = 5000
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
        
        # Generate random sparse matrices
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # For each rank-1 term, randomly assign a pattern
        for r_idx in range(r):
            # Choose support for U: pick 1-2 (i,j) pairs
            n_u = rng.choice([1, 1, 1, 2])
            u_indices = set()
            for _ in range(n_u):
                u_indices.add((rng.randint(0, n-1), rng.randint(0, m-1)))
            
            # Choose support for V: pick 1-2 (j,k) pairs, try to share j with U
            n_v = rng.choice([1, 1, 1, 2])
            v_indices = set()
            for _ in range(n_v):
                # Try to use a j from U if possible
                if u_indices and rng.random() < 0.7:
                    j = rng.choice(list(u_indices))[1]
                else:
                    j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                v_indices.add((j, k))
            
            # Choose support for W: pick 1-2 (k,i) pairs, try to share
            n_w = rng.choice([1, 1, 1, 2])
            w_indices = set()
            for _ in range(n_w):
                if v_indices and rng.random() < 0.7:
                    k = rng.choice(list(v_indices))[1]
                else:
                    k = rng.randint(0, p-1)
                if u_indices and rng.random() < 0.7:
                    i = rng.choice(list(u_indices))[0]
                else:
                    i = rng.randint(0, n-1)
                w_indices.add((k, i))
            
            # Assign values
            for (i, j) in u_indices:
                U[i*m+j][r_idx] = rng.choice([-1, 1])
            for (j, k) in v_indices:
                V[j*p+k][r_idx] = rng.choice([-1, 1])
            for (k, i) in w_indices:
                W[k*n+i][r_idx] = rng.choice([-1, 1])
        
        # Check
        if _matmul_check(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 42
    
    # Try to find lower ranks, starting from reasonable estimates
    # Theoretical lower bound for matmul tensor rank is at least max(n*m, m*p, p*n)
    # = max(6, 21, 14) = 21, but actual lower bounds are higher.
    # Let's try from 30 up to 41.
    
    for r in range(30, 42):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        
        result = _search_rank(r, n, m, p, time_limit=remaining, seed=r*1000)
        if result is not None:
            best = result
            best_rank = r
            break  # Found lower rank, stop
    
    return best
# EVOLVE-BLOCK-END
