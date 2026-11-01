"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses the schoolbook decomposition (rank 50) as the guaranteed baseline.
Attempts to find a lower-rank exact decomposition via randomized local search
within a structured family, with a strict time budget.
"""
import time
import random
import math

DIMENSIONS = [2, 5, 5]

def _validate(U, V, W, n, m, p, tol=1e-9):
    """
    Validate that the tensor decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each output element C[i,k] = sum_j A[i,j] B[j,k],
    we need: for all (i,j,k), the coefficient of A[i,j]*B[j,k] in
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] must be 1,
    and for all (i,j,k,i',j',k') with (i,j,k) != (i',j',k'),
    sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * 
          (indicator that (i,j,k) maps to same output as (i',j',k'))
    must be 0... actually the tensor decomposition requires:
    
    T[i,j,k; i',j',k'] = delta(i,i')*delta(j,j')*delta(k,k') for the multiplication tensor
    but the decomposition is:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta for matching (i,j,k)
    
    More precisely, the multiplication tensor T has indices (i,j,k) -> (i',j',k') where
    T[(i,j,k),(i',j',k')] = 1 if i=i', j=j', k=k' else 0.
    
    The decomposition says:
    T[(i,j,k),(i',j',k')] = sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i', r]
    
    Wait, let me re-read the convention. The output factor follows W[k*n+i, r].
    So for output C[i,k], the index in W is k*n+i.
    
    The tensor for matrix multiplication (2x5)*(5x5) -> (2x5):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[(i,j),(j,k),(k,i)] = 1 (using the trace convention for W)
    Actually, the standard form is:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    when we consider all index combinations.
    
    More carefully: the multiplication tensor is a 3-tensor of shape (n*m, m*p, p*n) where
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid (i,j,k), else 0.
    
    The decomposition requires:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[a][r] * V[b][r] * W[c][r]
                if abs(actual - expected) > tol:
                    return False
                # Check that non-matching entries are zero
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i2, j2, k2) == (i, j, k):
                                continue
                            a2 = i2 * m + j2
                            b2 = j2 * p + k2
                            c2 = k2 * n + i2
                            # This entry should be 0
                            val = 0.0
                            for r in range(rank):
                                val += U[a2][r] * V[b2][r] * W[c2][r]
                            if abs(val) > tol:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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

def _try_lower_rank(n, m, p, target_rank, time_budget, rng):
    """
    Attempt to find a decomposition of rank target_rank using randomized local search.
    Returns None if not found within time_budget.
    """
    start = time.time()
    deadline = start + time_budget
    
    # Initialize with random small values
    def random_init(rank):
        U = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p * n)]
        return U, V, W
    
    def compute_error(U, V, W, rank):
        """Compute the Frobenius norm of the error tensor."""
        err_sq = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    err_sq += (val - 1.0) ** 2
        # Also check off-diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i2, j2, k2) == (i, j, k):
                                    continue
                                a2 = i2 * m + j2
                                b2 = j2 * p + k2
                                c2 = k2 * n + i2
                                val = 0.0
                                for r in range(rank):
                                    val += U[a2][r] * V[b2][r] * W[c2][r]
                                err_sq += val ** 2
        return math.sqrt(err_sq)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple random restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        U, V, W = random_init(target_rank)
        
        # Gradient-free local search (random perturbations)
        for step in range(200):
            if time.time() > deadline:
                break
            
            err = compute_error(U, V, W, target_rank)
            if err < 1e-8:
                return {"U": U, "V": V, "W": W}
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
            
            # Random perturbation
            lr = 0.1 / (1 + step * 0.01)
            for _ in range(3):
                mat = rng.choice([U, V, W])
                row = rng.randrange(len(mat))
                col = rng.randrange(target_rank)
                mat[row][col] += rng.gauss(0, lr)
        
        if time.time() > deadline:
            break
    
    # Check best found
    if best_U is not None and best_error < 1e-6:
        # Verify with tighter tolerance
        if _validate(best_U, best_V, best_W, n, m, p, tol=1e-6):
            return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 50
    
    # Get the schoolbook baseline
    result = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    rng = random.Random(42)
    
    # Try ranks from base_rank-1 down, with decreasing time budgets
    for target_rank in range(base_rank - 1, max(base_rank - 10, 40), -1):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        time_budget = min(remaining * 0.5, 15.0)
        if time_budget < 2.0:
            break
        
        candidate = _try_lower_rank(n, m, p, target_rank, time_budget, rng)
        if candidate is not None:
            # Verify the candidate
            if _validate(candidate["U"], candidate["V"], candidate["W"], n, m, p, tol=1e-9):
                result = candidate
                break
    
    return result
# EVOLVE-BLOCK-END
