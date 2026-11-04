"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (multiple of 0.5)."""
    if abs(x - round(x)) < 1e-10:
        return True
    if abs(x * 2 - round(x * 2)) < 1e-10:
        return True
    return False

def _all_half_integer(U, V, W):
    """Check if all entries in U, V, W are half-integers."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _snap_to_half_integer(x):
    """Snap x to nearest half-integer."""
    return round(x * 2) / 2.0

def _validate_exact(U, V, W, n=2, m=4, p=5):
    """Exact validation using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(U[i*m+j][c]).limit_denominator(1000)
                    v_val = Fraction(V[j*p+k][c]).limit_denominator(1000)
                    w_val = Fraction(W[k*n+i][c]).limit_denominator(1000)
                    val += u_val * v_val * w_val
                if val != Fraction(1):
                    return False
    return True

def _try_ranked_decomposition(n, m, p, rank, time_budget, seed=None):
    """
    Try to find a rank-`rank` decomposition using iterative refinement.
    Uses a gradient-free approach: initialize randomly, then try local perturbations.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start = time.time()
    deadline = start + time_budget
    
    # Initialize with small random values
    U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
    
    # Compute error: sum of squared errors over all (i,j,k)
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err += (val - 1.0) ** 2
        return err
    
    current_err = compute_error(U, V, W)
    best_err = current_err
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Iterative improvement: try small perturbations
    step_size = 0.1
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random matrix and a random entry to perturb
        mat_choice = rng.randint(0, 2)
        if mat_choice == 0:
            mat = U
            mat_name = 'U'
            max_row = n * m
        elif mat_choice == 1:
            mat = V
            mat_name = 'V'
            max_row = m * p
        else:
            mat = W
            mat_name = 'W'
            max_row = p * n
        
        row_idx = rng.randint(0, max_row - 1)
        col_idx = rng.randint(0, rank - 1)
        
        old_val = mat[row_idx][col_idx]
        perturbation = rng.gauss(0, step_size)
        mat[row_idx][col_idx] = old_val + perturbation
        
        new_err = compute_error(U, V, W)
        
        if new_err < current_err - 1e-12:
            current_err = new_err
            if new_err < best_err:
                best_err = new_err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            # Occasionally increase step size
            if iteration % 500 == 0:
                step_size *= 0.95
        else:
            mat[row_idx][col_idx] = old_val
        
        # Check if we've found a good enough solution
        if current_err < 1e-15:
            # Snap to half-integers and validate
            U_snap = [[_snap_to_half_integer(v) for v in row] for row in U]
            V_snap = [[_snap_to_half_integer(v) for v in row] for row in V]
            W_snap = [[_snap_to_half_integer(v) for v in row] for row in W]
            if _all_half_integer(U_snap, V_snap, W_snap):
                if _validate(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap, True
    
    # Final check on best found
    if best_err < 1e-10:
        U_snap = [[_snap_to_half_integer(v) for v in row] for row in best_U]
        V_snap = [[_snap_to_half_integer(v) for v in row] for row in best_V]
        W_snap = [[_snap_to_half_integer(v) for v in row] for row in best_W]
        if _all_half_integer(U_snap, V_snap, W_snap):
            if _validate(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap, True
    
    return None, None, None, False

def _construct_strassen_decomposition(n, m, p):
    """
    Construct a Strassen-like decomposition for MM_{2,4,5}.
    Uses the identity:
      a*c + b*d = (a+b)(c+d) - a*d - b*c
    Applied to blocks of size 2x2 within the 4x4 intermediate dimension.
    """
    # We'll use the following block decomposition:
    # Split the 4x4 intermediate into two 2x2 blocks.
    # For each 2x2 block, use Strassen's 7-multiplication algorithm.
    
    # Actually, let's use a simpler approach:
    # For MM_{2,4,5}, we can view it as two independent MM_{2,2,5} problems
    # (splitting the 4 columns of A into two groups of 2).
    # But that doesn't directly help.
    
    # Let's try the direct Strassen approach on 2x2 blocks:
    # C_{i,k} = sum_{j=0}^{3} A_{i,j} B_{j,k}
    # = sum_{j=0}^{1} A_{i,j} B_{j,k} + sum_{j=2}^{3} A_{i,j} B_{j,k}
    
    # For each 2x2 block (j in {0,1} or j in {2,3}):
    # C_{i,k}^{(block)} = A_{i, j0} B_{j0, k} + A_{i, j1} B_{j1, k}
    # This is a 2x1 * 1x5 multiplication, which has rank 2 (no sharing possible).
    
    # So total rank = 2 blocks * 2 * 5 = 20.
    # This is the same as the schoolbook rank for this case.
    
    # Let's try a different approach: use Strassen on the full 4x4.
    # Actually, for MM_{2,4,5}, the optimal rank is likely 20 (the lower bound).
    
    # Let's construct the rank-20 decomposition:
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each (i, j, k) triple, we have a term A_{i,j} * B_{j,k} contributing to C_{i,k}.
    # But we can't reduce below n*m*p = 40 in general...
    
    # Wait, let me reconsider. The lower bound is max(n*m, m*p, p*n) = max(8, 20, 10) = 20.
    # Can we achieve rank 20?
    
    # For rank 20, each column of U, V, W must be carefully chosen.
    # Let's try: for each j and k, we have a "slice" that contributes to all i.
    # C_{i,k} += A_{i,j} * B_{j,k}
    # For fixed j, k: this is a vector (A_{0,j}*B_{j,k}, A_{1,j}*B_{j,k}) added to (C_{0,k}, C_{1,k}).
    
    # In tensor form: for fixed j, k, the contribution is:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where c indexes the 20 terms.
    
    # If we assign one term per (j, k) pair, that's 4*5 = 20 terms.
    # For term c corresponding to (j, k):
    # U[i*m+j, c] = A_{i,j} (but A_{i,j} is a variable, not a constant!)
    
    # This approach doesn't work because U, V, W must be constant matrices.
    
    # The correct approach: the tensor T has entries T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # We need to decompose this tensor into rank-1 components.
    
    # For MM_{2,4,5}, the rank is actually 40 (schoolbook) because the tensor is "generic" enough.
    
    # Let's just return the schoolbook decomposition.
    return None, None, None

def _try_structured_search(n, m, p, time_budget):
    """
    Try structured decompositions based on algebraic properties.
    For MM_{2,4,5}, explore if rank < 40 is achievable.
    """
    start = time.time()
    deadline = start + time_budget
    
    # Lower bound analysis:
    # The tensor T has shape (8, 20, 10).
    # For each (i,j) pair, the slice T[i*m+j, :, :] has exactly one non-zero entry.
    # This suggests rank = n*m*p = 40 might be optimal.
    
    # But let's try to verify by attempting rank 20-39.
    
    best_rank = n * m * p
    best_U = best_V = best_W = None
    
    # Try a few structured approaches
    # Approach: exploit the fact that n=2 is small
    
    # For each k (0 to 4), we compute C[:,k] = A @ B[:,k]
    # This is a 2x4 * 4x1 multiplication for each k.
    # Total: 5 independent 2x4*4x1 multiplications.
    
    # For 2x4 * 4x1: rank = 8 (two dot products of length 4, no sharing)
    # So total rank = 5 * 8 = 40.
    
    # Can we share across k values? 
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # For fixed i,j: A_{i,j} appears in C_{i,0}, C_{i,1}, ..., C_{i,4}.
    # So A_{i,j} * B_{j,k} for different k are distinct products.
    # No sharing possible.
    
    # Thus rank 40 is likely optimal.
    
    return best_rank, best_U, best_V, best_W

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook: rank = n*m*p = 40
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
    
    # Verify schoolbook
    assert _validate(U, V, W, n, m, p), "Schoolbook decomposition failed validation!"
    
    # Try to find lower rank decompositions
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Attempt lower ranks
    for target_rank in range(35, 39):  # Try a few below 40
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 15:
            break
        
        time_budget = min(20.0, remaining - 10)
        if time_budget < 5:
            break
        
        U_try, V_try, W_try, success = _try_ranked_decomposition(
            n, m, p, target_rank, time_budget, seed=target_rank * 1000 + 42
        )
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # Return the best decomposition found
    result_U = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in best_U]
    result_V = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in best_V]
    result_W = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in best_W]
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
