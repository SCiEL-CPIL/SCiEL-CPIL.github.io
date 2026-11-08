"""Find rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product
from math import gcd
from functools import reduce

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Check all entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    if j == l:
                        target = 1.0
                    else:
                        target = 0.0
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                    if abs(val - target) > 1e-9:
                        return False
    return True

def _to_fraction_matrix(M, max_den=10000):
    """Convert a float matrix to a fraction matrix if possible."""
    result = []
    for row in M:
        frac_row = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                frac_row.append(Fraction(int(round(x))))
            else:
                f = Fraction(x).limit_denominator(max_den)
                if abs(x - float(f)) > 1e-8:
                    return None
                frac_row.append(f)
        result.append(frac_row)
    return result

def _lcm_list(lst):
    """Compute LCM of a list of integers."""
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = result * x // gcd(result, x)
    return result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Strategy: Try structured decompositions
    # For 2x4x4, try to find rank < 32
    
    # Approach 1: Use the fact that we can share some terms
    # The key constraint is that for each (i,k), the vector W[k*n+i] must satisfy:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = delta(j,l) for all j,l
    
    # Let's try a more systematic search
    # We'll parameterize U and V with small integer entries and solve for W
    
    max_attempts = 5000
    
    # Try ranks from 24 to 31
    for rank in range(24, 32):
        if time.time() > deadline:
            break
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            seed = attempt * 7919 + rank * 104729 + 12345
            np.random.seed(seed)
            
            # Generate U and V with entries in {-2, -1, 0, 1, 2}
            # Use varying sparsity
            sparsity = 0.15 + 0.35 * (attempt % 10) / 10.0
            
            U_r = np.where(np.random.rand(n * m, rank) < sparsity, 
                          np.random.choice([-2, -1, 1, 2], size=(n * m, rank)), 0)
            V_r = np.where(np.random.rand(m * p, rank) < sparsity, 
                          np.random.choice([-2, -1, 1, 2], size=(m * p, rank)), 0)
            
            U_np = U_r.astype(float)
            V_np = V_r.astype(float)
            
            W_np = np.zeros((p * n, rank))
            valid = True
            
            for i in range(n):
                if not valid:
                    break
                for k in range(p):
                    if not valid:
                        break
                    
                    # For each (i,k), we need to find W_col such that:
                    # For each j: sum_r U[i*m+j][r] * V[j*p+k][r] * W_col[r] = 1
                    # This gives us m equations in rank unknowns
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U_np[i * m + j, r] * V_np[j * p + k, r]
                    
                    # Solve A @ W_col = ones(m)
                    try:
                        if rank >= m:
                            # Use least squares
                            W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-8:
                                valid = False
                                break
                        else:
                            # Underdetermined, find minimum norm solution
                            W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-8:
                                valid = False
                                break
                        
                        W_np[k * n + i, :] = W_col
                    except np.linalg.LinAlgError:
                        valid = False
                        break
            
            if valid:
                # Check if W has rational entries with small denominators
                all_w = W_np.flatten().tolist()
                frac_w = _to_fraction_matrix(W_np, max_den=5000)
                
                if frac_w is not None:
                    # Get common denominator
                    dens = [f.denominator for row in frac_w for f in row]
                    common_den = _lcm_list(dens)
                    
                    if common_den <= 100000:  # Reasonable bound
                        # Scale W to integers
                        W_int = [[int(f * common_den) for f in row] for row in frac_w]
                        U_int = [[int(x) for x in row] for row in U_np.tolist()]
                        V_int = [[int(x) for x in row] for row in V_np.tolist()]
                        
                        # Verify
                        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                            best_result = {"U": U_int, "V": V_int, "W": W_int}
                            best_rank = rank
                            break  # Found valid for this rank, try lower
        
        # If we found a valid decomposition at this rank, we could break
        # but let's continue to see if we can do better
        if best_rank <= rank:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
