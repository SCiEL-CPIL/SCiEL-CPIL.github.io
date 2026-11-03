"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    terms = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        {'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,0): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
    ]
    return terms

def _build_strassen_blocks(n, m, p):
    """Build decomposition using 2x2 Strassen blocks for the 4x4x5 case.
    
    Split: n=[2,2], m=[2,2], p=[2,2,1]
    """
    n_blocks = [2, 2]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 1]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2]
    p_offsets = [0, 2, 4]
    
    rank = 0
    U = {}
    V = {}
    W = {}
    
    for ni, ns in enumerate(n_blocks):
        for mi, ms in enumerate(m_blocks):
            for pi, ps in enumerate(p_blocks):
                no = n_offsets[ni]
                mo = m_offsets[mi]
                po = p_offsets[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    for term in _strassen_2x2_terms():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                else:
                    # Naive for non-2x2x2 blocks
                    for li in range(ns):
                        for lj in range(ms):
                            for lk in range(ps):
                                i, j, k = no + li, mo + lj, po + lk
                                U.setdefault((i, j), []).append((1, rank))
                                V.setdefault((j, k), []).append((1, rank))
                                W.setdefault((k, i), []).append((1, rank))
                                rank += 1
    
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for coeff, r in entries:
            U_mat[i * m + j][r] += coeff
    for (j, k), entries in V.items():
        for coeff, r in entries:
            V_mat[j * p + k][r] += coeff
    for (k, i), entries in W.items():
        for coeff, r in entries:
            W_mat[k * n + i][r] += coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _try_slice_elimination(U, V, W, n, m, p, deadline):
    """Try to reduce rank by identifying redundant bilinear forms.
    
    Approach: For each output slice (fixed k), the contribution is:
    sum_{i,j} C[i,j] * A[i,:] * B[:,k] = sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r]
    
    We try to find linear dependencies among the rank-1 components.
    """
    rank = len(U[0])
    best_rank = rank
    best_result = None
    
    # Build the "slice tensor" for each k
    # For each k, we have a matrix M_k where M_k[i,j] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal delta(i,j,k) for the multiplication tensor
    
    # Try random linear combinations of columns to see if we can merge
    # This is a heuristic: try to find two columns that can be combined
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Pick two random columns to try to merge
        r1, r2 = random.sample(range(best_rank), 2)
        
        # Try to eliminate column r2 by expressing it as a linear combination
        # of other columns (excluding r1)
        # This is complex; instead, try a simpler heuristic:
        # Check if column r2 is "close" to being a multiple of column r1
        
        # Actually, let's try a different approach: 
        # Remove one column and see if the decomposition still works
        # by adjusting other columns
        
        # Skip this for now - too slow
        pass
    
    return best_result

def _random_local_search(U, V, W, n, m, p, deadline, max_iter=5000):
    """Perform random local search to reduce rank.
    
    Strategy: Start from a valid decomposition, then try to:
    1. Remove a column (set to zero) and check if still valid
    2. Perturb a column slightly and check
    3. Swap entries between columns
    """
    rank = len(U[0])
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    no_improve = 0
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try to remove a random column
        r = random.randint(0, best_rank - 1)
        
        # Create a copy with column r zeroed out
        U_test = [row[:] for row in best_U]
        V_test = [row[:] for row in best_V]
        W_test = [row[:] for row in best_W]
        
        for i in range(n * m):
            U_test[i][r] = 0
        for j in range(m * p):
            V_test[j][r] = 0
        for k in range(p * n):
            W_test[k][r] = 0
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            # Success! Remove column r
            best_U = [row[:r] + row[r+1:] for row in U_test]
            best_V = [row[:r] + row[r+1:] for row in V_test]
            best_W = [row[:r] + row[r+1:] for row in W_test]
            best_rank -= 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 100:
                break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None

def _structured_initialization(n, m, p):
    """Create a structured initialization that might be easier to optimize.
    
    Use symmetry: since we're multiplying 4x4 by 4x5, there's some structure.
    """
    # Start with Strassen blocks
    return _build_strassen_blocks(n, m, p)

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based initialization
    result = _structured_initialization(n, m, p)
    current_rank = len(result["U"][0])
    
    # Try local search to reduce rank
    while time.time() < deadline:
        improved = False
        
        # Try random local search
        search_result = _random_local_search(
            result["U"], result["V"], result["W"], n, m, p, deadline, max_iter=2000
        )
        
        if search_result is not None:
            new_rank = len(search_result["U"][0])
            if new_rank < current_rank:
                result = search_result
                current_rank = new_rank
                improved = True
        
        if not improved:
            # Try a different random restart
            # Perturb the current solution
            if time.time() > deadline - 10:
                break
            
            # Add small random perturbations to see if we can escape local minima
            # This is a heuristic - may not work well
            break
    
    # Final verification
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
            return result
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
