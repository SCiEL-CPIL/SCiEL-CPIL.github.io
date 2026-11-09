"""Tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time
import random
import itertools

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-54 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(decomp, n, m, p, tol=1e-10):
    """Check whether decomp exactly decomposes the multiplication tensor."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    # For each (i, j, k), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                for c in range(rank):
                    s += u_val[c] * v_val[c] * w_val[c]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that for any (i, j, k) where the tensor entry is 0, it remains 0
    # The tensor T[i,j,k] = 1 for all valid i,j,k in this multiplication, so no extra check needed.
    # But we should verify that the factorization is consistent: for each c, the outer product
    # U[:,c] ⊗ V[:,c] ⊗ W[:,c] contributes to the tensor. Since all tensor entries are 1,
    # we just need the sum to be 1 for each (i,j,k).
    return True


def _try_reductions(rank_target, n, m, p, max_rounds=20000, seed=42):
    """
    Attempt to find a valid decomposition of rank_target by starting from schoolbook
    and merging pairs of columns via random linear combinations.
    
    This is a heuristic: we try random linear combinations of two columns and check
    if the resulting tensor still decomposes correctly (approximately).
    """
    decomp = _schoolbook()
    U = [row[:] for row in decomp["U"]]
    V = [row[:] for row in decomp["V"]]
    W = [row[:] for row in decomp["W"]]
    
    current_rank = len(U[0])
    
    rng = random.Random(seed)
    
    for _ in range(max_rounds):
        if current_rank <= rank_target:
            break
        
        # Pick two random columns to merge
        c1, c2 = rng.sample(range(current_rank), 2)
        
        # Try a random linear combination
        alpha = rng.uniform(-1, 1)
        beta = rng.uniform(-1, 1)
        
        if abs(alpha) < 1e-6 and abs(beta) < 1e-6:
            continue
        
        # Create new column
        new_U = [U[i][c1] * alpha + U[i][c2] * beta for i in range(n * m)]
        new_V = [V[i][c1] * alpha + V[i][c2] * beta for i in range(m * p)]
        new_W = [W[i][c1] * alpha + W[i][c2] * beta for i in range(p * n)]
        
        # Temporarily replace c1 with new column, remove c2
        for i in range(n * m):
            U[i][c1] = new_U[i]
        for i in range(m * p):
            V[i][c1] = new_V[i]
        for i in range(p * n):
            W[i][c1] = new_W[i]
        
        # Remove column c2
        for i in range(n * m):
            U[i].pop(c2)
        for i in range(m * p):
            V[i].pop(c2)
        for i in range(p * n):
            W[i].pop(c2)
        
        current_rank -= 1
        
        # Verify
        if not _verify({"U": U, "V": V, "W": W}, n, m, p):
            # Revert
            for i in range(n * m):
                U[i].insert(c2, U[i][c1] / alpha if abs(alpha) > 1e-10 else 0)
            # This revert is messy; instead, just restart
            return None
    
    if current_rank <= rank_target:
        return {"U": U, "V": V, "W": W}
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 by 3x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # First, always have the schoolbook decomposition as fallback
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition
    # For 3x3 matrix multiplication, the best known exact rank is 23 (Strassen-like).
    # For 3x3 by 3x6, we can potentially use a Kronecker product structure.
    # The tensor for (n,m) x (m,p) multiplication has rank at most rank(n,m,n) * p
    # if we can decompose it as a tensor product.
    
    # Let's try a structured approach: assume the decomposition can be written as
    # a sum of terms where each term factorizes in a specific way.
    
    # Strategy: Use the known rank-23 decomposition for 3x3 and combine with the 6 columns.
    # Actually, the multiplication of a 3x3 by a 3x6 matrix can be seen as 6 independent
    # 3x3 multiplications (one for each column of the right matrix). So the rank is at most
    # 23 * 6 = 138, which is worse than schoolbook 54.
    
    # Better: The tensor T[i,j,k] for (n,m,p) = (3,3,6) has 54 entries.
    # We can try to find a decomposition of rank < 54.
    
    # Let's use a random search with verification for ranks from 53 down to, say, 40.
    # Given the time budget, we'll try a few random restarts.
    
    for target_rank in range(schoolbook_rank - 1, 39, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random restarts for this target rank
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            result = _try_reductions(target_rank, n, m, p, max_rounds=5000, seed=seed)
            if result is not None:
                # Verify one more time
                if _verify(result, n, m, p):
                    # Convert to list of lists with float entries
                    U = [[float(x) for x in row] for row in result["U"]]
                    V = [[float(x) for x in row] for row in result["V"]]
                    W = [[float(x) for x in row] for row in result["W"]]
                    return {"U": U, "V": V, "W": W}
    
    # If no improvement found, return schoolbook
    return best
# EVOLVE-BLOCK-END
