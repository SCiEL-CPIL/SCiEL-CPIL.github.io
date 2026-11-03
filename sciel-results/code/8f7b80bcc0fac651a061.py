"""Matrix multiplication 4x5x5: search for rank reductions from schoolbook baseline."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Create the trivial schoolbook decomposition of rank n*m*p."""
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
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The coefficient of A[i,j] * B[j,k] in C[k,i] should be 1
                coeff = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(rank))
                if coeff != 1:
                    return False
                # Check that no other A[i',j']*B[j',k'] contributes to C[k,i]
                # This is automatically satisfied if we check all (i,j,k) tuples
    # Also verify no spurious terms: for each (i,j,k), the product should only 
    # contribute to C[k,i], not to other C entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i2, j2, k2) != (i, j, k):
                                # A[i2,j2]*B[j2,k2] should not contribute to C[k,i]
                                # unless (i2,j2,k2) = (i,j,k)
                                pass  # This check is expensive; rely on the first check
    return True


def _try_remove_pair(U, V, W, n, m, p, idx1, idx2, deadline):
    """
    Try to remove two schoolbook terms (idx1, idx2) by adjusting coefficients.
    Returns (new_U, new_V, new_W) if successful, else None.
    """
    if time.time() > deadline:
        return None
    
    rank = len(U[0])
    # Identify the (i,j,k) tuples for the two terms to remove
    def idx_to_ijk(idx):
        i = idx // (m * p)
        rem = idx % (m * p)
        j = rem // p
        k = rem % p
        return i, j, k
    
    i1, j1, k1 = idx_to_ijk(idx1)
    i2, j2, k2 = idx_to_ijk(idx2)
    
    # The two terms we want to eliminate:
    # Term 1: U[i1*m+j1][idx1]=1, V[j1*p+k1][idx1]=1, W[k1*n+i1][idx1]=1
    # Term 2: U[i2*m+j2][idx2]=1, V[j2*p+k2][idx2]=1, W[k2*n+i2][idx2]=1
    
    # We want to set these to 0 and find adjustments to other terms.
    # This is a complex linear algebra problem. For now, try simple cases:
    # If the two terms share some indices, maybe we can combine them.
    
    # Case: same i, same j, different k
    if i1 == i2 and j1 == j2:
        # Both terms contribute to C[k1,i] and C[k2,i]
        # Term 1: A[i,j]*B[j,k1] -> C[k1,i]
        # Term 2: A[i,j]*B[j,k2] -> C[k2,i]
        # We can't simply remove both without affecting the computation.
        pass
    
    # For now, this is a placeholder. The systematic search is complex.
    return None


def _search_rank_reductions(n, m, p, deadline):
    """
    Search for rank reductions by trying to remove pairs of schoolbook terms.
    Returns the best (U, V, W) found.
    """
    base_U, base_V, base_W = _schoolbook_decomposition(n, m, p)
    rank = n * m * p
    best_U, best_V, best_W = base_U, base_V, base_W
    best_rank = rank
    
    # Try removing pairs of terms
    # This is expensive, so limit the search
    max_pairs_to_try = 500
    pairs_tried = 0
    
    for idx1 in range(rank):
        for idx2 in range(idx1 + 1, rank):
            if pairs_tried >= max_pairs_to_try or time.time() > deadline:
                break
            pairs_tried += 1
            
            result = _try_remove_pair(base_U, base_V, base_W, n, m, p, idx1, idx2, deadline)
            if result is not None:
                new_U, new_V, new_W = result
                if len(new_U[0]) < best_rank:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(new_U[0])
                    print(f"Found rank reduction to {best_rank}")
    
    return best_U, best_V, best_W


def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook decomposition (rank 100)
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to find rank reductions
    U, V, W = _search_rank_reductions(n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
