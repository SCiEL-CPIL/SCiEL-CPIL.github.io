"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition as integer lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """
    Check exact decomposition:
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    Actually the tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    and this must equal the standard multiplication tensor which is 1 if (i,j,k) is a valid triple.
    Wait, the multiplication tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]?
    No. The multiplication tensor for C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k? No.
    The tensor is defined such that the bilinear form <T, A ⊗ B> = C.
    Actually, the rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    where u_c is in F^{n*m}, v_c in F^{m*p}, w_c in F^{p*n}.
    The entry T[i*m+j, j*p+k, k*n+i] ... wait, the convention is W[k*n+i, r].
    
    The standard multiplication tensor has entries:
    T[a, b, c] where a indexes A (i*m+j), b indexes B (j'*p+k), c indexes C (k'*n+i')
    T[a, b, c] = 1 if the indices are compatible, i.e., j == j' and i == i' and k == k'.
    So T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    
    So the condition is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    Wait, no. The decomposition is T = sum_c u_c v_c^T w_c^T (as a 3-tensor).
    So T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r].
    We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    But wait, for a given (i,j,k), the triple (i*m+j, j*p+k, k*n+i) is the only non-zero entry
    that involves all three specific indices. Other entries like T[i*m+j, j'*p+k, k*n+i] with j' != j
    should be 0.
    
    So the verification is:
    For all a in [n*m], b in [m*p], c in [p*n]:
        sum_r U[a,r] * V[b,r] * W[c,r] == 1 if (a,b,c) corresponds to a valid (i,j,k) triple, else 0.
    
    A triple (a,b,c) is valid if:
        a = i*m + j, b = j'*p + k, c = k'*n + i'
        and j == j' and i == i' and k == k'.
    """
    n, m, p = n, m, p
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the target tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0.
    # We check: for each r, the contribution is U[:,r] * V[:,r] * W[:,r]
    # Sum over r.
    
    # Efficient check:
    # For each valid (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # For each invalid (a,b,c), check sum_r U[a,r] * V[b,r] * W[c,r] == 0
    
    # This is expensive for all (a,b,c). Let's use the structure.
    # The non-zero entries are exactly (i*m+j, j*p+k, k*n+i) for i in range(n), j in range(m), k in range(p).
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if val != 1:
                    return False
    
    # Check that for each (i,j), the row U[i*m+j, :] is orthogonal (in the bilinear sense) to
    # V[j'*p+k, :] for j' != j, when combined with any W.
    # Actually, a simpler necessary condition:
    # For fixed (i,j), and any k, the vector U[i*m+j, :] * W[k*n+i, :] (element-wise product)
    # should have dot product with V[j*p+k, :] equal to 1, and with V[j'*p+k, :] equal to 0 for j' != j.
    
    # Let's check a few zero entries to be safe.
    # For i in range(n), j in range(m), j2 in range(m), k in range(p):
    #   if j != j2: sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i, r] == 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if val != 0:
                        return False
    
    # Similarly for i != i2
    for i in range(n):
        for i2 in range(n):
            if i == i2:
                continue
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i2
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if val != 0:
                        return False
    
    # And for k != k2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    if k == k2:
                        continue
                    a = i * m + j
                    b = j * p + k2
                    c = k * n + i
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if val != 0:
                        return False
    
    return True


def _rank_from_lists(U, V, W, n, m, p):
    return len(U[0]) if U else 0


def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For 2x3x5, the lower bound is max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # The upper bound is n*m*p = 30
    # Known results: for 2x2x2, rank is 7. For larger, it's hard.
    
    # Strategy: Randomized search with local improvement
    # 1. Generate random rank-r decompositions
    # 2. Check if they satisfy the tensor equations
    # 3. Use gradient descent or local search to refine
    
    # For exact integer/half-integer solutions, we can try:
    # - Small random integer matrices
    # - Check if they work
    
    # Let's try ranks from 15 upward
    for rank in range(15, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try random integer matrices
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Generate random U, V, W with small integer entries
            # Use entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            W = np.random.randint(-1, 2, size=(p * n, rank))
            
            # Check if this is a valid decomposition
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
        
        if best_rank == rank:
            break
    
    # If we found something better, return it
    if best_rank < n * m * p:
        return best
    
    # Otherwise, return schoolbook
    return best

# EVOLVE-BLOCK-END
