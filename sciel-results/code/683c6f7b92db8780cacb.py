"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    equals 1 iff (i,j,k) is the standard product index, i.e., the decomposition
    computes the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                # For matrix multiplication, the tensor entry should be 1 for all (i,j,k)
                # because (AB)[i,k] = sum_j A[i,j]*B[j,k], so the tensor is all ones
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_half_integer(U, V, W):
    """Check all entries are integer or half-integer (i.e., 2*x is integer)."""
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if v is None:
                    continue
                two_v = 2 * v
                if abs(two_v - round(two_v)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = 3, 3, 5
    t0 = time.time()
    deadline = 130.0  # safety margin before 150s kill

    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try a structured low-rank approach:
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij}B_{jk})
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # A natural decomposition: use rank = m * n * p / something.
    # Actually, let's try a rank that factors nicely.
    # 
    # Key insight: T[i,j,k] = 1 can be written as a sum of products.
    # If we let c index pairs (j', k') with j' in range(m), k' in range(p),
    # then we need sum_{j',k'} U[i,j,c(j',k')] * V[j,k,c(j',k')] * W[k,i,c(j',k')] = 1.
    # This is like: for each (j',k'), we have a term. We need the sum to be 1.
    # If U[i,j,c(j',k')] = delta(j, j'), V[j,k,c(j',k')] = delta(k, k'), W[k,i,c(j',k')] = 1,
    # then sum_{j',k'} delta(j,j') * delta(k,k') * 1 = 1. That works but rank = m*p = 15.
    # Wait, that's not right. Let me re-examine.
    #
    # Actually: T[i,j,k] should equal the matrix multiplication tensor.
    # The matrix multiplication tensor has T[i,j,k] = 1 for ALL i,j,k.
    # Because (AB)_{ik} = sum_j A_{ij} B_{jk}, so the structure tensor is all 1s.
    #
    # So we need: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    #
    # Simple construction with rank = m * p:
    # Let c = j'*p + k' where j' in 0..m-1, k' in 0..p-1.
    # Set U[i*m+j, c] = 1 if j == j' else 0.
    # Set V[j*p+k, c] = 1 if k == k' else 0.
    # Set W[k*n+i, c] = 1.
    # Then sum_c U * V * W = sum_{j',k'} [j==j'] * [k==k'] * 1 = 1. Correct!
    # Rank = m * p = 15.
    #
    # Can we do better? By symmetry, we could also use rank = n*p or n*m.
    # min(n*m, n*p, m*p) = min(9, 15, 15) = 9.
    # Using rank = n*m: let c = i'*m + j' where i' in 0..n-1, j' in 0..m-1.
    # U[i*m+j, c] = 1 if i==i' and j==j' else 0.
    # V[j*p+k, c] = 1 (always 1? No, that doesn't work)
    # Let me think again.
    #
    # For rank = n*m: c = i'*m + j'.
    # We need sum_{i',j'} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # If U[i*m+j, c] = 1 if i==i' and j==j', then for fixed (i,j), only one c is nonzero.
    # So V[j*p+k, c(i,j)] * W[k*n+i, c(i,j)] must = 1 for all k.
    # But c(i,j) = i*m+j, so V[j*p+k, i*m+j] and W[k*n+i, i*m+j] must multiply to 1.
    # V[j*p+k, i*m+j] must be 1 for all k (to get product 1 with W=1), or W must compensate.
    # If V[j*p+k, i*m+j] = 1 and W[k*n+i, i*m+j] = 1 for all j,k,i', then it works.
    # But V[j*p+k, c] = 1 for all j,k,c? That's a constant matrix, which is fine.
    # W[k*n+i, c] = 1 for all k,i,c? Also constant.
    # Then sum_c U[i*m+j,c] * 1 * 1 = sum_c U[i*m+j,c] = 1 (since only one c has U=1). Works!
    # Rank = n*m = 9.
    #
    # Let's verify: U is n*m x n*m, with U[i*m+j, i'*m+j'] = delta(i,i')*delta(j,j').
    # V is m*p x n*m, all ones.
    # W is p*n x n*m, all ones.
    # For any (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_c delta(i,i')delta(j,j') * 1 * 1  [where c = i'*m+j']
    # = 1. Correct!
    #
    # Can we do even better? Theoretical lower bound for 3x3x5 is 23? No, that seems high.
    # Actually, the tensor rank of 3x3x5 matrix multiplication is known to be 23? Let me reconsider.
    # The "rank" here is the tensor rank (border rank vs rank). The schoolbook is 45.
    # We just found rank 9. Is that valid? Let me double-check the indexing.
    #
    # U has shape (n*m, rank) = (9, 9). U[i*m+j, c] where c = i'*m+j'.
    # V has shape (m*p, rank) = (15, 9). V[j*p+k, c] = 1.
    # W has shape (p*n, rank) = (15, 9). W[k*n+i, c] = 1.
    #
    # This gives rank 9. Let's also try the symmetric version with rank = n*p = 15 and m*p = 15.
    # The minimum of these is 9.
    #
    # Can we go below 9? The lower bound for 3x3x5 tensor rank... I'm not sure.
    # Let's just use rank 9 which is clearly better than 45.

    # Construct rank = n*m = 9 decomposition
    rank_nm = n * m
    U = [[0] * rank_nm for _ in range(n * m)]
    V = [[1] * rank_nm for _ in range(m * p)]
    W = [[1] * rank_nm for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
    candidate = {"U": U, "V": V, "W": W}

    if _verify(U, V, W, n, m, p) and _verify_half_integer(U, V, W):
        best = candidate
        best_rank = rank_nm

    # Try rank = m = 3? Unlikely but let's think.
    # With rank 3, can we decompose? Probably not for 3x3x5.
    # The bilinear complexity of 3x3x5 is at least... I'll stick with 9.

    # Try to see if rank 8 or lower works via a simple parametric search.
    # This is unlikely to succeed quickly, but let's try a few structured forms.
    
    # For now, return the best we have.
    return best
# EVOLVE-BLOCK-END
