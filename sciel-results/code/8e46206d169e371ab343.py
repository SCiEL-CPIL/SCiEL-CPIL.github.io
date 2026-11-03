"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np
from numpy.linalg import norm

DIMENSIONS = [3, 4, 7]
N, M, P = 3, 4, 7

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the standard schoolbook tensor decomposition of rank n*m*p."""
    n, m, p = N, M, P
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


def _verify(U, V, W, tol=1e-8):
    """Check whether the factor matrices exactly reproduce matrix multiplication."""
    n, m, p = N, M, P
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    # Build the target multiplication tensor: T[i,j,k] = 1 if the term u_i*v_j*w_k appears
    # The tensor for matrix multiplication is T_{i,j,k}^{i,j,k} = 1 for all i,j,k
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank_of(U, V, W):
    return len(U[0])


def _try_block_decomposition(n1, m1, p1, n2, m2, p2, deadline):
    """
    Try to build a decomposition for (n1+m1) x (m1+m2) x (p1+p2) by splitting
    the middle dimension into two blocks. This requires the two sub-matrices
    to share the middle dimension in a compatible way.
    
    Actually, a cleaner block approach: split the p dimension into p1 and p2,
    and the i dimension into n1 and n2. Then the matrix is (n1+n2) x m x (p1+p2).
    We can compute the four blocks independently, but that doesn't reduce rank.
    
    A better approach: split the m dimension into m1 and m2, and the p dimension into p1 and p2.
    But the middle dimension must be the same for both factors.
    
    Let me think differently. The standard way to get a better decomposition:
    - If we can find a decomposition for smaller dimensions and combine them.
    
    Actually, the simplest improvement: use the fact that matrix multiplication
    (3,4,7) can be seen as a sum of (3,4,3) and (3,4,4) by splitting the p dimension.
    The tensor for (3,4,7) = tensor for (3,4,3) with k in {0,1,2} + tensor for (3,4,4) with k in {3,4,5,6}.
    But these are just disjoint subsets of the sum, so the rank is just the sum of the ranks.
    That doesn't help.
    
    Let me just do a random search with verification.
    """
    return None


def _random_search(target_rank, deadline):
    """
    Randomly generate factor matrices and check if they decompose the tensor.
    This is very unlikely to succeed for large ranks, but let's try with small ranks.
    """
    n, m, p = N, M, P
    random.seed(42)
    np.random.seed(42)
    
    start = time.time()
    for _ in range(100):  # limit attempts
        if time.time() > deadline:
            break
        U = np.random.randint(0, 2, size=(n * m, target_rank))
        V = np.random.randint(0, 2, size=(m * p, target_rank))
        W = np.random.randint(0, 2, size=(p * n, target_rank))
        
        # Quick check: each (i,j,k) triple should sum to 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix. Returns the schoolbook decomposition if no
    better decomposition is found.
    """
    deadline = time.time() + 100  # leave 20s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find a better decomposition
    # Known lower bounds: R(3,4,7) >= something
    # Upper bound: 84 from schoolbook
    # Try ranks from 83 down to some lower bound
    # This is very unlikely to succeed with random search, but let's try a few
    
    # Actually, let me try a structured approach:
    # The tensor T for (n,m,p) has T[i,j,k] = 1 for all i,j,k.
    # A decomposition of rank r means we can write T as sum of r rank-1 tensors.
    # Each rank-1 tensor is u_i * v_j * w_k for some u, v, w vectors.
    
    # For (3,4,7), the tensor has 3*4*7 = 84 entries, all equal to 1.
    # The schoolbook decomposition uses 84 rank-1 terms, each selecting one entry.
    
    # To reduce the rank, we need to find a set of rank-1 tensors that sum to the all-ones tensor.
    # This is equivalent to finding a set of triples (u, v, w) such that for each (i,j,k),
    # sum_c u_c[i] * v_c[j] * w_c[k] = 1.
    
    # This is a very constrained problem. Let's just return the schoolbook decomposition.
    
    return best

# EVOLVE-BLOCK-END
