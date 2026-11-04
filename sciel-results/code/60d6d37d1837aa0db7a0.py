"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k, l] = delta_{j,k}.
    Shape: (n, m, p, m) where T[i, j, k, l] = 1 if j==k else 0, for all i.
    Actually the tensor for (n,m) x (m,p) -> (n,p) is:
    T[i, l, j, k] such that C[i,l] = sum_{j,k} A[i,j] B[k,l] T[i,l,j,k]
    where T[i,l,j,k] = 1 if j==k else 0.
    
    But in the factorization convention:
    C = sum_r U_r * V_r * W_r (outer products)
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The tensor T has entries T[a, b, c] where a indexes (i,j), b indexes (k,l), c indexes (r = l*n+i)
    and T[a,b,c] = 1 if j==k else 0.
    
    For rank-1 decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    tensor = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    if j == k:
                        a = i * m + j
                        b = k * m + l
                        c = l * n + i
                        tensor[a, b, c] = 1
    return tensor

def _verify_decomposition(U, V, W, tensor, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    recon = np.zeros_like(tensor, dtype=np.float64)
    for r in range(rank):
        recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.max(np.abs(recon - tensor)) < tol

def _schoolbook_decomposition(n, m, p):
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

def _try_rank_reduced(n, m, p, target_rank):
    """Try to find a decomposition of given rank using integer programming or random search."""
    # This is a placeholder for more sophisticated search
    pass

def construct():
    deadline = time.time() + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # The matrix multiplication tensor for (n,m) x (m,p) has a known lower bound.
    # For 2x5 times 5x5, the rank is known to be 42 (or close to that).
    # Let's try to find a smaller rank using random search with integer/half-integer values.
    
    # Random search approach: generate random integer matrices and check
    # if they can form a valid decomposition.
    
    # Actually, let's use the fact that we can search over small integer values
    # in U, V, W and check if the tensor decomposition holds.
    
    # For a rank-r decomposition, we need:
    # sum_{c=0}^{r-1} U[a,c] * V[b,c] * W[c_row,c] == T[a,b,c_row] for all a,b,c_row
    
    # This is a system of equations. Let's try small ranks.
    
    # Known results: For 2x5 x 5x5, the border rank is 42, exact rank may be 42 or higher.
    # Let's try to verify rank 42 first.
    
    # Try a structured approach: use the schoolbook but merge some terms
    # This is complex, so let's try random search for small improvements.
    
    # For now, return schoolbook if no better found
    return best

# EVOLVE-BLOCK-END
