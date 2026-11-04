"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random
import itertools
from collections import defaultdict

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor.
    T[a, b, c] where a indexes (i,j) for A, b indexes (k,l) for B, c indexes (r = l*n+i) for C
    T[a,b,c] = 1 if j==k else 0
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
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
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

def _try_structured_merge(n, m, p, deadline):
    """Try to find a lower-rank decomposition by merging schoolbook terms using Strassen-like identities."""
    tensor = _build_tensor(n, m, p)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Schoolbook terms: each term r has (i,j) in U, (j,l) in V, (l,i) in W
    # Term index: r = i*m*p + j*p + l where i in [0,n), j in [0,m), l in [0,p)
    # Wait, let me re-index: for term (i,j,l): U[i*m+j]=1, V[j*p+l]=1, W[l*n+i]=1
    
    # Build list of terms
    terms = []
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                terms.append((a, b, c))
    
    schoolbook_rank = len(terms)
    
    # Try to find linear dependencies among the rank-1 tensors
    # Each term is a rank-1 tensor u_r ⊗ v_r ⊗ w_r
    # We want to find coefficients c_r such that sum c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    # with not all c_r = 0, and some c_r = 0 to reduce rank
    
    # Build the tensor as a matrix (flattened)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For schoolbook: T[a,b,c] = sum_{r: a_r=a, b_r=b, c_r=c} 1
    
    # Let's try a different approach: use random linear combinations
    # to find a lower-rank decomposition
    
    best_rank = schoolbook_rank
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try ranks from schoolbook-1 down
    # For each target rank, try to find U,V,W with entries in {0, 1/2, -1/2, 1, -1}
    
    # Better approach: use the fact that the tensor has a specific structure
    # and try to find a decomposition using block methods
    
    # For 2x5x5, let's try to use a 2x2x2 Strassen block decomposition
    # Split the 5 into blocks: e.g., 2+3 or 1+1+3 etc.
    
    # Actually, let's try a direct search with a better value set
    values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), 
              Fraction(2), Fraction(-2), Fraction(3), Fraction(-3)]
    
    for target_rank in range(schoolbook_rank - 1, max(40, schoolbook_rank - 10), -1):
        if time.time() > deadline:
            break
        
        # Try random search with more iterations for this rank
        found = _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=200)
        if found is not None:
            best_rank = target_rank
            best_result = found
            break
    
    return best_rank, best_result

def _random_search_for_rank(n, m, p, target_rank, tensor, deadline, max_iters=1000):
    """Random search for a rank-r decomposition with integer/half-integer entries."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Values to try: integers and half-integers in a small range
    values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate random U, V, W with small integer/half-integer entries
        U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n_m)]
        V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m_p)]
        W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p_n)]
        
        # Convert to float for verification
        U_f = np.array([[float(x) for x in row] for row in U])
        V_f = np.array([[float(x) for x in row] for row in V])
        W_f = np.array([[float(x) for x in row] for row in W])
        
        recon = np.zeros((n_m, m_p, p_n), dtype=np.float64)
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U_f[:, r], V_f[:, r], W_f[:, r])
        
        if np.max(np.abs(recon - tensor)) < 1e-9:
            # Found a valid decomposition - convert back to fractions
            return {"U": U, "V": V, "W": W}
    
    return None

def _structured_rank_search(n, m, p, deadline):
    """Try structured approaches to find lower rank decompositions."""
    tensor = _build_tensor(n, m, p)
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try merging terms in schoolbook decomposition
    # The key insight: some terms might be linearly dependent
    
    # For now, try a simple approach: check if we can reduce by 1
    # This is a placeholder for more sophisticated structured search
    
    return best_rank, best_result

def construct():
    deadline = time.time() + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Try structured merge
    if time.time() < deadline - 10:
        new_rank, new_result = _try_structured_merge(n, m, p, deadline - 10)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
    
    return best_result

# EVOLVE-BLOCK-END
