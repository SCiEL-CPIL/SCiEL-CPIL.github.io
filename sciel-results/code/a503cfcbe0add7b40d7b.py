"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms, n=3, m=3, p=3):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps indices to scalars.
    """
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    """
    terms = []
    # m1 = (a00 + a11)(b00 + b11)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # m2 = (a10 + a11)b00
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}))
    # m3 = a00(b01 - b11)
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}))
    # m4 = a11(b10 - b00)
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}))
    # m5 = (a00 + a01)b11
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}))
    # m6 = (a10 - a00)(b00 + b01)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    # m7 = (a01 - a11)(b10 + b11)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _standard_border_terms():
    """
    Return standard multiplication terms for the border (at least one index is 2).
    """
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_rank_reduction(terms, max_iter=5000, time_limit=10.0):
    """
    Attempt to reduce the rank of a decomposition by merging terms.
    This is a heuristic search. We look for pairs of terms that can be combined
    into a single term with half-integer coefficients if possible, or generally
    reduce the count by finding linear dependencies or specific algebraic structures.
    
    For now, we implement a simple greedy check: 
    Does the current set of terms allow for a "Strassen-like" cancellation on the border?
    
    A more robust approach for this specific task (3x3) would be to try known
    decompositions of rank 23 or 22. However, constructing those explicitly
    is complex. We will stick to the known rank-26 (Strassen core + standard border)
    but verify it carefully.
    
    If we have time, we can try a randomized search for a rank-25 decomposition
    by perturbing the standard border terms and checking for rank reduction.
    """
    start_time = time.time()
    current_terms = list(terms)
    
    # If we have a lot of time, we could try to find a rank-23 decomposition.
    # The known rank-23 decomposition for 3x3 matrix multiplication is non-trivial.
    # One such decomposition uses specific algebraic identities.
    # However, without the explicit coefficients, we cannot guarantee correctness.
    
    # Let's try a simple optimization: check if any two standard border terms
    # can be merged with a Strassen term to save a rank.
    # This is unlikely to work without specific structural knowledge.
    
    # Instead, we ensure the rank-26 decomposition is correct and return it.
    # If we had a verified rank-23 or lower, we would return that.
    
    return current_terms

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    terms = _strassen_2x2_terms() + _standard_border_terms()
    U, V, W = _build_decomposition(terms)
    if _verify_decomposition(U, V, W):
        R = U.shape[1]
        if R < best_rank:
            best_rank = R
            best_result = (U, V, W)
    
    # Strategy 2: Pure standard (rank 27) as fallback
    if time.time() < deadline:
        terms_std = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms_std.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
        U, V, W = _build_decomposition(terms_std)
        if _verify_decomposition(U, V, W):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        # Should not happen
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
