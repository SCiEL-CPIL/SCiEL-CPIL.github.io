"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 72
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by merging columns
    # Strategy: for each pair of columns, check if they can be merged
    # (i.e., their product patterns don't conflict)
    
    # First, try a structured approach:
    # The key constraint: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # With binary U,V,W, each column c "covers" a set of (i,j,k) triples.
    # We need each triple to be covered exactly once.
    
    # Try greedy column merging with backtracking
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank_sb
    
    # Precompute the "signature" of each column: which (i,j,k) it covers
    def get_signature(U_mat, V_mat, W_mat, col, n, m, p):
        sig = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if U_mat[i*m+j, col] * V_mat[j*p+k, col] * W_mat[k*n+i, col] != 0:
                        sig.add((i, j, k))
        return sig
    
    # Try to merge columns greedily
    merged = True
    while merged and time.time() < deadline - 5:
        merged = False
        for c1 in range(current_rank):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, current_rank):
                # Check if columns c1 and c2 can be merged
                # They can be merged if for every (i,j,k), at most one of them
                # has all three entries non-zero
                sig1 = get_signature(current_U, current_V, current_W, c1, n, m, p)
                sig2 = get_signature(current_U, current_V, current_W, c2, n, m, p)
                if not sig1.isdisjoint(sig2):
                    # They overlap, can't simply merge
                    continue
                # Try merging: new column = element-wise product? No.
                # Actually, we need: for each (i,j,k), the sum stays 1.
                # If sig1 and sig2 are disjoint, we can't merge them into one
                # column because a single column would need to cover both sets,
                # but the product U*V*W for a single column is either 0 or 1
                # for each (i,j,k), so it can only cover a "box" structure.
                # 
                # Alternative: try to find a new column that covers sig1 ∪ sig2
                # This requires U[i*m+j,new]=1 for all (i,j) in sig1∪sig2's i,j pairs,
                # V[j*p+k,new]=1 for all (j,k) in sig1∪sig2's j,k pairs,
                # W[k*n+i,new]=1 for all (k,i) in sig1∪sig2's k,i pairs,
                # AND for all (i,j,k) NOT in sig1∪sig2, at least one of U,V,W is 0.
                
                # This is complex. Let me try a different approach:
                # Use a local search to minimize rank directly.
                break
    
    # Try a more direct approach: parameterized search
    # For small ranks, try to find solutions with entries in {0, 1/2, 1}
    
    # Lower bound: rank >= max(n*m, m*p, p*n) / something
    # Actually, a simple lower bound: for fixed (i,j), the equation
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
    # means that the vector (V[j*p+k,c]*W[k*n+i,c])_k dotted with U[i*m+j,:] = 1 for all i
    # This gives constraints on the rank.
    
    # Let me try a numerical approach: set up the equations and solve
    # For rank r, we have r*(n*m + m*p + p*n) variables and n*m*p equations.
    # For r=53: 53*(9+24+24)=2808 variables, 72 equations. Very underdetermined.
    # But we need integer/half-integer solutions.
    
    # Try a randomized local search starting from schoolbook
    best_solution = _local_search(n, m, p, deadline)
    if best_solution is not None:
        return best_solution
    
    # Return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _local_search(n, m, p, deadline):
    """Local search to find lower-rank decomposition."""
    rank_sb = n * m * p
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing columns that are redundant
    # A column c is redundant if removing it and adjusting others still works.
    # This is hard to check directly. Instead, try a different approach.
    
    # Approach: use the fact that for 3x3x8, we can potentially use
    # a decomposition based on the structure of the 3x3 part.
    
    # Known result: 3x3x3 has rank 23 (Landsberg 1993).
    # For 3x3x8, we can think of it as 3x3x3 * (1x1x8) in some sense,
    # but that gives rank 23*8=184 which is worse.
    
    # Better: view 3x3x8 as computing 8 outputs, each a 3x3x1 multiplication.
    # The 3x3x1 tensor has rank 9 (schoolbook is optimal for this case).
    # But we can share structure across the 8 outputs.
    
    # Let me try a constructive approach:
    # U[i*m+j, c] depends only on (i,j), so it's the same for all k.
    # V[j*p+k, c] and W[k*n+i, c] depend on k.
    
    # For a fixed c, let u_c[i,j] = U[i*m+j, c], v_c[j,k] = V[j*p+k, c], w_c[k,i] = W[k*n+i, c].
    # The constraint is: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    # This is a 3-tensor decomposition problem. Let me try to find a low-rank
    # solution using a structured ansatz.
    
    # Ansatz: use rank-3 blocks for the i dimension.
    # Since n=3, we can use a 3x3x3 decomposition (rank 23) and then
    # extend to 8 columns.
    
    # Actually, let me just try a numerical optimization approach.
    # Set up the problem as minimizing ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2
    # subject to the entries being in {0, 1/2, 1}.
    
    # For now, return None to use schoolbook.
    return None

# EVOLVE-BLOCK-END
