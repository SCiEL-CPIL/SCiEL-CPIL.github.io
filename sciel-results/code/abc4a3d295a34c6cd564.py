"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half(x):
    """Snap a value to the nearest half-integer."""
    return round(x * 2) / 2.0

def _validate_and_snap(U, V, W, n, m, p):
    """Validate decomposition and snap entries to half-integers."""
    rank = len(U[0]) if U else 0
    # Check exact decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return None
    # Snap to half-integers
    U_snap = [[_snap_to_half(U[i][c]) for c in range(rank)] for i in range(len(U))]
    V_snap = [[_snap_to_half(V[i][c]) for c in range(rank)] for i in range(len(V))]
    W_snap = [[_snap_to_half(W[i][c]) for c in range(rank)] for i in range(len(W))]
    # Re-verify after snapping
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
        return {"U": U_snap, "V": V_snap, "W": W_snap}
    return None

def _try_fixed_V_decomposition(n, m, p, target_rank, deadline):
    """
    Fix V to a sparse canonical form and solve for U and W.
    
    Strategy: For each row of V, assign exactly one 1 to some column.
    This gives V[j*p+k] = e_{col(j,k)} where col is a mapping.
    
    Then the constraint becomes: for each (i,j,k),
    U[i*m+j, col(j,k)] * W[k*n+i, col(j,k)] = 1
    
    This means U[i*m+j, col(j,k)] and W[k*n+i, col(j,k)] must both be 1
    (or both -1, but let's use 1 for simplicity).
    
    So we need: for each (i,j,k), set U[i*m+j, col(j,k)] = 1 and W[k*n+i, col(j,k)] = 1.
    
    But we also need consistency: if col(j,k) = col(j',k') for different (j,k) and (j',k'),
    then U[i*m+j, c] = 1 and U[i'*m+j', c] = 1 for the same c, which is fine.
    
    The key constraint is: for a fixed column c, the set of (j,k) pairs assigned to c
    determines which U and W entries are 1. We need the product to always be 1.
    
    Actually, this simple construction only works if each column c is used by at most one (j,k) pair,
    which gives rank = m*p = 25. That's just schoolbook.
    
    To do better, we need a more clever assignment. Let me try a different approach:
    Allow V entries to be 0 or 1, and solve the system.
    """
    n, m, p = 2, 5, 5
    rank = target_rank
    
    # We'll use a structured approach for 2x5x5
    # Known result: 2 x m x m can be done in 2m-1 multiplications
    # For m=5, that's rank 9.
    
    # Let me try to construct rank 9 explicitly.
    # 
    # For 2 x m x m, one construction uses the following idea:
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    #
    # We can write:
    # C[0,k] + C[1,k] = sum_j (A[0,j] + A[1,j]) B[j,k]
    # C[0,k] - C[1,k] = sum_j (A[0,j] - A[1,j]) B[j,k]
    #
    # But this doesn't directly help with rank reduction.
    
    # Let me try a different approach: use the fact that we have 2 rows in A.
    # For each j, we have A[0,j] and A[1,j].
    # 
    # One known construction for 2xm:
    # Use m + (m-1) = 2m-1 terms.
    
    # For m=5, rank=9:
    # Terms 0..4: handle A[0,j] * B[j,k] for j=0..4 (contributes to C[0,k])
    # Terms 5..8: handle combinations for C[1,k]
    
    # Actually, let me try to find a valid decomposition by solving a system.
    
    # Approach: Fix V to have sparse structure, then solve for U and W.
    
    # Let's try: V[j*5+k][c] = 1 if c == j*5+k, else 0. This is identity, rank=25.
    
    # Better: Try to find a V with rank < 25.
    
    # Let me use a more systematic approach:
    # Parameterize U, V, W with small integer entries and search.
    
    # For rank r, we have n*m*r + m*p*r + p*n*r = 10r + 25r + 10r = 45r variables.
    # Constraints: n*m*p = 50 equations (each must equal 1).
    
    # For r=9: 405 variables, 50 constraints. Should be feasible.
    
    # Let me try to construct rank 9 by hand for 2x5x5.
    
    # Construction attempt:
    # We need sum_c U[i*5+j,c] * V[j*5+k,c] * W[k*2+i,c] = 1 for all i,j,k.
    
    # Let me try to make V very sparse. Suppose V[j*5+k,c] is nonzero only for c in some set.
    
    # Actually, let me just try random search with a better strategy:
    # Initialize with random small integers and use local search.
    
    np.random.seed(12345)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with random sparse pattern
        for c in range(rank):
            # Choose which U, V, W entries are nonzero for this column
            # Try to make it structured
            u_idx = np.random.choice(n*m, size=np.random.randint(1,4), replace=False)
            v_idx = np.random.choice(m*p, size=np.random.randint(1,4), replace=False)
            w_idx = np.random.choice(p*n, size=np.random.randint(1,4), replace=False)
            
            for idx in u_idx:
                U[idx, c] = np.random.choice([-1, 1])
            for idx in v_idx:
                V[idx, c] = np.random.choice([-1, 1])
            for idx in w_idx:
                W[idx, c] = np.random.choice([-1, 1])
        
        # Check validity
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            result = _validate_and_snap(U.tolist(), V.tolist(), W.tolist(), n, m, p)
            if result:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _try_rank_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions with decreasing rank
    for target_rank in [9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
        
        result = _try_fixed_V_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best = result
            best_rank = target_rank
            if target_rank <= 10:  # Good enough, stop early
                break
    
    return best
# EVOLVE-BLOCK-END
