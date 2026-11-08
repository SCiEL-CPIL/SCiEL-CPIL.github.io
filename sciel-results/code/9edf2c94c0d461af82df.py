"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_numpy(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B using NumPy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # and check required and spurious entries
    # Efficient: compute T as a 3D array via einsum or broadcasting
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a tensor contraction. We can compute it as:
    # For each r, outer product of U[:,r], V[:,r], W[:,r]
    # But that's rank * n*m * m*p * p*n which is fine for small sizes
    
    # More efficient: use einsum
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    
    # Check required entries: T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check spurious entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                i_prime = c % n
                k_prime = c // n
                
                if i == i_prime and j == j_prime and k == k_prime:
                    continue
                
                if abs(T[a, b, c]) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10."""
    rank = 55
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # Direct terms for row 2 (A02, A12) with B2
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _try_rank54_random(n, m, p, deadline):
    """
    Attempt to find a rank-54 decomposition by starting from rank-55 
    and trying to eliminate one term through sparse linear algebra tricks.
    """
    rank = 54
    start_time = time.time()
    
    if time.time() > deadline - 10:
        return None
    
    # Get the rank-55 decomposition as a starting point
    res55 = _rank55(n, m, p)
    U55 = np.array(res55["U"], dtype=np.float64)
    V55 = np.array(res55["V"], dtype=np.float64)
    W55 = np.array(res55["W"], dtype=np.float64)
    rank55 = 55
    
    # Strategy: try to set one column of each factor to zero and solve for the
    # remaining entries. We'll try each possible column to eliminate.
    # For column r to eliminate, we need:
    #   sum_{r' != r} U[a,r']*V[b,r']*W[c,r'] = T[a,b,c] for all (a,b,c)
    # where T is the matrix multiplication tensor.
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # For each candidate column r to eliminate, check if the remaining columns
    # can span the tensor. This is a linear system:
    # We need to find U', V', W' (with column r = 0) such that
    # sum_{r' != r} U'[a,r']*V'[b,r']*W'[c,r'] = T[a,b,c]
    
    # This is a bilinear/trilinear system. A simpler approach:
    # Fix U and V, solve for W, or use iterative methods.
    
    # Alternative: try to find a linear dependence among the 55 terms.
    # If there's a non-trivial linear combination of the 55 rank-1 tensors
    # that sums to zero, we can remove one term.
    
    # Let's try: for each pair of columns (r1, r2), check if we can replace
    # both with a single column that produces the same tensor contribution.
    # This is equivalent to checking if the sum of two rank-1 tensors is
    # also rank-1 (which is rare) or if we can find a 3-term combination
    # that's rank-1.
    
    # More practical: try random perturbations and check if we can reduce rank.
    # Start with rank-55, randomly set some entries to 0, and check if the
    # tensor is still correctly decomposed with the remaining non-zero entries.
    
    # Actually, let's try a different approach: use the fact that the rank-55
    # decomposition has a specific block structure. Try to merge blocks.
    
    # The rank-55 uses 5 blocks of 11 terms each. Let's see if we can
    # restructure to use fewer terms per block.
    
    # For now, try a simple approach: check if any two columns are linearly
    # dependent in a way that allows removal.
    
    # For each column r, check if U[:,r] is in the span of other U columns,
    # V[:,r] is in the span of other V columns, AND W[:,r] is in the span of
    # other W columns simultaneously. This is unlikely but worth checking.
    
    U_cols = U55.T  # shape (55, n*m)
    V_cols = V55.T  # shape (55, m*p)
    W_cols = W55.T  # shape (55, p*n)
    
    for r in range(rank55):
        if time.time() > deadline - 15:
            return None
        
        # Check if column r of U is in the span of others
        U_others = np.delete(U_cols, r, axis=0)
        V_others = np.delete(V_cols, r, axis=0)
        W_others = np.delete(W_cols, r, axis=0)
        
        # Check if U[:,r] is in span of U_others
        try:
            # Solve U_others.T * x = U[:,r]
            sol_U, residuals_U, rank_U, _ = np.linalg.lstsq(U_others, U_cols[r], rcond=None)
            if residuals_U and residuals_U[0] > 1e-9:
                continue
        except:
            continue
        
        try:
            sol_V, residuals_V, rank_V, _ = np.linalg.lstsq(V_others, V_cols[r], rcond=None)
            if residuals_V and residuals_V[0] > 1e-9:
                continue
        except:
            continue
        
        try:
            sol_W, residuals_W, rank_W, _ = np.linalg.lstsq(W_others, W_cols[r], rcond=None)
            if residuals_W and residuals_W[0] > 1e-9:
                continue
        except:
            continue
        
        # All three are in span! Try to construct the rank-54 decomposition.
        # This is complex; for now, just verify that removing column r still works
        # (it won't in general, but let's check).
        U54 = np.delete(U55, r, axis=1)
        V54 = np.delete(V55, r, axis=1)
        W54 = np.delete(W55, r, axis=1)
        
        if _verify_numpy(U54.tolist(), V54.tolist(), W54.tolist(), n, m, p, 54):
            return {"U": U54.tolist(), "V": V54.tolist(), "W": W54.tolist()}
    
    # Try a more aggressive search: random sparse modifications
    np.random.seed(42)
    for attempt in range(100):
        if time.time() > deadline - 15:
            return None
        
        # Start from rank-55 and try to zero out random entries
        U_mod = U55.copy()
        V_mod = V55.copy()
        W_mod = W55.copy()
        
        # Try to make one column nearly zero
        r = np.random.randint(rank55)
        # Scale down column r
        U_mod[:, r] *= 0.0
        V_mod[:, r] *= 0.0
        W_mod[:, r] *= 0.0
        
        # Check if this still works (it won't unless column r was redundant)
        if _verify_numpy(U_mod.tolist(), V_mod.tolist(), W_mod.tolist(), n, m, p, 54):
            return {"U": U_mod.tolist(), "V": V_mod.tolist(), "W": W_mod.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    # First, try the known rank-55 decomposition
    try:
        res55 = _rank55(n, m, p)
        if time.time() < deadline - 5:
            if _verify_numpy(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                # Optionally try to improve to rank 54
                if time.time() < deadline - 10:
                    res54 = _try_rank54_random(n, m, p, deadline)
                    if res54 is not None:
                        if _verify_numpy(res54["U"], res54["V"], res54["W"], n, m, p, 54):
                            return res54
                return res55
    except Exception:
        pass
    
    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
