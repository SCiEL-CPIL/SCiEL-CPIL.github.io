import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if exact, False otherwise."""
    rank = len(U[0])
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Convert to numpy for faster computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            # Expected: 1 if i==ip and j==jp and k==kp, else 0
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            
                            # Computed: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            
                            if abs(computed - expected) > 1e-9:
                                return False
    return True

def _verify_exact_fast(U, V, W, n, m, p):
    """Faster verification using numpy broadcasting."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape for efficient computation
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # For each (i,j,jp,k,ip,kp), check sum_r U[i*m+j,r]*V[jp*p+k,r]*W[kp*n+ip,r]
    # This is a 6D tensor contraction
    
    # Build the full tensor T[i,j,jp,k,ip,kp] = sum_r U[i*m+j,r]*V[jp*p+k,r]*W[kp*n+ip,r]
    # We can do this efficiently:
    
    # For each rank component r, we have outer products
    # Let's compute T as sum over r of (U_col_r outer V_col_r outer W_col_r) reshaped
    
    # U_col_r shape: (n, m)
    # V_col_r shape: (m, p)  
    # W_col_r shape: (p, n)
    
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    
    for r in range(rank):
        Ur = U_np[:, r].reshape(n, m)
        Vr = V_np[:, r].reshape(m, p)
        Wr = W_np[:, r].reshape(p, n)
        
        # For each (i,j,jp,k,ip,kp): Ur[i,j] * Vr[jp,k] * Wr[kp,ip]
        # This is Ur[i,j] * Vr[jp,k] * Wr[kp,ip]
        
        # Build as outer products
        # Ur: (n,m), Vr: (m,p), Wr: (p,n)
        # We need T[i,j,jp,k,ip,kp] += Ur[i,j] * Vr[jp,k] * Wr[kp,ip]
        
        # Use broadcasting
        # Ur[:, :, None, None, None, None] * Vr[None, None, :, :, None, None] * Wr[None, None, None, None, :, :]
        term = Ur[:, :, None, None, None, None] * Vr[None, None, :, :, None, None] * Wr[None, None, None, None, :, :]
        T += term
    
    # Now check against expected
    # Expected[i,j,jp,k,ip,kp] = 1 if i==ip and j==jp and k==kp else 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            if abs(T[i,j,jp,k,ip,kp] - expected) > 1e-9:
                                return False
    return True

def _try_rank_search(n, m, p, max_rank, max_entries, deadline, start_rank=None):
    """Search for a tensor decomposition of given rank with small integer entries."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    if start_rank is None:
        start_rank = max_rank
    
    # Try ranks from start_rank down to some minimum
    # For 2x3x4, the theoretical minimum is unknown but <= 24
    # Known results: rank(2x3x3) <= 18, rank(3x3x3) <= 23
    
    # We'll search for ranks below 24
    # Use a structured approach: try entries from {-1, 0, 1} or {-1, 0, 1, 2}
    
    entries = list(range(-max_entries, max_entries + 1))
    
    # For each candidate rank, we need to find U, V, W with entries in `entries`
    # such that the decomposition holds.
    
    # This is a hard search. Let's use a smarter approach:
    # Fix the structure based on known decompositions or use random search with verification.
    
    # Strategy: Random search with restarts
    import random
    
    best_result = None
    best_rank = 24  # schoolbook
    
    # First, get the schoolbook solution as baseline
    U_school = [[0] * 24 for _ in range(dim_U)]
    V_school = [[0] * 24 for _ in range(dim_V)]
    W_school = [[0] * 24 for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank solutions
    # Known: for 2x3x3, rank 18 is achievable. For 2x3x4, let's try ranks 18-23.
    
    for rank in range(start_rank, 17, -1):  # Try ranks down to 18
        if time.time() > deadline:
            break
        
        # Random search for this rank
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse matrices with small entries
            density = 0.3
            U = np.zeros((dim_U, rank))
            V = np.zeros((dim_V, rank))
            W = np.zeros((dim_W, rank))
            
            for r in range(rank):
                # Each column has some non-zero entries
                nz_U = np.random.choice(dim_U, size=max(1, int(dim_U * density)), replace=False)
                nz_V = np.random.choice(dim_V, size=max(1, int(dim_V * density)), replace=False)
                nz_W = np.random.choice(dim_W, size=max(1, int(dim_W * density)), replace=False)
                
                for idx in nz_U:
                    U[idx, r] = np.random.choice(entries)
                for idx in nz_V:
                    V[idx, r] = np.random.choice(entries)
                for idx in nz_W:
                    W[idx, r] = np.random.choice(entries)
            
            # Verify
            if _verify_exact_fast(U, V, W, n, m, p):
                # Found a valid decomposition!
                found = True
                best_rank = rank
                best_result = {
                    'U': U.tolist(),
                    'V': V.tolist(),
                    'W': W.tolist()
                }
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Return schoolbook
        return {
            'U': U_school,
            'V': V_school,
            'W': W_school
        }

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    deadline = time.time() + 120
    
    # Try to find a lower-rank decomposition
    result = _try_rank_search(n, m, p, max_rank=23, max_entries=2, deadline=deadline, start_rank=23)
    
    return result
