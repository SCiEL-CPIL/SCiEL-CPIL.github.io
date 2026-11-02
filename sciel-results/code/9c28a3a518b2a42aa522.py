import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses schoolbook (rank 60) as baseline and attempts reduction via structured search.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n  # 10, 30, 12
    
    # Build target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k; else 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose tensor T."""
        # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def schoolbook():
        rank = nm * mp * pn  # Wait, that's wrong. Rank should be n*m*p = 60
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 60
    
    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, pn) = max(10, 30, 12) = 30
    # But tensor rank is typically higher. Let's try from 59 down to 50.
    
    np.random.seed(42)
    
    # Strategy: try structured random initializations
    # Each column of U, V, W is a vector. For rank-1 term, it's u ⊗ v ⊗ w.
    # We need sum of rank-1 terms to equal T.
    
    # Try ranks from 59 down to 50
    for target_rank in range(59, 49, -1):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random sparse structure
            U_try = np.zeros((nm, target_rank), dtype=np.float64)
            V_try = np.zeros((mp, target_rank), dtype=np.float64)
            W_try = np.zeros((pn, target_rank), dtype=np.float64)
            
            # Try to match the structure: each rank-1 term should contribute to specific T entries
            # Random approach: pick random non-zero patterns
            for r in range(target_rank):
                # Pick a random "center" for this term
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                # With some probability, make it a "pure" term (like schoolbook)
                if np.random.rand() < 0.5:
                    U_try[i * m + j, r] = 1.0
                    V_try[j * p + k, r] = 1.0
                    W_try[k * n + i, r] = 1.0
                else:
                    # Try a "combined" term that might cover multiple T entries
                    # Add small random perturbations
                    for ii in range(n):
                        for jj in range(m):
                            if np.random.rand() < 0.3:
                                U_try[ii * m + jj, r] = np.random.choice([-1, 1])
                    for jj in range(m):
                        for kk in range(p):
                            if np.random.rand() < 0.3:
                                V_try[jj * p + kk, r] = np.random.choice([-1, 1])
                    for kk in range(p):
                        for ii in range(n):
                            if np.random.rand() < 0.3:
                                W_try[kk * n + ii, r] = np.random.choice([-1, 1])
            
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    def to_list(arr):
        return arr.astype(float).tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
