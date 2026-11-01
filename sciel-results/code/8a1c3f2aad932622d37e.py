"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    def to_list(mat):
        return mat.tolist()
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the tensor."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        # Compute the tensor decomposition
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i*m+j, j*p+k, k*n+i]
        # For efficiency, check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if actual != expected:
                        return False
        # Also verify that entries NOT in the tensor are zero
        # The tensor has exactly n*m*p non-zero entries
        # Check a sample of other entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This is already checked above
                    pass
        # Check some off-diagonal entries
        for _ in range(100):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            # Decode indices
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            # This entry is in the tensor iff j1 == j2 and i1 == i2 and k1 == k2
            if not (j1 == j2 and i1 == i2 and k1 == k2):
                actual = 0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if actual != 0:
                    return False
        return True
    
    # Schoolbook decomposition as baseline
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # Strategy: Try to combine pairs of schoolbook terms
    # A rank-1 tensor U*V*W can have multiple non-zero entries if U, V, W have multiple non-zeros
    
    # For a given set of (i,j,k) tuples, can we find U, V, W such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k) in the set
    # and 0 elsewhere?
    
    # Let's try to find pairs that can be combined
    # Key observation: If two terms (i1,j1,k1) and (i2,j2,k2) share the same j,
    # we might be able to combine them.
    
    # Actually, let's try a more systematic approach:
    # For each possible rank r from 41 down to some lower bound, try to find a decomposition.
    # But this is computationally expensive.
    
    # Instead, let's try to find specific patterns:
    # Pattern 1: Two terms with same i and j, different k
    #   (i,j,k1) and (i,j,k2)
    #   U[i*m+j, r] * V[j*p+k1, r] * W[k1*n+i, r] = 1
    #   U[i*m+j, r] * V[j*p+k2, r] * W[k2*n+i, r] = 1
    #   This requires V[j*p+k1, r] * W[k1*n+i, r] = V[j*p+k2, r] * W[k2*n+i, r]
    #   And U[i*m+j, r] = 1/(that value)
    #   But we need integer/half-integer entries, so this is restrictive.
    
    # Let's try a randomized search for rank 41
    def try_rank(r, max_attempts=10000):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        # Initialize with random sparse vectors
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n * m, r), dtype=np.float64)
            V = np.zeros((m * p, r), dtype=np.float64)
            W = np.zeros((p * n, r), dtype=np.float64)
            
            # Randomly assign non-zero entries
            for r_idx in range(r):
                # Choose random positions
                u_pos = np.random.randint(0, n * m, size=np.random.randint(1, 4))
                v_pos = np.random.randint(0, m * p, size=np.random.randint(1, 4))
                w_pos = np.random.randint(0, p * n, size=np.random.randint(1, 4))
                
                for pos in u_pos:
                    U[pos, r_idx] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1, 2])
                for pos in v_pos:
                    V[pos, r_idx] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1, 2])
                for pos in w_pos:
                    W[pos, r_idx] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1, 2])
            
            # Verify
            if verify(U, V, W):
                # Convert to integer/half-integer if possible
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                if verify(U_int, V_int, W_int):
                    return U_int, V_int, W_int
        
        return None
    
    # Try rank 41 first
    result = try_rank(41, max_attempts=5000)
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # If rank 41 fails, return schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
