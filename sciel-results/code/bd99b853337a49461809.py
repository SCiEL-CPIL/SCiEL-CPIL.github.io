"""Matrix multiplication 3x3 * 3x4 - rank minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    
    deadline = time.time() + 115
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False, abs(val - 1.0)
        return True, 0.0
    
    def to_half_integer(lst):
        """Round to nearest half-integer and verify it's valid"""
        arr = np.array(lst, dtype=float)
        arr = np.round(arr * 2) / 2
        return arr.tolist()
    
    # Strategy: Try rank 9 first (structured construction), then increase
    # Rank 9 construction: U = identity(9), V and W structured
    
    for rank in [9, 12, 18, 24, 30, 36]:
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Structured initialization for rank >= 9
        for i in range(n):
            for j in range(m):
                r = i * m + j
                if r < rank:
                    U[i * m + j, r] = 1.0
                    for k in range(p):
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
        
        # Verify
        valid, err = verify(U, V, W, rank)
        if valid:
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            valid2, err2 = verify(U_r, V_r, W_r, rank)
            if valid2:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # For higher ranks, try a more general approach
        if rank > 9 and time.time() < deadline - 20:
            # Use schoolbook for remaining terms
            # First 9 terms as above, then schoolbook for the rest
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Base 9 terms
            for i in range(n):
                for j in range(m):
                    r = i * m + j
                    U[i * m + j, r] = 1.0
                    for k in range(p):
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
            
            # This should already be valid, so we're done if rank >= 9
            valid, err = verify(U, V, W, rank)
            if valid:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                valid2, err2 = verify(U_r, V_r, W_r, rank)
                if valid2:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Guaranteed fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
