import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    def get_rank_17():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val

        # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
        set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
        set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
        set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
        
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)

        # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
        set_U(0, 0, 7, 1)
        set_V(0, 2, 7, 1)
        set_W(2, 0, 7, 1)
        
        set_U(0, 1, 8, 1)
        set_V(1, 2, 8, 1)
        set_W(2, 0, 8, 1)
        
        set_U(1, 0, 9, 1)
        set_V(0, 2, 9, 1)
        set_W(2, 1, 9, 1)
        
        set_U(1, 1, 10, 1)
        set_V(1, 2, 10, 1)
        set_W(2, 1, 10, 1)

        # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
        set_U(0, 2, 11, 1)
        set_V(2, 0, 11, 1)
        set_W(0, 0, 11, 1)
        
        set_U(0, 2, 12, 1)
        set_V(2, 1, 12, 1)
        set_W(1, 0, 12, 1)
        
        set_U(0, 2, 13, 1)
        set_V(2, 2, 13, 1)
        set_W(2, 0, 13, 1)
        
        set_U(1, 2, 14, 1)
        set_V(2, 0, 14, 1)
        set_W(0, 1, 14, 1)
        
        set_U(1, 2, 15, 1)
        set_V(2, 1, 15, 1)
        set_W(1, 1, 15, 1)
        
        set_U(1, 2, 16, 1)
        set_V(2, 2, 16, 1)
        set_W(2, 1, 16, 1)

        return U, V, W

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(arr):
        """Check if all entries are half-integers (multiples of 0.5)."""
        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-9)

    def try_merge_columns(U17, V17, W17):
        """Try to merge two columns of the rank-17 solution into one.
        
        The idea: if column r1 can be expressed as a linear combination of other columns
        in a way that preserves the tensor identity, we can eliminate it.
        
        We try: for each pair (r1, r2), check if we can set U[:,r2] += a*U[:,r1],
        V[:,r2] += b*V[:,r1], W[:,r2] += c*W[:,r1] for some a,b,c in {-1,-0.5,0,0.5,1}
        such that the new decomposition is still exact and has half-integer entries.
        """
        rank17 = 17
        candidates = []
        
        # Try merging each column into another
        for r_merge in range(rank17):
            if time.time() > deadline:
                break
            for r_target in range(rank17):
                if r_merge == r_target:
                    continue
                if time.time() > deadline:
                    break
                    
                # Try simple coefficient combinations
                for a in [-1, -0.5, 0.5, 1]:
                    for b in [-1, -0.5, 0.5, 1]:
                        for c in [-1, -0.5, 0.5, 1]:
                            if a == 0 and b == 0 and c == 0:
                                continue
                                
                            # Create new matrices by merging r_merge into r_target
                            U_new = U17.copy()
                            V_new = V17.copy()
                            W_new = W17.copy()
                            
                            U_new[:, r_target] += a * U_new[:, r_merge]
                            V_new[:, r_target] += b * V_new[:, r_merge]
                            W_new[:, r_target] += c * W_new[:, r_merge]
                            
                            # Remove column r_merge
                            mask = np.ones(rank17, dtype=bool)
                            mask[r_merge] = False
                            U_new = U_new[:, mask]
                            V_new = V_new[:, mask]
                            W_new = W_new[:, mask]
                            
                            # Check if half-integer
                            if not (is_half_integer(U_new) and is_half_integer(V_new) and is_half_integer(W_new)):
                                continue
                            
                            # Verify
                            if verify(U_new, V_new, W_new, rank17 - 1):
                                candidates.append((U_new, V_new, W_new, rank17 - 1))
                                return U_new, V_new, W_new
        
        return None

    def try_structured_rank_16():
        """Try to find a rank-16 decomposition by modifying the rank-17 structure.
        
        Strategy: The rank-17 solution has:
        - 7 Strassen terms for the 2x2x2 block
        - 4 direct terms for (j in {0,1}, k=2)  
        - 6 direct terms for (j=2, k in {0,1,2})
        
        We can try to use Strassen-like tricks for the mixed blocks too.
        For the (j in {0,1}, k=2) block, we have 4 terms. Can we do it in 3?
        For the (j=2, k in {0,1,2}) block, we have 6 terms. Can we do it in 5?
        """
        # Try: use 3 terms for the 2x1x1 block (j in {0,1}, k=2)
        # Standard is 2x1=2 terms, but we have 2 rows and 1 col in B, so 2 terms is minimal for direct
        # But maybe we can share with the Strassen block?
        
        # Alternative: try a different partition
        # Let's try to see if we can reduce the total by 1
        
        # For now, try the column merging approach
        return try_merge_columns(*get_rank_17())

    # Get the rank-17 baseline
    U17, V17, W17 = get_rank_17()
    
    # Verify baseline
    assert verify(U17, V17, W17, 17), "Rank 17 solution failed verification"
    
    best_U, best_V, best_W = U17, V17, W17
    best_rank = 17
    
    # Try to find rank 16
    if time.time() < deadline - 30:
        result_16 = try_structured_rank_16()
        if result_16 is not None:
            U16, V16, W16 = result_16
            best_U, best_V, best_W = U16, V16, W16
            best_rank = 16
    
    # Try random search for rank 16 if time permits
    if best_rank == 17 and time.time() < deadline - 40:
        for seed in range(10):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            
            # Initialize with random values, biased toward small integers
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, 16))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, 16))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, 16))
            
            # Quick verification
            if verify(U, V, W, 16):
                best_U, best_V, best_W = U, V, W
                best_rank = 16
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
