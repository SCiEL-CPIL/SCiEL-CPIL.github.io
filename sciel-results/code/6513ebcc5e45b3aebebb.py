import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    
    best_rank = 17
    best_U, best_V, best_W = None, None, None
    
    # Store the current rank 17 solution as fallback
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Copy rank 17 solution
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)
    
    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)
    
    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)
    
    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)
    
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)
    
    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)
    
    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)
    
    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)
    
    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)
    
    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)
    
    best_U, best_V, best_W = U17, V17, W17

    def compute_error(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    error += (s - 1.0) ** 2
        return error

    def is_exact(U, V, W, rank):
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

    # Try to find rank-16 by merging two simple terms that share structure
    def try_rank16_structured():
        """
        The rank-17 solution has 7 complex terms + 10 simple terms.
        Simple terms cover: (0,0,2),(0,1,2),(1,0,2),(1,1,2),
                           (0,2,0),(0,2,1),(0,2,2),(1,2,0),(1,2,1),(1,2,2)
        
        To get rank 16, we need to cover 10 entries with 9 terms.
        One term must cover two entries.
        
        Key observation: entries (0,2,0) and (0,2,1) both have i=0, j=2.
        So they share U-index = 0*3+2 = 2.
        Entry (0,2,0): V-index = 2*3+0 = 6, W-index = 0*2+0 = 0
        Entry (0,2,1): V-index = 2*3+1 = 7, W-index = 1*2+0 = 2
        
        A single term can't give 1 to both unless we use half-integers
        or restructure. Let's try a different pairing.
        
        Actually, let's try: can we modify the 7 complex terms slightly
        to absorb one of the simple terms?
        
        Alternative approach: use a linear algebra / optimization approach.
        Fix the structure and solve for entries.
        """
        if time.time() > deadline:
            return None, None, None
        
        # Strategy: Try to find a rank-16 decomposition by parameterizing
        # and solving a system of polynomial equations.
        # 
        # For 2x3x3, we have 18 tensor entries to match.
        # With rank 16: 6*16 + 9*16 + 6*16 = 336 unknowns.
        # 
        # Let's try a greedy/structured approach:
        # Use 7 terms for the "hard" part and 9 for the "easy" part,
        # but allow the 9 easy terms to have half-integer values that
        # let one term cover two entries.
        
        # Try merging (0,2,0) and (1,2,0): both have k=0, j=2
        # Entry (0,2,0): U[0,2]=1, V[2,0]=1, W[0,0]=1
        # Entry (1,2,0): U[1,2]=1, V[2,0]=1, W[0,1]=1
        # They share V-index = 2*3+0 = 6!
        # So a term with V[6]=1, U[2]=a, U[5]=b, W[0]=c, W[1]=d
        # gives: a*c at (0,2,0) and b*d at (1,2,0)
        # We need a*c = 1 and b*d = 1.
        # With a=b=1, c=d=1: works! But then we need separate W entries.
        # W[0,0] and W[0,1] are different indices, so this works with
        # U[0,2]=1, U[1,2]=1, V[2,0]=1, W[0,0]=1, W[0,1]=1
        # That's ONE term covering TWO entries!
        
        # Let's construct rank-16:
        U16 = np.zeros((n * m, 16))
        V16 = np.zeros((m * p, 16))
        W16 = np.zeros((p * n, 16))
        
        # Copy first 7 terms from rank-17
        for r in range(7):
            U16[:, r] = U17[:, r]
            V16[:, r] = V17[:, r]
            W16[:, r] = W17[:, r]
        
        # Term 7: covers (0,0,2) - U[0,0]=1, V[0,2]=1, W[2,0]=1
        U16[0*3+0, 7] = 1  # U[0,0]
        V16[0*3+2, 7] = 1  # V[0,2]
        W16[2*2+0, 7] = 1  # W[2,0]
        
        # Term 8: covers (0,1,2) - U[0,1]=1, V[1,2]=1, W[2,0]=1
        U16[0*3+1, 8] = 1  # U[0,1]
        V16[1*3+2, 8] = 1  # V[1,2]
        W16[2*2+0, 8] = 1  # W[2,0]
        
        # Term 9: covers (1,0,2) - U[1,0]=1, V[0,2]=1, W[2,1]=1
        U16[1*3+0, 9] = 1  # U[1,0]
        V16[0*3+2, 9] = 1  # V[0,2]
        W16[2*2+1, 9] = 1  # W[2,1]
        
        # Term 10: covers (1,1,2) - U[1,1]=1, V[1,2]=1, W[2,1]=1
        U16[1*3+1, 10] = 1  # U[1,1]
        V16[1*3+2, 10] = 1  # V[1,2]
        W16[2*2+1, 10] = 1  # W[2,1]
        
        # Term 11: MERGED - covers (0,2,0) AND (1,2,0)
        # U[0,2]=1, U[1,2]=1, V[2,0]=1, W[0,0]=1, W[0,1]=1
        U16[0*3+2, 11] = 1  # U[0,2]
        U16[1*3+2, 11] = 1  # U[1,2]
        V16[2*3+0, 11] = 1  # V[2,0]
        W16[0*2+0, 11] = 1  # W[0,0]
        W16[0*2+1, 11] = 1  # W[0,1]
        
        # Term 12: covers (0,2,1) - U[0,2]=1, V[2,1]=1, W[1,0]=1
        U16[0*3+2, 12] = 1  # U[0,2]
        V16[2*3+1, 12] = 1  # V[2,1]
        W16[1*2+0, 12] = 1  # W[1,0]
        
        # Term 13: covers (0,2,2) - U[0,2]=1, V[2,2]=1, W[2,0]=1
        U16[0*3+2, 13] = 1  # U[0,2]
        V16[2*3+2, 13] = 1  # V[2,2]
        W16[2*2+0, 13] = 1  # W[2,0]
        
        # Term 14: covers (1,2,1) - U[1,2]=1, V[2,1]=1, W[1,1]=1
        U16[1*3+2, 14] = 1  # U[1,2]
        V16[2*3+1, 14] = 1  # V[2,1]
        W16[1*2+1, 14] = 1  # W[1,1]
        
        # Term 15: covers (1,2,2) - U[1,2]=1, V[2,2]=1, W[2,1]=1
        U16[1*3+2, 15] = 1  # U[1,2]
        V16[2*3+2, 15] = 1  # V[2,2]
        W16[2*2+1, 15] = 1  # W[2,1]
        
        # Verify: check that this gives exactly 1 for all 18 entries
        if is_exact(U16, V16, W16, 16):
            return U16, V16, W16
        
        # If not exact, try other merge strategies
        # Try merging (0,2,1) and (1,2,1): both have j=2, k=1
        # Entry (0,2,1): U[0,2]=1, V[2,1]=1, W[1,0]=1
        # Entry (1,2,1): U[1,2]=1, V[2,1]=1, W[1,1]=1
        # They share V-index = 2*3+1 = 7!
        # Merge: U[0,2]=1, U[1,2]=1, V[2,1]=1, W[1,0]=1, W[1,1]=1
        
        U16b = np.zeros((n * m, 16))
        V16b = np.zeros((m * p, 16))
        W16b = np.zeros((p * n, 16))
        
        for r in range(7):
            U16b[:, r] = U17[:, r]
            V16b[:, r] = V17[:, r]
            W16b[:, r] = W17[:, r]
        
        # Terms 7-10: same as before
        U16b[0*3+0, 7] = 1; V16b[0*3+2, 7] = 1; W16b[2*2+0, 7] = 1
        U16b[0*3+1, 8] = 1; V16b[1*3+2, 8] = 1; W16b[2*2+0, 8] = 1
        U16b[1*3+0, 9] = 1; V16b[0*3+2, 9] = 1; W16b[2*2+1, 9] = 1
        U16b[1*3+1, 10] = 1; V16b[1*3+2, 10] = 1; W16b[2*2+1, 10] = 1
        
        # Term 11: covers (0,2,0)
        U16b[0*3+2, 11] = 1; V16b[2*3+0, 11] = 1; W16b[0*2+0, 11] = 1
        
        # Term 12: MERGED - covers (0,2,1) AND (1,2,1)
        U16b[0*3+2, 12] = 1; U16b[1*3+2, 12] = 1
        V16b[2*3+1, 12] = 1
        W16b[1*2+0, 12] = 1; W16b[1*2+1, 12] = 1
        
        # Term 13: covers (0,2,2)
        U16b[0*3+2, 13] = 1; V16b[2*3+2, 13] = 1; W16b[2*2+0, 13] = 1
        
        # Term 14: covers (1,2,0)
        U16b[1*3+2, 14] = 1; V16b[2*3+0, 14] = 1; W16b[0*2+1, 14] = 1
        
        # Term 15: covers (1,2,2)
        U16b[1*3+2, 15] = 1; V16b[2*3+2, 15] = 1; W16b[2*2+1, 15] = 1
        
        if is_exact(U16b, V16b, W16b, 16):
            return U16b, V16b, W16b
        
        # Try merging (0,2,2) and (1,2,2): both have j=2, k=2
        # Entry (0,2,2): U[0,2]=1, V[2,2]=1, W[2,0]=1
        # Entry (1,2,2): U[1,2]=1, V[2,2]=1, W[2,1]=1
        # They share V-index = 2*3+2 = 8!
        
        U16c = np.zeros((n * m, 16))
        V16c = np.zeros((m * p, 16))
        W16c = np.zeros((p * n, 16))
        
        for r in range(7):
            U16c[:, r] = U17[:, r]
            V16c[:, r] = V17[:, r]
            W16c[:, r] = W17[:, r]
        
        U16c[0*3+0, 7] = 1; V16c[0*3+2, 7] = 1; W16c[2*2+0, 7] = 1
        U16c[0*3+1, 8] = 1; V16c[1*3+2, 8] = 1; W16c[2*2+0, 8] = 1
        U16c[1*3+0, 9] = 1; V16c[0*3+2, 9] = 1; W16c[2*2+1, 9] = 1
        U16c[1*3+1, 10] = 1; V16c[1*3+2, 10] = 1; W16c[2*2+1, 10] = 1
        
        # Term 11: covers (0,2,0)
        U16c[0*3+2, 11] = 1; V16c[2*3+0, 11] = 1; W16c[0*2+0, 11] = 1
        
        # Term 12: covers (0,2,1)
        U16c[0*3+2, 12] = 1; V16c[2*3+1, 12] = 1; W16c[1*2+0, 12] = 1
        
        # Term 13: MERGED - covers (0,2,2) AND (1,2,2)
        U16c[0*3+2, 13] = 1; U16c[1*3+2, 13] = 1
        V16c[2*3+2, 13] = 1
        W16c[2*2+0, 13] = 1; W16c[2*2+1, 13] = 1
        
        # Term 14: covers (1,2,0)
        U16c[1*3+2, 14] = 1; V16c[2*3+0, 14] = 1; W16c[0*2+1, 14] = 1
        
        # Term 15: covers (1,2,1)
        U16c[1*3+2, 15] = 1; V16c[2*3+1, 15] = 1; W16c[1*2+1, 15] = 1
        
        if is_exact(U16c, V16c, W16c, 16):
            return U16c, V16c, W16c
        
        return None, None, None
    
    # Try the structured rank-16 search
    U16, V16, W16 = try_rank16_structured()
    if U16 is not None:
        best_rank = 16
        best_U, best_V, best_W = U16, V16, W16
    
    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
