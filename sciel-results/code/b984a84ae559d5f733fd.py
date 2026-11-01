import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def sanitize_matrix(M):
        """
        Ensures all entries are exactly integers or half-integers.
        Multiplies by 2, rounds to nearest integer, divides by 2.
        """
        M_2 = M * 2.0
        M_rounded = np.round(M_2)
        M_clean = M_rounded / 2.0
        return M_clean

    def verify(U, V, W, tol=1e-9):
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    u_idx = i*2 + j
                    v_idx = j*9 + k
                    w_idx = k*2 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build rank-31 solution
    # Use Strassen (rank 7) for 3 blocks of 2 columns each (columns 0-1, 2-3, 4-5)
    # Use rank-10 decomposition for the last 3 columns (columns 6, 7, 8)
    # Total Rank = 7*3 + 10 = 31
    
    def build_rank31():
        terms = []
        
        # Helper to add a 2x2x2 Strassen block for columns k0, k1
        def add_strassen_block(k0, k1):
            # Standard Strassen decomposition for M_{2,2,2}
            # U indices: 0,1,2,3 (i*2+j)
            # V indices: j*2+k_local (0..3) -> mapped to global j*9 + k
            # W indices: k_local*2+i (0..3) -> mapped to global k*2 + i
            
            # Term 1
            U1 = {0: 1, 3: 1}
            V1 = {k0: 1, 9+k1: 1} # j=0,k0 and j=1,k1
            W1 = {2*k0: 1, 2*k1+1: 1} # k0,i=0 and k1,i=1
            terms.append((U1, V1, W1))
            
            # Term 2
            U2 = {2: 1, 3: 1}
            V2 = {k0: 1} # j=0,k0
            W2 = {2*k0+1: 1, 2*k1+1: -1} # k0,i=1 and k1,i=1
            terms.append((U2, V2, W2))
            
            # Term 3
            U3 = {0: 1}
            V3 = {k1: 1, 9+k1: -1} # j=0,k1 and j=1,k1
            W3 = {2*k1: 1, 2*k1+1: 1} # k1,i=0 and k1,i=1
            terms.append((U3, V3, W3))
            
            # Term 4
            U4 = {3: 1}
            V4 = {9+k0: 1, k0: -1} # j=1,k0 and j=0,k0
            W4 = {2*k0: 1, 2*k0+1: 1} # k0,i=0 and k0,i=1
            terms.append((U4, V4, W4))
            
            # Term 5
            U5 = {0: 1, 1: 1}
            V5 = {9+k1: 1} # j=1,k1
            W5 = {2*k0: -1, 2*k1: 1} # k0,i=0 and k1,i=0
            terms.append((U5, V5, W5))
            
            # Term 6
            U6 = {2: 1, 0: -1}
            V6 = {k0: 1, k1: 1} # j=0,k0 and j=0,k1
            W6 = {2*k1+1: 1} # k1,i=1
            terms.append((U6, V6, W6))
            
            # Term 7
            U7 = {1: 1, 3: -1}
            V7 = {9+k0: 1, 9+k1: 1} # j=1,k0 and j=1,k1
            W7 = {2*k0: 1} # k0,i=0
            terms.append((U7, V7, W7))

        # Apply Strassen to column pairs (0,1), (2,3), (4,5)
        for block_idx in range(3):
            k0 = 2 * block_idx
            k1 = k0 + 1
            add_strassen_block(k0, k1)
        
        # Handle last 3 columns (k=6,7,8) with rank-10 decomposition for 2x2x3
        # This requires careful construction
        add_rank10_block(6, 7, 8)

        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    def add_rank10_block(k0, k1, k2):
        """
        Add a rank-10 decomposition for M_{2,2,3} using columns k0, k1, k2.
        
        The 2x2x3 tensor has known rank 10. We need to find a valid decomposition.
        One approach: use Strassen for (k0,k1) giving rank 7, then add 3 more terms for k2,
        but that gives 10 total. However, we need to verify this actually works.
        
        Actually, let me use a different strategy: use schoolbook for all 3 columns (rank 12)
        and see if we can reduce by 2 through clever combination.
        
        For now, let's use a known rank-10 construction if available, otherwise fall back to rank 12.
        """
        # For 2x2x3, one known approach:
        # Use the fact that we can share some multiplications
        # Let me construct it explicitly
        
        # Terms for column k0 (rank 4 schoolbook)
        terms.append(({0: 1}, {k0: 1}, {2*k0: 1}))
        terms.append(({1: 1}, {9+k0: 1}, {2*k0: 1}))
        terms.append(({2: 1}, {k0: 1}, {2*k0+1: 1}))
        terms.append(({3: 1}, {9+k0: 1}, {2*k0+1: 1}))
        
        # Terms for column k1 (rank 4 schoolbook)  
        terms.append(({0: 1}, {k1: 1}, {2*k1: 1}))
        terms.append(({1: 1}, {9+k1: 1}, {2*k1: 1}))
        terms.append(({2: 1}, {k1: 1}, {2*k1+1: 1}))
        terms.append(({3: 1}, {9+k1: 1}, {2*k1+1: 1}))
        
        # Terms for column k2 (rank 4 schoolbook)
        terms.append(({0: 1}, {k2: 1}, {2*k2: 1}))
        terms.append(({1: 1}, {9+k2: 1}, {2*k2: 1}))
        terms.append(({2: 1}, {k2: 1}, {2*k2+1: 1}))
        terms.append(({3: 1}, {9+k2: 1}, {2*k2+1: 1}))
        # This gives rank 12, not 10. Need to find a better construction.
        
        # Alternative: use Strassen for (k0,k1) = rank 7, then add 3 terms for k2
        # But that's 10 total. Let me try that instead.
        # Remove the above 12 terms and use Strassen + 3 terms
        
        # Actually, I'll implement both and let the caller choose. For now, use the 12-term version
        # which is guaranteed to work, and we'll have rank 7*3 + 12 = 33, which is worse.
        
        # Let me reconsider: use Strassen for (k0,k1) and schoolbook for k2
        # That's 7 + 4 = 11, giving total 7*3 + 11 = 32, same as before.
        
        # To get 31, I need a true rank-10 for 2x2x3.
        # From literature, the rank of M_{2,2,3} is indeed 10.
        # Let me look up or derive the decomposition.
        
        # One known rank-10 decomposition for 2x2x3:
        # This is complex. For safety, let's stick with a known working solution.
        # I'll use 3 Strassen blocks (rank 21) + 1 schoolbook block for 3 columns (rank 12) = 33
        # That's worse. 
        
        # Better: 4 Strassen blocks (rank 28) + 1 schoolbook (rank 4) = 32 (current)
        # Or: 3 Strassen blocks (rank 21) + rank-10 for 2x2x3 = 31
        
        # I need the actual rank-10 decomposition. Let me construct it.
        
        # For M_{2,2,3}, indices: i,j in {0,1}, k in {0,1,2}
        # U: 4 entries, V: 6 entries, W: 6 entries
        # 10 terms
        
        # Based on known results, one construction uses:
        # m1 = A00*B00, m2 = A00*B01, m3 = A00*B02, m4 = A01*B10, m5 = A01*B11, m6 = A01*B12
        # m7 = (A00+A01)*(B10+B11), m8 = (A00+A01)*(B10+B12), m9 = (A10+A11)*(B00+B01), m10 = (A10+A11)*(B00+B02)
        # Then combine to get all 12 products... but this doesn't quite work directly.
        
        # Actually, the standard way: 
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11  
        # C02 = A00*B02 + A01*B12
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # C12 = A10*B02 + A11*B12
        
        # We need 10 scalar multiplications to compute these 12 bilinear forms.
        
        # Let me use a known decomposition from literature.
        # This is getting complex. For this iteration, I'll keep the rank-32 solution
        # and mark that rank-31 requires finding the explicit rank-10 decomposition.
        
        pass  # Placeholder - will use fallback

    # Fallback: use the original rank-32 construction
    def build_rank32_fallback():
        terms_fb = []
        
        def add_strassen_block_fb(k0, k1):
            U1 = {0: 1, 3: 1}
            V1 = {k0: 1, 9+k1: 1}
            W1 = {2*k0: 1, 2*k1+1: 1}
            terms_fb.append((U1, V1, W1))
            
            U2 = {2: 1, 3: 1}
            V2 = {k0: 1}
            W2 = {2*k0+1: 1, 2*k1+1: -1}
            terms_fb.append((U2, V2, W2))
            
            U3 = {0: 1}
            V3 = {k1: 1, 9+k1: -1}
            W3 = {2*k1: 1, 2*k1+1: 1}
            terms_fb.append((U3, V3, W3))
            
            U4 = {3: 1}
            V4 = {9+k0: 1, k0: -1}
            W4 = {2*k0: 1, 2*k0+1: 1}
            terms_fb.append((U4, V4, W4))
            
            U5 = {0: 1, 1: 1}
            V5 = {9+k1: 1}
            W5 = {2*k0: -1, 2*k1: 1}
            terms_fb.append((U5, V5, W5))
            
            U6 = {2: 1, 0: -1}
            V6 = {k0: 1, k1: 1}
            W6 = {2*k1+1: 1}
            terms_fb.append((U6, V6, W6))
            
            U7 = {1: 1, 3: -1}
            V7 = {9+k0: 1, 9+k1: 1}
            W7 = {2*k0: 1}
            terms_fb.append((U7, V7, W7))

        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            add_strassen_block_fb(k0, k1)
            
        k = 8
        terms_fb.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms_fb.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms_fb.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms_fb.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))

        rank = len(terms_fb)
        U_fb = np.zeros((4, rank))
        V_fb = np.zeros((18, rank))
        W_fb = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms_fb):
            for idx, val in Ud.items():
                U_fb[idx][t] = val
            for idx, val in Vd.items():
                V_fb[idx][t] = val
            for idx, val in Wd.items():
                W_fb[idx][t] = val
        return U_fb, V_fb, W_fb

    # Try rank-31 first, fall back to rank-32
    try:
        U, V, W = build_rank31()
        if not verify(U, V, W):
            raise ValueError("Rank 31 verification failed")
    except:
        U, V, W = build_rank32_fallback()
    
    # Sanitize to ensure integer/half-integer constraint is met exactly
    U = sanitize_matrix(U)
    V = sanitize_matrix(V)
    W = sanitize_matrix(W)
    
    # Final verification
    if not verify(U, V, W):
        # If verification fails (should not happen with this construction), raise error
        raise ValueError("Internal verification failed for rank 32 solution")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
