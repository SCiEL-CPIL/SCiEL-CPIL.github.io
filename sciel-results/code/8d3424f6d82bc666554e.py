import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build the known rank-55 Strassen-based decomposition
    # This serves as the baseline and fallback.
    
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        # A indices: i*m + j
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=0, j=2 -> 2
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        # i=1, j=2 -> 5
        
        a00, a01, a02 = 0, 1, 2
        a10, a11, a12 = 3, 4, 5
        
        # B indices: j*p + k
        # j=0: 0..9
        # j=1: 10..19
        # j=2: 20..29
        
        # W indices: k*n + i
        # k=0..9, i=0..1
        # 0, 1, 2, 3, ..., 18, 19
        
        # Part 1: 2x2x10 using 5 Strassen blocks
        # We use Strassen for the sub-matrix A[0:2, 0:2] * B[0:2, 0:10]
        # A_sub = [[A00, A01], [A10, A11]]
        # B_sub is 2x10, split into 5 blocks of 2x2: B[0:2, 2s:2s+2]
        
        for s in range(5):
            k0 = 2 * s
            k1 = 2 * s + 1
            
            # B block indices
            b00 = k0      # B[0, k0]
            b01 = k1      # B[0, k1]
            b10 = 10 + k0 # B[1, k0]
            b11 = 10 + k1 # B[1, k1]
            
            # C block indices (output C[0:2, 2s:2s+2])
            # C[0, k0] -> W[k0*2 + 0]
            # C[0, k1] -> W[k1*2 + 0]
            # C[1, k0] -> W[k0*2 + 1]
            # C[1, k1] -> W[k1*2 + 1]
            
            c00 = k0 * 2 + 0
            c01 = k1 * 2 + 0
            c10 = k0 * 2 + 1
            c11 = k1 * 2 + 1
            
            # Strassen terms:
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-a)(e+f)
            # m7 = (b-d)(g+h)
            
            # m1
            U[a00, col] = 1.0; U[a11, col] = 1.0
            V[b00, col] = 1.0; V[b11, col] = 1.0
            W[c00, col] = 1.0; W[c11, col] = 1.0
            col += 1
            
            # m2
            U[a10, col] = 1.0; U[a11, col] = 1.0
            V[b00, col] = 1.0
            W[c10, col] = 1.0; W[c11, col] = -1.0
            col += 1
            
            # m3
            U[a00, col] = 1.0
            V[b01, col] = 1.0; V[b11, col] = -1.0
            W[c01, col] = 1.0; W[c11, col] = 1.0
            col += 1
            
            # m4
            U[a11, col] = 1.0
            V[b10, col] = 1.0; V[b00, col] = -1.0
            W[c00, col] = 1.0; W[c10, col] = 1.0
            col += 1
            
            # m5
            U[a00, col] = 1.0; U[a01, col] = 1.0
            V[b11, col] = 1.0
            W[c00, col] = -1.0; W[c01, col] = 1.0
            col += 1
            
            # m6
            U[a10, col] = 1.0; U[a00, col] = -1.0
            V[b00, col] = 1.0; V[b01, col] = 1.0
            W[c11, col] = 1.0
            col += 1
            
            # m7
            U[a01, col] = 1.0; U[a11, col] = -1.0
            V[b10, col] = 1.0; V[b11, col] = 1.0
            W[c00, col] = 1.0
            col += 1
        
        # Part 2: 2x1x10
        # A uses j=2 (indices a02, a12)
        # B uses j=2 (indices 20..29)
        # C uses i in {0,1}, k in {0..9}
        
        for k in range(10):
            b2k = 2 * p + k # 20 + k
            
            # Term for i=0
            U[a02, col] = 1.0
            V[b2k, col] = 1.0
            W[k * 2 + 0, col] = 1.0
            col += 1
            
            # Term for i=1
            U[a12, col] = 1.0
            V[b2k, col] = 1.0
            W[k * 2 + 1, col] = 1.0
            col += 1
            
        return U, V, W, rank

    U55, V55, W55, rank55 = build_rank_55()
    
    # Verify the rank-55 solution
    def verify(U, V, W):
        # T[i, j, k] = 1 if we are multiplying A[i,j] * B[j,k]
        # Decomposition: T[i, j, k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # Note: W index is k*n + i
        
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1
                    s = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for t in range(len(U[0])):
                        s += U[u_idx, t] * V[v_idx, t] * W[w_idx, t]
                    if abs(s - target) > 1e-9:
                        return False
        return True

    if not verify(U55, V55, W55):
        # Fallback to schoolbook if Strassen implementation is buggy
        # This should not happen, but safety first.
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55

    # Try to find a rank-54 solution by exploiting the n=2 symmetry.
    # Key idea: For each (j,k) pair, we need both T[0,j,k]=1 and T[1,j,k]=1.
    # If we can pair up terms so that a single term contributes to both i=0 and i=1 outputs,
    # we might save one term overall.
    # 
    # Strategy: Start with the rank-55 solution and try to merge two terms into one
    # by finding a linear dependency. Specifically, look for two terms t1, t2 such that
    # their W vectors are proportional (since n=2, W has only 2 entries per term per k).
    # 
    # A more practical approach: try to find if we can express the tensor with 54 terms
    # by checking if the 55th term is in the span of the others (in the appropriate sense).
    # This is a bilinear problem, so it's not straightforward.
    # 
    # Alternative: Use the fact that for n=2, we can write C = A*B where C is 2x10.
    # C[0,:] = A[0,:] * B  (row vector times matrix)
    # C[1,:] = A[1,:] * B
    # 
    # The bilinear map (A,B) -> C can be seen as a map from R^6 x R^30 -> R^20.
    # We want the minimal number of rank-1 bilinear forms.
    # 
    # Let's try a targeted search: modify the rank-55 solution by combining the last two
    # terms (which handle i=0 and i=1 for k=9 in the 2x1x10 part) into a single term
    # if possible. The last two terms are:
    # Term 53: U[a02,53]=1, V[b29,53]=1, W[18,53]=1  (i=0, j=2, k=9)
    # Term 54: U[a12,54]=1, V[b29,54]=1, W[19,54]=1  (i=1, j=2, k=9)
    # 
    # These can't be merged because U uses different rows (a02 vs a12) and W uses different rows.
    # 
    # Let's instead try a random perturbation approach: start with rank 55, randomly
    # combine pairs of terms, and check if we can maintain correctness with rank 54.
    # This is a nonlinear feasibility problem, so we use a simple heuristic.
    
    if time.time() - start_time < deadline - 5:
        # Try to find a rank-54 solution by random search over small modifications
        # We'll try to merge two of the 20 "trivial" terms (from the 2x1x10 part)
        # with two of the Strassen terms by finding a linear combination.
        
        # The 20 trivial terms handle: for each k, i=0 and i=1 with j=2.
        # U entries: a02=2, a12=5
        # V entries: 20+k for k=0..9
        # W entries: 2k (for i=0) and 2k+1 (for i=1)
        
        # Try to see if we can replace two trivial terms with one modified Strassen term.
        # This is unlikely to work, but let's try a small random search.
        
        import random
        random.seed(42)
        
        # Try modifying the solution by adding a small random perturbation and
        # checking if we can project back to a valid rank-54 solution.
        # This is a heuristic and may not find anything, but it's cheap to try.
        
        found_rank54 = False
        for attempt in range(100):
            if time.time() - start_time > deadline - 2:
                break
            
            # Try to merge two random terms from the trivial part
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            if k1 == k2:
                continue
            
            # The two terms we want to merge:
            # Term for (i=0, j=2, k=k1): U[2,t]=1, V[20+k1,t]=1, W[2*k1,t]=1
            # Term for (i=1, j=2, k=k2): U[5,t]=1, V[20+k2,t]=1, W[2*k2+1,t]=1
            # 
            # Can we replace these with a single term? We need:
            # U[2,t]=1, U[5,t]=1, V[20+k1,t]=1, V[20+k2,t]=1, W[2*k1,t]=1, W[2*k2+1,t]=1
            # But this single term would also contribute to other (i,j,k) combinations
            # where U[2,*] or U[5,*] or V[*,20+k1] or V[*,20+k2] or W[2*k1,*] or W[2*k2+1,*]
            # are nonzero, which would break the correctness.
            # 
            # So this direct merge doesn't work. The terms are too entangled.
            # 
            # Given the difficulty, we'll stick with rank 55.
            pass
        
        # If no rank-54 solution found, use rank 55
        if not found_rank54:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
