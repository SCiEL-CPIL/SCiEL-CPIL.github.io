"""Program for rank-minimal tensor decomposition of 3x3 by 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) where u, v, w are 4-dimensional vectors.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        C = A*B: i=a*e+b*g, j=a*f+b*h, k=c*e+d*g, l=c*f+d*h
        
        Strassen:
        m1 = (a+d)(e+h)
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-c)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        
        i = m1 + m4 - m5 + m7
        j = m3 + m5
        k = m2 + m4
        l = m1 - m2 + m3 + m6
        """
        terms = [
            (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])),
            (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,-1])),
            (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,1])),
            (np.array([0,0,0,1]), np.array([0,0,1,-1]), np.array([1,0,1,0])),
            (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])),
            (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])),
            (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0])),
        ]
        return terms

    def build_3x3x3_rank28():
        """
        Build a rank-28 decomposition for 3x3x3 using 4 overlapping 2x2 Strassen blocks.
        Blocks: TL(0,1,0,1), TR(0,1,1,2), BL(1,2,0,1), BR(1,2,1,2)
        """
        rank = 28
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        strassen_terms = strassen_2x2x2_terms()
        
        blocks = [
            (0, 1, 0, 1),  # TL
            (0, 1, 1, 2),  # TR
            (1, 2, 0, 1),  # BL
            (1, 2, 1, 2),  # BR
        ]
        
        col = 0
        for r1, r2, c1, c2 in blocks:
            for s in range(7):
                u2, v2, w2 = strassen_terms[s]
                
                # A mapping: 2x2 idx = i*2+j -> 3x3 pos (r1+i, c1+j)
                for i in range(2):
                    for j in range(2):
                        idx2 = i * 2 + j
                        a3_row = r1 + i
                        a3_col = c1 + j
                        U[a3_row * 3 + a3_col, col] += u2[idx2]
                
                # B mapping: 2x2 idx = j*2+k -> 3x3 pos (c1+j, c1+k)
                for j in range(2):
                    for k in range(2):
                        idx2 = j * 2 + k
                        b3_row = c1 + j
                        b3_col = c1 + k
                        V[b3_row * 3 + b3_col, col] += v2[idx2]
                
                # W mapping: 2x2 idx = i*2+k -> 3x3 pos (r1+i, c1+k)
                # W index convention: W[k*n+i, s]
                for i in range(2):
                    for k in range(2):
                        idx2 = i * 2 + k
                        c3_row = r1 + i
                        c3_col = c1 + k
                        W[c3_col * 3 + c3_row, col] += w2[idx2]
                
                col += 1
        
        return U, V, W, rank

    def verify_3x3x3(U, V, W):
        """Verify 3x3x3 decomposition."""
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = np.dot(U[i*3+j, :], V[j*3+k, :] * W[k*3+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_3x3x6_from_3x3x3(U33, V33, W33, rank33):
        """
        Lift a rank-r 3x3x3 decomposition to 3x3x6 with rank 2r.
        B is 3x6: split into two 3x3 halves.
        """
        rank = 2 * rank33
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        for s in range(rank33):
            # First half: B cols 0-2, C cols 0-2
            idx1 = 2 * s
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx1] = U33[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, idx1] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[k*3+i, idx1] = W33[k*3+i, s]
            
            # Second half: B cols 3-5, C cols 3-5
            idx2 = 2 * s + 1
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx2] = U33[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k+3, idx2] = V33[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i, idx2] = W33[k*3+i, s]
        
        return U, V, W, rank

    def verify_3x3x6(U, V, W):
        """Verify 3x3x6 decomposition."""
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    total = np.dot(U[i*3+j, :], V[j*6+k, :] * W[k*3+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def schoolbook_3x3x6():
        """Return schoolbook decomposition (rank 54) for 3x3x6."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def build_rank45_3x3x6():
        """
        Attempt to build a rank-45 decomposition for 3x3x6.
        Uses a structured approach: 3x3x6 = 3x3x3 + 3x3x3 with shared A.
        We use rank-22 3x3x3 blocks (if they exist) or fall back to rank-23.
        
        Strategy: Use 2 blocks of rank-23 (total 46) but try to merge
        1 rank unit by exploiting the shared A structure.
        
        Alternative: Direct construction using the fact that for 3x3x6,
        we can use a "block diagonal" approach where the A factor is shared.
        
        Let's try: rank = 3*15 = 45 by using 5 rank-9 sub-blocks?
        Or: Use the structure that C = A*B where B is 3x6.
        Write B = [B1, B2] where B1, B2 are 3x3.
        C = [A*B1, A*B2]
        
        If we have a rank-r decomposition of 3x3x3, we get rank 2r for 3x3x6.
        But we can do better by sharing the U factors.
        
        Key insight: In the decomposition T = sum_s u_s ⊗ v_s ⊗ w_s,
        for 3x3x6, the U factor has shape (9, rank). If we use the same
        U vectors for both halves, we can save rank.
        
        Specifically, if we write:
        T_3x3x6 = sum_{s=1}^{r} u_s ⊗ (v_s^1 ⊕ v_s^2) ⊗ (w_s^1 ⊕ w_s^2)
        where v_s^1, w_s^1 handle B1->C1 and v_s^2, w_s^2 handle B2->C2,
        and u_s is shared, then rank is r (not 2r).
        
        But this only works if the same u_s can serve both halves, which
        requires a specific structure.
        
        Let's try a different approach: Use rank-22 for 3x3x3 (known to exist
        from literature, e.g., using field extensions or specific constructions).
        Then 3x3x6 has rank at most 2*22 = 44, or possibly less with sharing.
        
        For now, let's construct a rank-45 solution by using a hybrid approach.
        """
        # Use rank-23 3x3x3 blocks (conservative)
        # Actually, let's just use the rank-28 3x3x3 we have and see if we can
        # do better with a direct 3x3x6 construction.
        
        # For now, return a valid rank-54 schoolbook as fallback
        # and let the main function try to improve.
        return None

    best_result = None
    best_rank = float('inf')

    # Method 1: 4 Strassen blocks for 3x3x3 (rank 28), lift to 3x3x6 (rank 56)
    try:
        U33, V33, W33, rank33 = build_3x3x3_rank28()
        err33 = verify_3x3x3(U33, V33, W33)
        if err33 < 1e-9:
            U, V, W, rank = build_3x3x6_from_3x3x3(U33, V33, W33, rank33)
            err = verify_3x3x6(U, V, W)
            if err < 1e-9 and rank < best_rank:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # Method 2: Schoolbook fallback (rank 54)
    U, V, W, rank = schoolbook_3x3x6()
    err = verify_3x3x6(U, V, W)
    if err < 1e-9 and rank < best_rank:
        best_rank = rank
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if best_result is not None:
        return best_result

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
