"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_structured():
        """
        Search for rank-13 by merging Strassen terms with same U-vector.
        """
        if time.time() > deadline - 10:
            return None

        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]), # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 1, 1]), # M7
        ]
        
        for li in range(7):
            if time.time() > deadline - 5:
                return None
            for ri in range(7):
                if strassen_terms[li][0] != strassen_terms[ri][0]:
                    continue
                
                R = 13
                U_mat = np.zeros((4, R))
                V_mat = np.zeros((8, R))
                W_mat = np.zeros((8, R))
                
                U_mat[:, 0] = strassen_terms[li][0]
                V_mat[0:4, 0] = strassen_terms[li][1]
                V_mat[4:8, 0] = strassen_terms[ri][1]
                W_mat[0:4, 0] = strassen_terms[li][2]
                W_mat[4:8, 0] = strassen_terms[ri][2]
                
                col = 1
                for l_idx in range(7):
                    if l_idx == li: continue
                    U_mat[:, col] = strassen_terms[l_idx][0]
                    V_mat[0:4, col] = strassen_terms[l_idx][1]
                    W_mat[0:4, col] = strassen_terms[l_idx][2]
                    col += 1
                for r_idx in range(7):
                    if r_idx == ri: continue
                    U_mat[:, col] = strassen_terms[r_idx][0]
                    V_mat[4:8, col] = strassen_terms[r_idx][1]
                    W_mat[4:8, col] = strassen_terms[r_idx][2]
                    col += 1
                
                if verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist()):
                    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
        return None
    
    def try_rank13_randomized():
        """
        Randomized bilinear reduction search: sample random linear combinations
        of schoolbook terms and attempt to find a rank-13 decomposition.
        """
        if time.time() > deadline - 15:
            return None
        
        # The 16 schoolbook terms: each is (u_vec, v_vec, w_vec) where
        # u_vec has 4 entries (a00,a01,a10,a11), v_vec has 8 entries (b00..b13), w_vec has 8 entries
        # Term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        schoolbook = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0]*4
                    v = [0]*8
                    w = [0]*8
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    schoolbook.append((u, v, w))
        
        # Try random subsets of size 13 from the 16 terms, then try to find
        # a rank-13 decomposition by expressing 3 schoolbook terms as combinations
        # of the other 13.
        
        # Alternative approach: randomly perturb the Strassen rank-14 solution
        # and try to eliminate one term.
        
        # Build the rank-14 Strassen solution
        R14 = 14
        U14 = np.zeros((4, R14))
        V14 = np.zeros((8, R14))
        W14 = np.zeros((8, R14))
        
        # Left block
        U14[0,0]=1; U14[3,0]=1; V14[0,0]=1; V14[5,0]=1; W14[0,0]=1; W14[3,0]=1
        U14[2,1]=1; U14[3,1]=1; V14[0,1]=1; W14[1,1]=1; W14[3,1]=-1
        U14[0,2]=1; V14[1,2]=1; V14[5,2]=-1; W14[2,2]=1; W14[3,2]=1
        U14[3,3]=1; V14[4,3]=1; V14[0,3]=-1; W14[0,3]=1; W14[1,3]=1
        U14[0,4]=1; U14[1,4]=1; V14[5,4]=1; W14[0,4]=-1; W14[2,4]=1
        U14[2,5]=1; U14[0,5]=-1; V14[0,5]=1; V14[1,5]=1; W14[3,5]=1
        U14[1,6]=1; U14[3,6]=-1; V14[4,6]=1; V14[5,6]=1; W14[0,6]=1
        
        # Right block
        U14[0,7]=1; U14[3,7]=1; V14[2,7]=1; V14[7,7]=1; W14[4,7]=1; W14[7,7]=1
        U14[2,8]=1; U14[3,8]=1; V14[2,8]=1; W14[5,8]=1; W14[7,8]=-1
        U14[0,9]=1; V14[3,9]=1; V14[7,9]=-1; W14[6,9]=1; W14[7,9]=1
        U14[3,10]=1; V14[6,10]=1; V14[2,10]=-1; W14[4,10]=1; W14[5,10]=1
        U14[0,11]=1; U14[1,11]=1; V14[7,11]=1; W14[4,11]=-1; W14[6,11]=1
        U14[2,12]=1; U14[0,12]=-1; V14[2,12]=1; V14[3,12]=1; W14[7,12]=1
        U14[1,13]=1; U14[3,13]=-1; V14[6,13]=1; V14[7,13]=1; W14[4,13]=1
        
        # For each pair of terms (one from left, one from right), try to merge them
        # by finding a common U vector and adjusting V, W accordingly.
        # Also try random linear combinations.
        
        # Strategy: try to find coefficients alpha, beta such that
        # alpha * U_left[t1] + beta * U_right[t2] = some simple vector,
        # and the combined V, W still give correct products.
        
        # More systematically: for each pair (t1, t2) from left/right blocks,
        # try to find a rank-13 by expressing one as a combination.
        
        # Let's try: can term t1 (left) be written as a linear combination
        # of the other 13 terms? If so, we can eliminate it.
        
        # For a rank-13 to exist, one of the 14 terms must be in the span
        # of the others (in the tensor sense).
        
        # Check if any single term is redundant given the other 13.
        # This requires solving a system. Let's try a different approach:
        # randomly sample 13 of the 14 terms and check if they still decompose
        # the full tensor (they won't, but we can try to adjust).
        
        # Better: use the fact that we need to find 13 rank-1 tensors whose
        # sum equals the multiplication tensor. Start from the 14-term solution
        # and try to perturb it.
        
        # Try: for each term t in 0..13, try to express it as a combination
        # of the others by solving a linear system.
        
        # The tensor equation is:
        # sum_r U[:,r] * V[:,r] * W[:,r] = T (the multiplication tensor)
        # where T[i,j,k] = 1 if (i,j,k) is a valid schoolbook term.
        
        # If we have 14 terms and want 13, we need:
        # U_t * V_t * W_t = sum_{s != t} c_s * U_s * V_s * W_s
        # This is a system of equations in c_s.
        
        # Let's try a more direct approach: randomly perturb and verify.
        
        random.seed(42)
        
        # Try random small perturbations of the rank-14 solution
        # to see if we can collapse one dimension.
        
        # Actually, let's try a cleaner approach: 
        # For each pair (li, ri) where U vectors match, we already handled that.
        # Now try: can we find a linear combination of two left terms that
        # equals one right term's U vector?
        
        # Let's try all pairs of left terms and see if their U vectors
        # can be combined to match a right term's U vector.
        
        strassen_U_left = [
            [1, 0, 0, 1],  # M1
            [0, 0, 1, 1],  # M2
            [1, 0, 0, 0],  # M3
            [0, 0, 0, 1],  # M4
            [1, 1, 0, 0],  # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1], # M7
        ]
        strassen_U_right = strassen_U_left  # Same A matrix
        
        # Try all pairs and see if a linear combination gives a simple vector
        # that we can use to merge.
        
        for attempt in range(2000):
            if time.time() > deadline - 5:
                return None
            
            # Randomly pick which term to try to eliminate
            t_elim = random.randint(0, 13)
            
            # Build a 13-term candidate by zeroing out term t_elim
            # and trying to adjust the others.
            
            # Actually, let's try a different randomized approach:
            # Sample random vectors u, v, w and check if adding them
            # to a subset of the 14 terms gives a valid rank-13.
            
            # Simpler: try all 14 choose 13 = 14 subsets and check
            # if any subset of 13 terms (possibly modified) works.
            
            # For each subset of 13 terms, we need to find coefficients
            # that make the tensor equation hold.
            
            # Let's try: for each term t to eliminate, solve for
            # modifications to the remaining 13 terms.
            
            # The constraint is that for all (i,j,k):
            # sum_{r in remaining} U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
            
            # This is a system of 32 equations (2*2*4) in the 13*3=39 unknowns
            # (after fixing which 13 terms to keep).
            
            # Let's try to solve this using numpy.
            
            # Build the system for eliminating term t_elim.
            # We keep terms 0..13 except t_elim.
            keep = [r for r in range(14) if r != t_elim]
            
            # Unknowns: for each kept term r, we have U[:,r], V[:,r], W[:,r]
            # But that's 4*13 + 8*13 + 8*13 = 52 + 104 + 104 = 260 unknowns.
            # That's too many. Let's fix U and V and solve for W, or similar.
            
            # Alternative: fix U and W, solve for V. Or fix U and V, solve for W.
            
            # Let's fix U and V from the original solution (for kept terms)
            # and solve for W.
            
            # For each output (i,j,k), the equation is:
            # sum_{r in keep} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
            
            # This is linear in W if U and V are fixed.
            # But W[k*n+i, r] appears in multiple equations.
            
            # Let's set up the linear system.
            # Variables: W[0,0], W[0,1], ..., W[7,12] (8*13 = 104 variables)
            # Equations: 32 (for each i,j,k)
            
            # Actually, let's just try the structured merge approach more broadly.
            
            # Try: for each pair (li, ri) with different U vectors, can we
            # find a linear combination?
            
            # Let's try a completely different randomized approach:
            # Sample random rank-1 tensors and try to find 13 that sum to T.
            
            # This is equivalent to finding 13 vectors u_r, v_r, w_r such that
            # sum_r u_r[i] * v_r[j] * w_r[k] = T[i,j,k] for all i,j,k.
            
            # Let's parameterize: u_r in R^4, v_r in R^8, w_r in R^8.
            # 13 * (4+8+8) = 260 unknowns, 32 equations.
            # This is underdetermined, so solutions exist if rank <= 13.
            
            # Let's try to find a solution by random initialization + least squares.
            
            # Initialize randomly
            U_rand = np.random.randn(4, 13)
            V_rand = np.random.randn(8, 13)
            W_rand = np.random.randn(8, 13)
            
            # Check if this works
            # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should be 1
            # if (i,j,k) is a schoolbook term, else 0.
            
            # Build the tensor T
            T = np.zeros((2, 2, 4))
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        T[i,j,k] = 1.0
            
            # Compute the candidate tensor
            # C[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C = np.zeros((2, 2, 4))
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        s = 0.0
                        for r in range(13):
                            s += U_rand[i*2+j, r] * V_rand[j*4+k, r] * W_rand[k*2+i, r]
                        C[i,j,k] = s
            
            err = np.sum((C - T)**2)
            if err < 1e-10:
                return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
        
        return None
    
    # Try structured rank-13 first
    result = try_rank13_structured()
    if result is not None:
        return result
    
    # Try randomized rank-13
    result = try_rank13_randomized()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
