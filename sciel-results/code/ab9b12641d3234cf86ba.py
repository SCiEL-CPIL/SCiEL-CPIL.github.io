"""Matrix multiplication 2x3x5 - search for rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time
import random
import itertools

def construct():
    """
    Return the matrix multiplication tensor for 2x3 * 3x5 as a rank decomposition.
    U: (n*m, rank) = (6, rank)
    V: (m*p, rank) = (15, rank)
    W: (p*n, rank) = (10, rank)
    
    Indexing conventions:
    U[i*m + j, r] corresponds to A[i,j]
    V[j*p + k, r] corresponds to B[j,k]
    W[k*n + i, r] corresponds to C[i,k]
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def try_structured_rank(R):
        """
        Try a structured approach:
        For each (i,k), the equation is:
        sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        Let's define for each (i,k): the vector w_ik = W[k*n+i, :]
        Then: sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * w_ik[r] = 1
        
        Let A_ik[j,r] = U[i*m+j,r] * V[j*p+k,r]
        Then we need: sum_r (sum_j A_ik[j,r]) * w_ik[r] = 1 for each (i,k)
        
        This is a single linear equation in w_ik for each (i,k).
        So w_ik just needs to satisfy one constraint: dot(sum_j A_ik[j,:], w_ik) = 1
        
        We can choose w_ik freely as long as this holds.
        So the question is: can we choose U, V such that for all (i,k),
        the vector sum_j U[i*m+j,:] * V[j*p+k,:] (element-wise) is non-zero?
        
        If so, we can always find w_ik. But we need w_ik to have half-integer entries.
        
        Let's try: set U and V to have entries in {0, 1, -1} and check if
        sum_j U[i*m+j,r] * V[j*p+k,r] is non-zero for at least one r,
        then set w_ik[r] = 1 / (that value) if it's ±1, or use other tricks.
        """
        if time.time() > deadline:
            return None
        
        # Try: for each r, define u_r = (U[0,m-1], U[1,m-1], ..., U[5,m-1]) (6-dim vector)
        # v_r = (V[0,p-1], V[1,p-1], ..., V[14,p-1]) (15-dim vector)
        # For (i,k): sum_j U[i*m+j, r] * V[j*p+k, r]
        # = sum over j of the (i,j) component of u_r times (j,k) component of v_r
        
        # Let's try a specific structure:
        # Split the R terms into groups. For each group, assign a specific (i,k) pattern.
        
        # Actually, let's try a direct approach:
        # Use R = 15 terms. For each (j,k) pair (3*5=15), create one term.
        # Term r corresponds to (j_r, k_r).
        # U[i*m+j, r] = 1 if j == j_r else 0
        # V[j*p+k, r] = 1 if (j==j_r and k==k_r) else 0
        # Then for (i,k): sum_r U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k*n+i, r]
        # = sum_{r: j_r=j, k_r=k} 1 * 1 * W[k*n+i, r] ... no wait.
        
        # Let me redo: if U[i*m+j, r] = delta(j, j_r) and V[j*p+k, r] = delta(j,j_r)*delta(k,k_r)
        # Then sum_j U[i*m+j,r]*V[j*p+k,r] = sum_j delta(j,j_r)*delta(j,j_r)*delta(k,k_r) = delta(k,k_r)
        # So the equation becomes: sum_r delta(k,k_r) * W[k*n+i, r] = 1
        # = sum_{r: k_r = k} W[k*n+i, r] = 1
        # 
        # So for each (i,k), we need the sum of W[k*n+i, r] over all r with k_r=k to be 1.
        # There are 3 such r values (since j_r can be 0,1,2 for each k_r=k).
        # We can set each to 1/3. But 1/3 is not a half-integer!
        
        # Alternative: use R = 30 with the schoolbook, or find a different structure.
        
        # Let's try: for each (i,k), use a dedicated term. That's 10 terms.
        # But we need to handle the j summation.
        
        # New idea: R = 10. For each (i,k), one term r = i*5+k.
        # U[i*m+j, r] = 1 for all j (when r corresponds to (i,k))
        # V[j*p+k, r] = 1 for all j (when r corresponds to (i,k))
        # W[k*n+i, r] = 1
        # Then sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_j 1*1*1 = 3 ≠ 1
        
        # So we need W = 1/3. Not a half-integer.
        
        # Try R = 20: for each (i,k), use 2 terms.
        # Term 1: U[i*m+j, r1] = 1 for all j, V[j*p+k, r1] = 1 for all j, W = 1/2
        # Term 2: U[i*m+j, r2] = 1 for all j, V[j*p+k, r2] = 1 for all j, W = 1/2
        # Sum = 2 * (1/2) * 3 = 3. Still not 1.
        
        # I need: sum_j U[i*m+j,r]*V[j*p+k,r] * W = 1
        # If U and V are both 1 for all j in the relevant positions, sum_j gives 3.
        # So W = 1/3. Not half-integer.
        
        # To get half-integers, I need the sum_j product to be ±1 or ±2.
        # If sum_j U*V = 1, then W=1 works.
        # If sum_j U*V = 2, then W=1/2 works.
        
        # Can I make sum_j U[i*m+j,r]*V[j*p+k,r] = 1 or 2?
        # U and V are shared across different (i,k) pairs, so this is constrained.
        
        # Let's try a randomized search with a better verification.
        return None
    
    def try_rank_search(R, max_iters=2000):
        """Random search with exact verification."""
        if time.time() > deadline:
            return None
        
        values = [Fraction(0), Fraction(1,2), Fraction(1), Fraction(-1,2), Fraction(-1)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random U and V
            U = [[random.choice(values) for _ in range(R)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(R)] for _ in range(m*p)]
            
            # For each (i,k), we need: sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
            # Let S_ik[r] = sum_j U[i*m+j,r]*V[j*p+k,r]
            # Then: sum_r S_ik[r] * W_ik[r] = 1
            
            W = [[Fraction(0)] * R for _ in range(p*n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Compute S_ik
                    S = [Fraction(0)] * R
                    for j in range(m):
                        for r in range(R):
                            S[r] += U[i*m+j][r] * V[j*p+k][r]
                    
                    # Find if there's a simple solution: check if some S[r] is ±1 or ±2
                    # Try: set W_ik[r] = 1/S[r] for one r where S[r] != 0, if 1/S[r] is half-integer
                    found = False
                    for r in range(R):
                        if S[r] != 0:
                            w_val = Fraction(1) / S[r]
                            # Check if w_val is a half-integer
                            if w_val.denominator in (1, 2):
                                W[k*n+i][r] = w_val
                                found = True
                                break
                    
                    if not found:
                        # Try sum of two: S[r1]*w1 + S[r2]*w2 = 1 with w1,w2 half-integers
                        found = False
                        for r1 in range(R):
                            if S[r1] == 0:
                                continue
                            for r2 in range(R):
                                if r2 == r1 or S[r2] == 0:
                                    continue
                                # S[r1]*w1 + S[r2]*w2 = 1
                                # Try small half-integer values for w1, w2
                                for w1 in [Fraction(0), Fraction(1,2), Fraction(1), Fraction(-1,2), Fraction(-1)]:
                                    rem = Fraction(1) - S[r1]*w1
                                    if rem == 0:
                                        W[k*n+i][r1] = w1
                                        found = True
                                        break
                                    if S[r2] != 0:
                                        w2 = rem / S[r2]
                                        if w2.denominator in (1, 2):
                                            W[k*n+i][r1] = w1
                                            W[k*n+i][r2] = w2
                                            found = True
                                            break
                            if found:
                                break
                        if not found:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Get schoolbook as fallback
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Try lower ranks
    for R in range(10, best_rank):
        if time.time() > deadline:
            break
        
        result = try_rank_search(R, max_iters=1000)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best
