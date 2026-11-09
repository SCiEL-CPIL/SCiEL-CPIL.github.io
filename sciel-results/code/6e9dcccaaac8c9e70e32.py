"""Matrix multiplication 2x3x5 - rank-minimal tensor decomposition search."""
import numpy as np
from fractions import Fraction
import time
import random

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
    deadline = start_time + 110  # Safety margin before 120s limit

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

    def try_rank_R(R, max_iters=20000):
        """
        Search for a rank-R decomposition.
        
        Strategy:
        1. Randomly assign U and V entries from {0, ±1/2, ±1}.
        2. For each (i, k), compute S_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r].
        3. Find W_ik such that sum_r W_ik[r] * S_ik[r] = 1, with W_ik entries in {0, ±1/2, ±1, ...}.
           We restrict W_ik to have at most 2 non-zero terms to keep search space small.
        """
        if time.time() > deadline:
            return None
        
        # Values for U and V
        uv_values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
        
        # Values to try for W entries in the 2-term solution
        # If S1*w1 + S2*w2 = 1, and we fix w1, we solve for w2.
        # We try small w1 values.
        w1_candidates = [Fraction(0), Fraction(1,2), Fraction(1), Fraction(-1,2), Fraction(-1)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random U and V
            U = [[random.choice(uv_values) for _ in range(R)] for _ in range(n*m)]
            V = [[random.choice(uv_values) for _ in range(R)] for _ in range(m*p)]
            
            # Pre-compute S_ik for all i, k
            # S_ik is a list of Fractions of length R
            S = [[Fraction(0)] * R for _ in range(p*n)]
            
            for i in range(n):
                for k in range(p):
                    idx_w = k*n + i
                    for r in range(R):
                        s_val = Fraction(0)
                        for j in range(m):
                            s_val += U[i*m+j][r] * V[j*p+k][r]
                        S[idx_w][r] = s_val
            
            # Try to find W for each (i, k)
            W = [[Fraction(0)] * R for _ in range(p*n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    idx_w = k*n + i
                    s_vec = S[idx_w]
                    
                    found = False
                    
                    # Check if any single term works: S[r] * w = 1 => w = 1/S[r]
                    for r in range(R):
                        if s_vec[r] != 0:
                            w_val = Fraction(1) / s_vec[r]
                            # Check if w_val is a half-integer
                            if w_val.denominator in (1, 2):
                                W[idx_w][r] = w_val
                                found = True
                                break
                    
                    if not found:
                        # Check if any pair of terms works
                        # S[r1]*w1 + S[r2]*w2 = 1
                        # We iterate over r1, r2. To avoid duplicates, r1 < r2.
                        # We fix w1 from candidates, solve for w2.
                        for r1 in range(R):
                            if s_vec[r1] == 0:
                                continue
                            for r2 in range(r1 + 1, R):
                                if s_vec[r2] == 0:
                                    continue
                                
                                for w1 in w1_candidates:
                                    rem = Fraction(1) - s_vec[r1] * w1
                                    if rem == 0:
                                        # w2 can be 0
                                        W[idx_w][r1] = w1
                                        # W[idx_w][r2] is already 0
                                        found = True
                                        break
                                    
                                    w2 = rem / s_vec[r2]
                                    # Check if w2 is a half-integer
                                    if w2.denominator in (1, 2):
                                        W[idx_w][r1] = w1
                                        W[idx_w][r2] = w2
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
            
            if valid:
                # Full verification just in case
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None

    # Get schoolbook as fallback
    best = make_schoolbook()
    best_rank = n * m * p
    
    # Theoretical lower bound for rank is max(nm, mp, np) = 15
    # We search from 15 upwards
    for R in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Allocate more iterations for lower ranks as they are harder
        # but we have more time to spare if we start early
        max_iters = 10000 if R < 20 else 5000
        
        result = try_rank_R(R, max_iters=max_iters)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best
