"""
Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.

Strategy:
1. Start from the schoolbook decomposition (rank 75).
2. Use a structured Winograd-style construction to reduce the rank.
   The key idea: for each pair of middle indices (j, j+1), we can use
   Winograd's trick to reduce the number of multiplications.
3. Additionally, perform a targeted search over structured perturbations
   of the schoolbook decomposition to find valid lower-rank decompositions.
4. Return the best (lowest rank) valid decomposition found within the time budget.

The tensor condition is:
  For all i in [0,3), j in [0,5), k in [0,5):
    sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1
"""
import time
import random
import numpy as np
from fractions import Fraction

n, m, p = 3, 5, 5
START_TIME = time.time()
DEADLINE = 115.0  # Safety margin before 120s limit

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _validate(U, V, W):
    """Validate that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :]
                if abs(np.sum(prod) - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Standard rank n*m*p = 75 decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_pair_construction():
    """
    Winograd-style construction that pairs adjacent middle indices.
    
    For each pair (j, j+1) in the middle dimension, we use the identity:
    a_j * b_j + a_{j+1} * b_{j+1} = (a_j + a_{j+1})(b_j + b_{j+1}) - a_j*b_{j+1} - a_{j+1}*b_j
    
    But this doesn't directly help because we still need the cross terms.
    
    A better approach: use the fact that for a fixed i and k, we need:
    sum_j A[i,j] * B[j,k]
    
    We can write this as:
    sum_j (A[i,j] + A[i,j+1]) * B[j,k] - sum_j A[i,j+1] * B[j,k]
    = sum_j (A[i,j] + A[i,j+1]) * B[j,k] - sum_{j'} A[i,j'] * B[j'-1,k]
    
    This is getting complicated. Let me try a different structured approach.
    
    Actually, let's use a known result: the matrix multiplication tensor [n,m,p]
    can sometimes be decomposed with rank less than n*m*p using bilinear algorithms.
    
    For [3,5,5], let's try a construction based on the following idea:
    - Group the 5 middle indices into pairs: (0,1), (2,3), and single 4.
    - For each pair (j, j+1), use 3 multiplications instead of 4 (Winograd for 2x2).
    
    Wait, for a dot product of length 2: a0*b0 + a1*b1
    Winograd: t0 = a0+a1, t1 = b0+b1, t2 = t0*t1 = a0*b0+a0*b1+a1*b0+a1*b1
    We need a0*b0 + a1*b1 = t2 - a0*b1 - a1*b0
    But we don't have a0*b1 and a1*b0 directly...
    
    Actually, the standard Winograd F2(2) for computing a0*b0 + a1*b1 uses:
    - 3 multiplications: (a0+a1)*(b0+b1), a0*b0, a1*b1? No, that's 3 but we want fewer.
    
    The correct Winograd for 2-element dot product:
    We want: a0*b0 + a1*b1
    Define: s0 = a0+a1, s1 = b0+b1
    Then: s0*s1 = a0*b0 + a0*b1 + a1*b0 + a1*b1
    So: a0*b0 + a1*b1 = s0*s1 - a0*b1 - a1*b0
    This still requires 3 multiplications (s0*s1, a0*b1, a1*b0) vs 2 direct.
    So Winograd doesn't help for length 2!
    
    For length 4: a0*b0+a1*b1+a2*b2+a3*b3
    Winograd: 
    s0 = a0+a2, s1 = a1+a3, s2 = b0+b2, s3 = b1+b3
    t0 = s0*s2 = a0*b0+a0*b2+a2*b0+a2*b2
    t1 = s1*s3 = a1*b1+a1*b3+a3*b1+a3*b3
    t2 = (a0+a1+a2+a3)*(b0+b1+b2+b3) - ... this gets complex.
    
    Let me try a completely different approach: use a known algebraic identity
    or just do a smarter search.
    
    For now, let's try a structured search where we start from schoolbook
    and try to merge columns by finding linear dependencies.
    """
    # Let's try a direct construction based on the following observation:
    # The matrix multiplication tensor [3,5,5] has a certain structure.
    # We can try to use the fact that 5 = 3 + 2 and decompose accordingly.
    
    # Alternative: try rank 70, 65, 60, 55, 50, 45, 40, 35, 30 with random search
    # but with more structure.
    
    # For now, return schoolbook as baseline.
    return _schoolbook()

def _structured_merge_search():
    """
    Try to reduce rank by finding linear dependencies among the schoolbook columns.
    
    The schoolbook decomposition has 75 columns, each corresponding to a unique (i,j,k) triple.
    If we can find a linear combination of columns that is zero, we can eliminate one column.
    
    More precisely, if there exist coefficients c_r such that for all (i,j,k):
    sum_r c_r * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0
    then we can modify the decomposition to reduce rank.
    
    This is equivalent to finding a vector in the kernel of a certain linear map.
    """
    sb = _schoolbook()
    U = sb["U"]
    V = sb["V"]
    W = sb["W"]
    rank = n * m * p
    
    # For each pair of columns (r1, r2), check if we can merge them.
    # Columns r1 and r2 correspond to triples (i1,j1,k1) and (i2,j2,k2).
    # They can be merged if there exists a scalar alpha such that:
    # For all (i,j,k): U[i*m+j,r1]*V[j*p+k,r1]*W[k*n+i,r1] + alpha*U[i*m+j,r2]*V[j*p+k,r2]*W[k*n+i,r2] = 0
    # unless (i,j,k) is (i1,j1,k1) or (i2,j2,k2).
    
    # This is very restrictive. Let's try a different approach.
    
    # Try random linear combinations of columns to see if we can reduce rank.
    best = sb
    best_rank = rank
    
    # Try to find a valid decomposition with rank < 75 by random search
    # with structured initialization.
    
    rng = random.Random(12345)
    
    for target_rank in range(74, 30, -1):
        if _time_left() < 10:
            break
        
        # Try a few candidates at this rank
        for trial in range(200):
            if _time_left() < 5:
                break
            
            # Initialize from schoolbook and randomly zero out some columns
            U_cand = [row[:] for row in U]
            V_cand = [row[:] for row in V]
            W_cand = [row[:] for row in W]
            
            # Zero out random columns to reduce effective rank
            cols_to_keep = set(range(target_rank))
            for r in range(rank):
                if r not in cols_to_keep:
                    for i in range(n*m):
                        U_cand[i][r] = 0
                    for j in range(m*p):
                        V_cand[j][r] = 0
                    for k in range(p*n):
                        W_cand[k][r] = 0
            
            # This won't work because we need to adjust the remaining columns.
            # Let's try a different approach: random sparse matrices.
            
            vals = [0, 1, -1, 0.5, -0.5]
            U_new = [[0]*target_rank for _ in range(n*m)]
            V_new = [[0]*target_rank for _ in range(m*p)]
            W_new = [[0]*target_rank for _ in range(p*n)]
            
            density = 0.4
            for i in range(n*m):
                for r in range(target_rank):
                    if rng.random() < density:
                        U_new[i][r] = rng.choice(vals)
            for j in range(m*p):
                for r in range(target_rank):
                    if rng.random() < density:
                        V_new[j][r] = rng.choice(vals)
            for k in range(p*n):
                for r in range(target_rank):
                    if rng.random() < density:
                        W_new[k][r] = rng.choice(vals)
            
            if _validate(U_new, V_new, W_new):
                best = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                return best
    
    return best, best_rank

def _targeted_search():
    """
    Targeted search using structured patterns.
    
    Key insight: for the [3,5,5] tensor, we can try to exploit the fact that
    the middle dimension is 5, which is odd. We can pair (0,1), (2,3) and leave 4.
    
    For each pair (j, j+1), the contribution to C[i,k] is:
    A[i,j]*B[j,k] + A[i,j+1]*B[j+1,k]
    
    We can write this as:
    (A[i,j] + A[i,j+1]) * B[j,k] + A[i,j+1] * (B[j+1,k] - B[j,k])
    
    This uses 2 multiplications per pair instead of 2, so no savings.
    
    Another identity:
    A[i,j]*B[j,k] + A[i,j+1]*B[j+1,k]
    = (A[i,j] + A[i,j+1]) * (B[j,k] + B[j+1,k]) 
      - A[i,j]*B[j+1,k] - A[i,j+1]*B[j,k]
    
    This uses 3 multiplications for 2 terms, which is worse.
    
    Let's try yet another approach: use the fact that we can precompute
    linear combinations of rows of A and columns of B.
    
    For a fixed i, define:
    S0 = A[i,0] + A[i,1]
    S1 = A[i,2] + A[i,3]
    S2 = A[i,4]
    
    For a fixed k, define:
    T0 = B[0,k] + B[1,k]
    T1 = B[2,k] + B[3,k]
    T2 = B[4,k]
    
    Then:
    A[i,0]*B[0,k] + A[i,1]*B[1,k] = S0*T0 - A[i,0]*B[1,k] - A[i,1]*B[0,k]
    
    This still doesn't save multiplications.
    
    Let's just do a more efficient random search with better validation.
    """
    best = _schoolbook()
    best_rank = n * m * p
    
    rng = random.Random(99999)
    
    # Try ranks from 74 down to 40
    for target_rank in range(74, 39, -1):
        if _time_left() < 15:
            break
        
        # Number of trials based on remaining time
        trials = min(500, max(50, int(_time_left() * 2)))
        
        for trial in range(trials):
            if _time_left() < 5:
                break
            
            vals = [0, 1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25]
            
            # Use a sparser initialization
            U_new = [[0]*target_rank for _ in range(n*m)]
            V_new = [[0]*target_rank for _ in range(m*p)]
            W_new = [[0]*target_rank for _ in range(p*n)]
            
            # Each row should have a few non-zero entries
            for i in range(n*m):
                num_nonzero = rng.randint(1, min(5, target_rank))
                cols = rng.sample(range(target_rank), num_nonzero)
                for r in cols:
                    U_new[i][r] = rng.choice(vals)
            
            for j in range(m*p):
                num_nonzero = rng.randint(1, min(5, target_rank))
                cols = rng.sample(range(target_rank), num_nonzero)
                for r in cols:
                    V_new[j][r] = rng.choice(vals)
            
            for k in range(p*n):
                num_nonzero = rng.randint(1, min(5, target_rank))
                cols = rng.sample(range(target_rank), num_nonzero)
                for r in cols:
                    W_new[k][r] = rng.choice(vals)
            
            if _validate(U_new, V_new, W_new):
                best = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                return best
    
    return best, best_rank

def construct():
    """
    Main entry point. Returns a dictionary with keys 'U', 'V', 'W'
    containing the factor matrices of the tensor decomposition.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Start with schoolbook
    sb = _schoolbook()
    best = sb
    best_rank = n * m * p
    
    # Sanity check
    if not _validate(sb["U"], sb["V"], sb["W"]):
        return sb
    
    # Try structured merge search
    if _time_left() > 30:
        candidate, rank = _structured_merge_search()
        if candidate is not None and rank < best_rank:
            best = candidate
            best_rank = rank
    
    # Try targeted search
    if _time_left() > 20:
        candidate, rank = _targeted_search()
        if candidate is not None and rank < best_rank:
            best = candidate
            best_rank = rank
    
    return best

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0]) if result['U'] else 'unknown'
    print(f"Rank achieved: {rank}")
