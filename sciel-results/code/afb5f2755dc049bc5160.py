"""Rewritten program: systematic search for rank-minimal 2x3*3x4 matrix mult tensor."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Build the multiplication tensor T[a, b, c] where
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)
    # c = k*n + i (i in 0..n-1, k in 0..p-1)
    # T[a,b,c] = 1 iff a,b,c form a valid (i,j,k) triple, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    # Precompute: for each (a,b) pair, which c indices are valid
    # valid_c[a,b] = list of c where T[a,b,c] != 0 (at most 1)
    # And for each (a,b), the set of c where T[a,b,c]==0
    valid_c = {}
    for a in range(n*m):
        for b in range(m*p):
            i_a, j_a = divmod(a, m)
            j_b, k_b = divmod(b, p)
            if j_a == j_b:
                c_val = k_b * n + i_a
                valid_c[(a,b)] = c_val
            else:
                valid_c[(a,b)] = None

    def validate_fast(U, V, W):
        """Fast vectorized validation. U: (n*m, r), V: (m*p, r), W: (p*n, r)"""
        r = U.shape[1]
        # Compute all T_hat[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Do this efficiently: for each r, outer product
        # T_hat[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # Use einsum: T_hat[a,b,c] = einsum('ar,br,cr->abc', U, V, W)
        T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # Check T_hat == T exactly (integer entries)
        return np.array_equal(T_hat, T)

    # Schoolbook solution: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=int)
    V_sb = np.zeros((m*p, rank_sb), dtype=int)
    W_sb = np.zeros((p*n, rank_sb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Candidate entry values: small integers and half-integers
    # For rank minimization, we want to find decompositions with rank < 24
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6,12,8) = 12
    # Known: for 2x3*3x4, the tensor rank is at most 24 (schoolbook)
    # Let's try to find lower rank solutions

    # Strategy: For small ranks, enumerate structured patterns
    # Each rank component r has support in U, V, W
    # U support: subset of {0..5} (6 entries)
    # V support: subset of {0..11} (12 entries)
    # W support: subset of {0..7} (8 entries)

    # For rank r, we need r components.
    # Key insight: each valid (i,j,k) triple must be "covered" by at least one component
    # where U[i*m+j] != 0, V[j*p+k] != 0, W[k*n+i] != 0
    # And the product must sum to 1, while invalid triples sum to 0.

    # Approach: Use the structure of the tensor.
    # For a fixed (j), the entries involve i in 0..n-1 and k in 0..p-1.
    # The "fiber" at j is an n x p matrix (identity-like in the tensor sense).

    # Let's try a systematic approach for ranks 12-23:
    # For each rank, try to find U, V, W with small integer entries.

    # Key structural approach:
    # The tensor can be seen as: T = sum over (i,j,k) of e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # This is the "matrix multiplication" tensor.

    # For 2x3*3x4: 
    # Input: 6 elements (2x3 matrix)
    # Middle: 12 elements (3x4 matrix) 
    # Output: 8 elements (2x4 matrix)
    # Rank of this tensor: we want minimal r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r

    # Let's use a more clever search:
    # Fix the pattern of which (i,j) each U component touches, etc.
    # Then solve for the values.

    # For rank r, each component r has:
    # - U_r: vector in R^6 (support = some subset S_U_r)
    # - V_r: vector in R^12 (support = some subset S_V_r)
    # - W_r: vector in R^8 (support = some subset S_W_r)

    # The constraint is: for all (a,b,c):
    # sum_r U_r[a] * V_r[b] * W_r[c] = T[a,b,c]

    # This is a system of 6*12*8 = 576 equations (but only 24 are non-zero in T)
    # with r*26 unknowns (6+12+8 per component).

    # For small r, we can enumerate support patterns and solve linear systems.

    # Better approach: use the fact that T has a very specific structure.
    # Let's try a direct algebraic approach.

    # For the 2x3 * 3x4 case:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2}, k in {0,1,2,3}

    # The tensor is: T = sum_{i,j,k} A_{i,j} * B_{j,k} * E_{i,k}
    # where E_{i,k} is the output basis element.

    # Let's try to find a rank-12 or rank-13 solution using structured patterns.

    # Pattern idea: For each j, we have n*p = 8 (i,k) pairs.
    # The "slice" at j is a 2x4 matrix of (i,k) indices.
    
    # Let's try: rank = m * max(n,p) = 3 * 4 = 12
    # For each j (3 values) and each k (4 values), one component:
    # Component (j,k): U has support on {i*m+j for i in 0..n-1} = {j, m+j} (for n=2)
    #                  V has support on {j*p+k} (single element)
    #                  W has support on {k*n+i for i in 0..n-1} = {k*2, k*2+1}
    
    # Let's check if this works:
    # For component (j,k):
    #   U_r[i*m+j] = 1 for i in 0..n-1, 0 otherwise
    #   V_r[j*p+k] = 1, 0 otherwise  
    #   W_r[k*n+i] = 1 for i in 0..n-1, 0 otherwise
    
    # Then T_hat[a,b,c] = sum_{j,k: a=i*m+j, b=j'*p+k, c=k'*n+i'} [j==j'] * [i==i'] * [k==k']
    # = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for the same (i,j,k)
    # That's exactly T! So rank = m*p = 12 works!

    # Wait, let me verify: rank = m*p = 3*4 = 12
    U12 = np.zeros((n*m, m*p), dtype=int)
    V12 = np.zeros((m*p, m*p), dtype=int)
    W12 = np.zeros((p*n, m*p), dtype=int)
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U12[i*m+j, col] = 1
            V12[j*p+k, col] = 1
            for i in range(n):
                W12[k*n+i, col] = 1
            col += 1

    if validate_fast(U12, V12, W12):
        best_rank = 12
        best_U, best_V, best_W = U12, V12, W12
        # Check if we can do even better
    else:
        print("Rank-12 pattern failed validation, checking...")
        # Debug
        T_hat = np.einsum('ar,br,cr->abc', U12, V12, W12, optimize=True)
        print("Mismatch:", np.sum(T_hat != T))

    # Try to go below 12: theoretical lower bound is max(6,12,8)=12
    # Actually the lower bound for matrix mult tensor m_{n,m,p} is max(n*m, m*p, p*n) = 12
    # So rank 12 is optimal if achievable. Let's verify it's correct.

    # If rank 12 works, that's the minimum possible (since m*p=12 is a lower bound)
    # Let's double-check the lower bound argument:
    # The tensor T has "fiber" dimensions. The rank is at least the max of the
    # dimensions of the three modes: max(n*m, m*p, p*n) = max(6, 12, 8) = 12.
    # So 12 is the minimum possible rank.

    # Convert to lists for the return format
    result = {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }

    return result
