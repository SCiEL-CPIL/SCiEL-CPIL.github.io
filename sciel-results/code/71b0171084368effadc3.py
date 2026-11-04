"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Should be 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k
    # else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all other entries are zero
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue  # valid entry, already checked
                s = np.sum(U_arr[:, a] * V_arr[:, b] * W_arr[:, c])
                if abs(s) > tol:
                    return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduced_rank(n, m, p):
    """Attempt to find a lower-rank decomposition for n=2.
    
    For n=2, m=5, p=5, we exploit the structure:
    C[0,k] = sum_j A[0,j]*B[j,k]
    C[1,k] = sum_j A[1,j]*B[j,k]
    
    We can write this as a 2x5 times 5x5 multiplication.
    Known upper bound: for 2xn matrix mult, rank <= 2n - 1 for certain cases,
    but here we have 2x5 * 5x5.
    
    Strategy: Use the fact that the two rows of A can be combined.
    For each pair of columns in B, we can share some multiplications.
    
    A concrete construction with rank < 50:
    We'll use the identity that for 2x1 * 1x1, rank is 2.
    For 2x5 * 5x5, think of it as 5 independent 2x1 * 1x1 problems
    plus cross-terms. Actually, let's try a direct approach.
    
    We'll construct a decomposition where some rank terms are shared
    between the two output rows.
    """
    if n != 2:
        return None
    
    # Try rank 45 construction
    # Idea: for each j, instead of having separate terms for (0,j,k) and (1,j,k),
    # try to combine them. But this is tricky because B[j,k] is the same.
    
    # Alternative: use the fact that we can express:
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    # 
    # = sum_j [(A[0,j]+A[1,j])/2 * B[j,k] + (A[0,j]-A[1,j])/2 * B[j,k]]
    # and
    # = sum_j [(A[0,j]+A[1,j])/2 * B[j,k] - (A[0,j]-A[1,j])/2 * B[j,k]]
    #
    # This still gives 2*5*5 = 50 terms if we expand.
    
    # Let me try a different approach: group by k.
    # For fixed k, we need to compute [A[0,:]; A[1,:]] @ B[:,k]
    # This is a 2x5 times 5x1 = 2x1 multiplication, which has rank at most 5
    # (since min(2*5, 5*1, 2*1) = 2, but the tensor rank of 2x5*5x1 is 5).
    # Wait, the bilinear complexity of 2x5 * 5x1 is at most 5*2=10, but actually
    # it's at most 5 since we can do it as 5 multiplications.
    # 
    # So for each k, rank <= 5, total rank <= 5*5 = 25? That can't be right
    # because we need to output 10 values and each needs at least... 
    # Actually no, the 5 multiplications for each k produce intermediate values
    # that are then combined. Let me reconsider.
    
    # For fixed k:
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    # 
    # This is 10 multiplications (A[0,j]*B[j,k] and A[1,j]*B[j,k] for j=0..4).
    # But we can't share across different k because B[j,k] differs.
    
    # So the straightforward bound is 2*5*5 = 50.
    
    # To do better, we need to find linear combinations that allow sharing.
    # One known technique: for 2x2 * 2x2, Strassen reduces from 8 to 7.
    # For 2x5 * 5x5, we might be able to reduce from 50 to something less.
    
    # Let's try a constructive approach with rank 48 or 47.
    # We'll keep most terms as-is but try to merge some.
    
    # Actually, let me try to verify if rank 50 is optimal or if we can do 49.
    # For now, let's just return the schoolbook and verify it works.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try reduced rank approach
    if still_going():
        result = _try_reduced_rank(n, m, p)
        if result is not None:
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                best = result
                best_rank = len(result['U'][0])
    
    return best
# EVOLVE-BLOCK-END
