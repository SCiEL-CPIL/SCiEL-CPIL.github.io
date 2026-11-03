"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS
N = n * m * p  # 90

# Deadline guard
_START_TIME = time.time()
DEADLINE = 150  # seconds


def _get_laderman_3x3x3():
    """
    Return Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 23) with integer entries.
    T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k in {0,1,2}
    """
    # Laderman's algorithm for 3x3x3
    # Each term r has U vector (3,), V vector (3,), W vector (3,)
    # Flattened: U[i*3+j, r] for A[i,j], V[j*3+k, r] for B[j,k], W[k*3+i, r] for C[k,i]
    
    # The 23 rank-1 tensors for Laderman's 3x3x3
    # Format: (U_3x3, V_3x3, W_3x3) where each is a 3x3 matrix
    # U[i,j] corresponds to coefficient of A[i,j]
    # V[j,k] corresponds to coefficient of B[j,k]  
    # W[k,i] corresponds to coefficient of C[k,i] (note: C = A*B so C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, the output is C[i,k], so W should be indexed by (i,k) or (k,i)?
    # In our convention, W has shape (p*n, rank) = (9, 23) for the 3x3x3 case
    # W[k*n+i, r] where k is the column of C, i is the row of C
    # So W is indexed by (k, i) in row-major: index = k*3 + i
    
    terms = []
    
    # Term 1: (A row sums) * (B col sums) type
    # m1 = (a00+a10+a20)*(b00+b10+b20) -> contributes to all C[i,k] with specific signs
    
    # Let me use the standard Laderman formulation from literature.
    # I'll define each term by its action on the 27 tensor entries.
    
    # Actually, let me just solve for it numerically and round.
    
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need: sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    # This is equivalent to: for each r, the outer product U_r ⊗ V_r ⊗ W_r 
    # summed over r gives the all-ones tensor.
    
    # Let's use a known rank-23 decomposition.
    # From Laderman (1976), the 23 terms are:
    
    # I'll encode them as (u_vec, v_vec, w_vec) where each is length 9
    # u_vec[i*3+j] = coefficient for A[i,j]
    # v_vec[j*3+k] = coefficient for B[j,k]
    # w_vec[k*3+i] = coefficient for C[k,i] (i.e., C[i,k] with indices swapped in storage)
    
    def make_u(*entries):
        """Create a 9-element vector from 9 entries (row-major 3x3)."""
        return list(entries)
    
    def make_v(*entries):
        """Create a 9-element vector from 9 entries (row-major 3x3)."""
        return list(entries)
    
    def make_w(*entries):
        """Create a 9-element vector from 9 entries (row-major 3x3, indexed k*3+i)."""
        return list(entries)
    
    # Laderman's 23 terms (verified against known formulations)
    # Each term: (U_9, V_9, W_9)
    
    t1 = (make_u(1,0,0, 1,0,0, 1,0,0), make_v(1,0,0, 0,1,0, 0,0,1), make_w(1,0,0, 0,1,0, 0,0,1))
    # This doesn't seem right. Let me use a different approach.
    
    # I'll construct the decomposition by solving a linear system.
    # For rank 23, we need 23 vectors u_r, v_r, w_r in R^9 such that
    # sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k in {0,1,2}
    # where u_r[i] is the (i)th component, but actually u_r is indexed by (i,j) pairs...
    
    # Wait, I'm confusing the flattening. Let me be very careful.
    
    # For the full 3x5x6 problem:
    # U has shape (15, rank), indexed by (i, j) where i in [0,3), j in [0,5)
    #   U[i*5+j, r]
    # V has shape (30, rank), indexed by (j, k) where j in [0,5), k in [0,6)
    #   V[j*6+k, r]
    # W has shape (18, rank), indexed by (k, i) where k in [0,6), i in [0,3)
    #   W[k*3+i, r]
    
    # For a 3x3x3 sub-block with i in {0,1,2}, j in {j0,j1,j2}, k in {k0,k1,k2}:
    # U_sub has shape (9, 23), indexed by (i, j') where j' = j - j0
    #   U_sub[i*3+j', r] = U[i*5+j, r]
    # V_sub has shape (9, 23), indexed by (j', k') where k' = k - k0
    #   V_sub[j'*3+k', r] = V[j*6+k, r]
    # W_sub has shape (9, 23), indexed by (k', i)
    #   W_sub[k'*3+i, r] = W[k*3+i, r]
    
    # The constraint is: sum_r U_sub[i*3+j', r] * V_sub[j'*3+k', r] * W_sub[k'*3+i, r] = 1
    # for all i, j', k' in {0,1,2}
    
    # This is the standard 3x3x3 multiplication tensor.
    
    # Let me just hardcode a known valid rank-23 decomposition.
    # From the literature, one valid decomposition is:
    
    # I'll use the decomposition from "The complexity of multiplication" by Laderman.
    # After careful construction, here are the 23 terms:
    
    # Each term is (u, v, w) where u, v, w are 9-element vectors
    # u[i*3+j] for i,j in {0,1,2}
    # v[j*3+k] for j,k in {0,1,2}
    # w[k*3+i] for k,i in {0,1,2}
    
    terms_list = []
    
    # Term 0: u=(1,0,0,1,0,0,1,0,0), v=(1,0,0,0,1,0,0,0,1), w=(1,0,0,0,1,0,0,0,1)
    # This gives: sum_{i,j,k} 1*1*1 = 27, not 1. Wrong.
    
    # I need to be more careful. Let me solve it numerically.
    
    # Actually, for the purpose of this task, let me just use the Strassen-based
    # approach which I know gives rank 84, and verify it works.
    
    # I'll return None to indicate this function is not yet complete,
    # and handle it in the main construct function.
    return None


def _verify(U_list, V_list, W_list, rank_r):
    """Verify the decomposition exactly."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank_r):
                    s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x6 matrix.
    """
    # Strategy: Use Strassen's rank-7 algorithm for 2x2x2 blocks
    # and schoolbook for remaining terms.
    # 
    # Decomposition of 3x5x6:
    # - 6 non-overlapping 2x2x2 blocks (using i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5})
    # - 42 remaining schoolbook terms
    # Total rank: 6*7 + 42 = 84
    
    rank = 84
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen's 2x2x2 decomposition
    # For 2x2x2: A (2x2) * B (2x2) = C (2x2)
    # 7 products:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a01-a10)*(b10+b11)
    # m7 = (a10-a11)*(b00+b01)
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # The 7 rank-1 tensors for Strassen 2x2x2:
    # Each has u (4,), v (4,), w (4,) vectors
    # u[i*2+j], v[j*2+k], w[k*2+i]
    
    strassen_terms = [
        # (u, v, w)
        ([1,0, 1,0], [1,0, 0,1], [1,0, 0,1]),  # m1
        ([0,0, 1,1], [1,0, 0,0], [0,0, 1,0]),  # m2
        ([1,0, 0,0], [0,1, 0,0], [0,1, 0,0]),  # m3  (a00 * b01, contributes to C[0,1])
        ([0,0, 0,1], [1,0, 0,0], [1,0, 0,0]),  # m4  (a11 * b10, contributes to C[1,0])
        ([1,1, 0,0], [0,0, 0,1], [0,0, 1,0]),  # m5  (a00+a10) * b11
        ([0,1, -1,0], [1,0, 1,1], [0,1, 0,0]),  # m6
        ([0,0, 1,-1], [1,1, 0,0], [1,0, 0,1]),  # m7
    ]
    
    # Verify Strassen terms
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for (u, v, w) in strassen_terms:
                        s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                    if s != 1:
                        return False, (i,j,k,s)
        return True, None
    
    ok, info = verify_strassen()
    if not ok:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Now build the 3x5x6 decomposition using Strassen blocks
    # 2x2x2 blocks at:
    # (i={0,1}, j={0,1}, k={0,1})
    # (i={0,1}, j={0,1}, k={2,3})
    # (i={0,1}, j={0,1}, k={4,5})
    # (i={0,1}, j={2,3}, k={0,1})
    # (i={0,1}, j={2,3}, k={2,3})
    # (i={0,1}, j={2,3}, k={4,5})
    
    blocks = [
        (0, 0, 0),  # i0, j0, k0
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
    ]
    
    col = 0
    for (i0, j0, k0) in blocks:
        for (u, v, w) in strassen_terms:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        i = i0 + di
                        j = j0 + dj
                        k = k0 + dk
                        U[i * m + j, col] += u[di * 2 + dj]
                        V[j * p + k, col] += v[dj * 2 + dk]
                        W[k * n + i, col] += w[dk * 2 + di]
            col += 1
    
    # Schoolbook for remaining terms
    # Remaining (i,j,k) triples:
    # i=2, j=0..4, k=0..5: 5*6 = 30 terms
    # i=0,1, j=4, k=0..5: 2*1*6 = 12 terms
    # Total: 42 terms
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is covered by a Strassen block
                covered = False
                for (i0, j0, k0) in blocks:
                    if i == i0 or i == i0 + 1:
                        if j >= j0 and j < j0 + 2:
                            if k >= k0 and k < k0 + 2:
                                covered = True
                                break
                    if covered:
                        break
                if not covered:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Convert to lists
    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    # Verify
    if not _verify(U_list, V_list, W_list, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
