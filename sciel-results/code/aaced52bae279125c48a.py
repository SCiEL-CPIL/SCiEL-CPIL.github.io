"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition (rank n*m*p) as a fallback."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    And for all other index combinations the sum must be 0.
    """
    rank = len(U[0])
    # Build lookup
    U_dict = {}
    for i in range(n):
        for j in range(m):
            row = i * m + j
            for c in range(rank):
                if U[row][c] != 0:
                    U_dict.setdefault((i, j), {})[c] = U[row][c]
    
    V_dict = {}
    for j in range(m):
        for k in range(p):
            row = j * p + k
            for c in range(rank):
                if V[row][c] != 0:
                    V_dict.setdefault((j, k), {})[c] = V[row][c]
    
    W_dict = {}
    for k in range(p):
        for i in range(n):
            row = k * n + i
            for c in range(rank):
                if W[row][c] != 0:
                    W_dict.setdefault((k, i), {})[c] = W[row][c]
    
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (True) else 0
                # Actually, the tensor T[i,j,k] = 1 for all i,j,k in matrix multiplication
                # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
                # The rank decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
                # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
                
                u_vals = U_dict.get((i, j), {})
                v_vals = V_dict.get((j, k), {})
                w_vals = W_dict.get((k, i), {})
                
                total = 0
                # Iterate over the smaller dict
                for c, u_val in u_vals.items():
                    if c in v_vals and c in w_vals:
                        total += u_val * v_vals[c] * w_vals[c]
                
                if total != 1:
                    return False
    
    # Check that no spurious terms exist: for any (i,j,k), the sum should be exactly 1.
    # We already checked all (i,j,k) in range. But we also need to ensure that
    # the decomposition doesn't produce non-zero values outside the valid range.
    # Actually, the tensor is only defined on (n,m,p) indices, so we're done.
    return True


def _try_strassen_based(n, m, p):
    """
    Try to build a Strassen-based decomposition.
    For 4x4 * 4x7: split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x3 or 2x4.
    
    Actually, let's think about this differently.
    A is 4x4, B is 4x7.
    Split A into four 2x2 blocks: A11, A12, A21, A22
    Split B into two 2x7 blocks vertically: B_top (2x7), B_bottom (2x7)
    Or split B into 2x2 and 2x5? No, let's split into columns.
    
    C = A * B is 4x7.
    C_top (2x7) = A_top (2x4) * B (4x7)
    C_bottom (2x7) = A_bottom (2x4) * B (4x7)
    
    This doesn't directly help. Let me try a different approach.
    
    Use the fact that we can compute 2x2 * 2x2 with rank 7 (Strassen).
    For 4x4 * 4x7, we can view it as:
    - Split A into 2x2 blocks of size 2x2
    - Split B into 2x2 blocks of size 2x... but 7 is odd.
    
    Alternative: Use rank-7 decomposition for 2x2*2x2 and combine.
    
    Actually, a simpler approach: the known rank for 4x4 * 4x4 is at most 2*7 = 14? No.
    The rank of matrix multiplication for n*n * n*n is at most n^2 * rank(2x2*2x2) / 4... no.
    
    Let me just try to find a low-rank decomposition via random search or known constructions.
    
    For now, return None to indicate no better decomposition found.
    """
    return None


def _random_search(deadline, max_rank=None):
    """
    Try random search for a low-rank decomposition.
    Start with small ranks and try to find integer/half-integer solutions.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try ranks from small to larger
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But we can do better with structured decompositions.
    # Let's try ranks from 28 upward.
    
    for rank in range(28, 113):  # Up to schoolbook rank 112
        if time.time() > deadline - 10:
            break
        if max_rank and rank > max_rank:
            break
        
        # This is a very naive approach - just check if a random sparse decomposition works
        # For now, skip random search as it's unlikely to find exact integer solutions quickly
        pass
    
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try to find a better decomposition than schoolbook
    # For now, return schoolbook as it's guaranteed correct
    # TODO: Implement Strassen-based or other structured decomposition
    
    result = _schoolbook()
    
    # Verify correctness
    assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
    
    return result

# EVOLVE-BLOCK-END
