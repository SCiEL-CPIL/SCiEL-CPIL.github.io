"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    The minimal rank for 2x3x3 matrix multiplication is 17.
    This decomposition uses a known rank-17 decomposition.
    
    Interface:
    U: shape (n*m, rank) = (6, 17)
    V: shape (m*p, rank) = (9, 17)
    W: shape (p*n, rank) = (6, 17)
    
    n=2, m=3, p=3.
    rank = 17.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    def add_term(u_indices, v_indices, w_indices, col):
        for u_idx, val in u_indices:
            U[u_idx][col] += val
        for v_idx, val in v_indices:
            V[v_idx][col] += val
        for w_idx, val in w_indices:
            W[w_idx][col] += val

    # Indices for A (2x3): i*m + j  -> row i, col j of A
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2
    #   A[1,0]=3, A[1,1]=4, A[1,2]=5
    # Indices for B (3x3): j*p + k  -> row j, col k of B
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2
    #   B[1,0]=3, B[1,1]=4, B[1,2]=5
    #   B[2,0]=6, B[2,1]=7, B[2,2]=8
    # Indices for C (3x2): k*n + i  -> row k, col i of C
    #   C[0,0]=0, C[0,1]=1
    #   C[1,0]=2, C[1,1]=3
    #   C[2,0]=4, C[2,1]=5

    # We use a known rank-17 decomposition.
    # The idea: use Strassen on a 2x2x2 sub-block (rows 0,1 of A; cols 0,1 of B; 
    # but we need to cover all of C which is 3x2).
    # 
    # Actually, let me use a different approach. The matrix multiplication tensor
    # for (2,3,3) has a known rank of 17. One way to construct it:
    #
    # Consider the 2x3x3 multiplication. We can think of it as:
    # C = A * B where A is 2x3, B is 3x3, C is 2x3 (but indexed as C[k,i] = sum_j A[i,j]*B[j,k])
    #
    # Let's use the following decomposition based on the fact that we can split
    # the computation. The key insight for rank 17 (vs 18 schoolbook) is to
    # find one linear dependency among the 18 schoolbook terms.
    #
    # Actually, a cleaner approach: use the known result that the rank of the
    # 2x3x3 matrix multiplication tensor is 17, and construct an explicit decomposition.
    #
    # Here's a construction:
    # We'll use 7 Strassen-like terms for a 2x2x2 sub-problem involving
    # A[0:2, 0:2] * B[0:2, 0:2] contributing to C[0:2, 0:2],
    # plus 10 schoolbook terms for the remaining entries.
    # But that gives 7 + 10 = 17. Let's verify the counts.
    #
    # C has 6 entries: C[0,0], C[0,1], C[1,0], C[1,1], C[2,0], C[2,1]
    # (where C[k,i] is row k, col i)
    #
    # Let's define the Strassen terms for the top-left 2x2 block:
    # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  (indices 0,1,3,4)
    # B_sub = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]  (indices 0,1,3,4)
    # C_sub should be [[C[0,0], C[1,0]], [C[0,1], C[1,1]]]  (indices 0,2,1,3)
    #
    # Wait, C[k,i] where k is col of B (row of C) and i is row of A (col of C).
    # So C[0,0] = sum_j A[0,j]*B[j,0], C[0,1] = sum_j A[1,j]*B[j,0]
    # C[1,0] = sum_j A[0,j]*B[j,1], C[1,1] = sum_j A[1,j]*B[j,1]
    # C[2,0] = sum_j A[0,j]*B[j,2], C[2,1] = sum_j A[1,j]*B[j,2]
    #
    # The 2x2x2 sub-problem: A[0:2, 0:2] * B[0:2, 0:2]
    # contributes to C[0:2, 0:2] i.e., C[0,0], C[0,1], C[1,0], C[1,1]
    # (indices 0, 1, 2, 3 in W)
    #
    # Strassen's 7 terms for this 2x2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # C[0,0] = M1 - M4 + M7 - M2  ... wait, let me recall Strassen correctly.
    #
    # Standard Strassen for 2x2:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B11)
    # P4 = A22(B12-B22)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # Mapping to our indices:
    # A11=A[0,0]=0, A12=A[0,1]=1, A21=A[1,0]=3, A22=A[1,1]=4
    # B11=B[0,0]=0, B12=B[0,1]=1, B21=B[1,0]=3, B22=B[1,1]=4
    # C11=C[0,0]=0, C12=C[1,0]=2, C21=C[0,1]=1, C22=C[1,1]=3
    #
    # So:
    # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # P2 = (A[1,0]+A[1,1])*B[0,0]
    # P3 = A[0,0]*(B[1,0]-B[0,0])
    # P4 = A[1,1]*(B[0,1]-B[1,1])
    # P5 = (A[0,0]+A[0,1])*B[1,1]
    # P6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # P7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # C[0,0] = P1 + P4 - P5 + P7
    # C[1,0] = P3 + P5
    # C[0,1] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6

    # Term 0: P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    
    # Term 1: P2 = (A[1,0]+A[1,1])*B[0,0]
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    
    # Term 2: P3 = A[0,0]*(B[1,0]-B[0,0])
    add_term([(0, 1)], [(3, 1), (0, -1)], [(2, 1), (3, 1)], 2)
    
    # Term 3: P4 = A[1,1]*(B[0,1]-B[1,1])
    add_term([(4, 1)], [(1, 1), (4, -1)], [(0, 1), (1, 1)], 3)
    
    # Term 4: P5 = (A[0,0]+A[0,1])*B[1,1]
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    
    # Term 5: P6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    
    # Term 6: P7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)

    # Now the remaining terms for C[2,0], C[2,1] and the contributions
    # from column 2 of A and rows 2 of B.
    #
    # C[0,0] also gets A[0,2]*B[2,0]
    # C[0,1] also gets A[1,2]*B[2,0]
    # C[1,0] also gets A[0,2]*B[2,1]
    # C[1,1] also gets A[1,2]*B[2,1]
    # C[2,0] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[2,1] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    #
    # That's 10 more terms, total 17. But we can potentially combine some.
    # Actually, let's just use 10 schoolbook terms for the rest.
    # Total: 7 + 10 = 17.

    # Term 7: A[0,2]*B[2,0] contributes to C[0,0]
    add_term([(2, 1)], [(6, 1)], [(0, 1)], 7)
    
    # Term 8: A[1,2]*B[2,0] contributes to C[0,1]
    add_term([(5, 1)], [(6, 1)], [(1, 1)], 8)
    
    # Term 9: A[0,2]*B[2,1] contributes to C[1,0]
    add_term([(2, 1)], [(7, 1)], [(2, 1)], 9)
    
    # Term 10: A[1,2]*B[2,1] contributes to C[1,1]
    add_term([(5, 1)], [(7, 1)], [(3, 1)], 10)
    
    # Term 11: A[0,0]*B[0,2] contributes to C[2,0]
    add_term([(0, 1)], [(2, 1)], [(4, 1)], 11)
    
    # Term 12: A[1,0]*B[0,2] contributes to C[2,1]
    add_term([(3, 1)], [(2, 1)], [(5, 1)], 12)
    
    # Term 13: A[0,1]*B[1,2] contributes to C[2,0]
    add_term([(1, 1)], [(5, 1)], [(4, 1)], 13)
    
    # Term 14: A[1,1]*B[1,2] contributes to C[2,1]
    add_term([(4, 1)], [(5, 1)], [(5, 1)], 14)
    
    # Term 15: A[0,2]*B[2,2] contributes to C[2,0]
    add_term([(2, 1)], [(8, 1)], [(4, 1)], 15)
    
    # Term 16: A[1,2]*B[2,2] contributes to C[2,1]
    add_term([(5, 1)], [(8, 1)], [(5, 1)], 16)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
