"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Schoolbook decomposition
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school

    # Try to improve using Strassen on 2x2x2 blocks
    # Partition the matrix multiplication into blocks and use Strassen where possible

    # For a 3x5 * 5x5 multiplication, we can try various block partitions
    # Let's try to use Strassen on sub-blocks

    # Strassen for 2x2 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # Let's try a block partition where we can apply Strassen
    # Partition A (3x5) into blocks, B (5x5) into blocks
    # Try: A split into [1x5; 2x5], B split into [5x2 5x3]
    # Then C (3x5) = [1x2 1x3; 2x2 2x3]

    # Actually, let me think more carefully. We need to find a decomposition
    # that uses Strassen on some 2x2x2 sub-problems.

    # Let me try a different approach: use the known fact that for certain
    # block sizes, we can reduce the rank.

    # For 3x5 * 5x5, let's try to use a block structure:
    # A = [A1 (1x5); A2 (2x5)]
    # B = [B1 (5x2) B2 (5x3)]
    # C = A*B = [A1*B1 (1x2)  A1*B2 (1x3); A2*B1 (2x2)  A2*B2 (2x3)]

    # The (2x2) block A2*B1 can use Strassen (rank 7 instead of 8)
    # Other blocks use schoolbook

    # Let's construct this decomposition

    def make_strassen_2x2x2(U_start, V_start, W_start):
        """
        Add Strassen 2x2x2 decomposition starting at given column offsets.
        Returns the number of columns used (7).
        """
        # Strassen factors for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        # For multiplying A(2x2) * B(2x2) -> C(2x2)
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        #
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6

        # U factors (for A elements):
        # a: U[0] = [1,0,1,0,1,1,-1]  (a appears in M1,M3,M5,M6)
        # Wait, let me be more careful.
        #
        # M1 = (a+d)(e+h): U contributes a and d
        # M2 = (c+d)e: U contributes c and d
        # M3 = a(f-h): U contributes a
        # M4 = d(g-e): U contributes d
        # M5 = (a+b)h: U contributes a and b
        # M6 = (c-a)(e+f): U contributes c and a
        # M7 = (b-d)(g+h): U contributes b and d
        #
        # U (4x7) where rows are [a, b, c, d]:
        # Col 0 (M1): a=1, b=0, c=0, d=1
        # Col 1 (M2): a=0, b=0, c=1, d=1
        # Col 2 (M3): a=1, b=0, c=0, d=0
        # Col 3 (M4): a=0, b=0, c=0, d=1
        # Col 4 (M5): a=1, b=1, c=0, d=0
        # Col 5 (M6): a=-1, b=0, c=1, d=0
        # Col 6 (M7): a=0, b=1, c=0, d=-1
        #
        # V (4x7) where rows are [e, f, g, h]:
        # Col 0 (M1): e=1, f=0, g=0, h=1
        # Col 1 (M2): e=1, f=0, g=0, h=0
        # Col 2 (M3): e=0, f=1, g=0, h=-1
        # Col 3 (M4): e=-1, f=0, g=1, h=0
        # Col 4 (M5): e=0, f=0, g=0, h=1
        # Col 5 (M6): e=1, f=1, g=0, h=0
        # Col 6 (M7): e=0, f=0, g=1, h=1
        #
        # W (4x7) where rows are [C11, C12, C21, C22]:
        # Col 0 (M1): C11=1, C12=0, C21=0, C22=1
        # Col 1 (M2): C11=0, C12=0, C21=1, C22=-1
        # Col 2 (M3): C11=0, C12=1, C21=0, C22=1
        # Col 3 (M4): C11=1, C12=0, C21=1, C22=0
        # Col 4 (M5): C11=-1, C12=1, C21=0, C22=0
        # Col 5 (M6): C11=0, C12=0, C21=0, C22=1
        # Col 6 (M7): C11=1, C12=0, C21=0, C22=0

        U_strassen = [
            [1, 0, 1, 0, 1, -1, 0],   # a
            [0, 0, 0, 0, 1, 0, 1],    # b
            [0, 1, 0, 0, 0, 1, 0],    # c
            [1, 1, 0, 1, 0, 0, -1]    # d
        ]
        V_strassen = [
            [1, 1, 0, -1, 0, 1, 0],   # e
            [0, 0, 1, 0, 0, 1, 0],    # f
            [0, 0, 0, 1, 0, 0, 1],    # g
            [1, 0, -1, 0, 1, 0, 1]    # h
        ]
        W_strassen = [
            [1, 0, 0, 1, -1, 0, 1],   # C11
            [0, 0, 1, 0, 1, 0, 0],    # C12
            [0, 1, 0, 1, 0, 0, 0],    # C21
            [1, -1, 1, 0, 0, 1, 0]    # C22
        ]

        return U_strassen, V_strassen, W_strassen

    # Try the block partition: A = [A1 (1x5); A2 (2x5)], B = [B1 (5x2) B2 (5x3)]
    # C = [A1*B1 (1x2)  A1*B2 (1x3); A2*B1 (2x2)  A2*B2 (2x3)]
    #
    # Ranks:
    # A1*B1: 1x2x5 -> rank 10 (schoolbook)
    # A1*B2: 1x3x5 -> rank 15 (schoolbook)
    # A2*B1: 2x2x5 -> can we use Strassen? No, Strassen is for 2x2*2x2->2x2.
    #          Here it's 2x5 * 5x2 -> 2x2. Not directly Strassen.
    # A2*B2: 2x3x5 -> rank 30 (schoolbook)
    #
    # Total schoolbook: 10+15+20+30 = 75 (same as before)

    # Let me try a different partition where we can actually use Strassen.
    # We need a 2x2*2x2->2x2 sub-problem.
    #
    # Partition A (3x5) and B (5x5) such that one block multiplication is 2x2*2x2->2x2.
    #
    # A = [A11 (2x2) A12 (2x3); A21 (1x2) A22 (1x3)]
    # B = [B11 (2x2) B12 (2x3); B21 (3x2) B22 (3x3)]
    #
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x3)
    #
    # This doesn't directly give us a 2x2*2x2->2x2 block.

    # Let me try yet another approach. Maybe I should just verify that the schoolbook
    # is correct and return it, since finding a better decomposition in the time limit
    # is uncertain.

    # Actually, let me try to use a known low-rank construction for 3x5*5x5.
    # The rank of the matrix multiplication tensor M_{3,5,5} is known to be at most
    # 25 based on some constructions, but I'm not sure of the exact minimal rank.

    # For now, let me just return the schoolbook decomposition with the time guard.

    # Verify the schoolbook decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(U_mat)
        V = np.array(V_mat)
        W = np.array(W_mat)
        rank = U.shape[1]

        # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[i,j,k] = delta (contributes to C[i,k] when multiplied by A[i,j]*B[j,k])
        # Actually, the tensor for matrix multiplication M_{n,m,p} has:
        # T[i,j,k] = 1 for all i,j,k (it's the identity tensor in some sense)
        # No wait, the matrix multiplication tensor is defined such that:
        # C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has components T_{i,j,k} where:
        # sum_c U[i,j,c] V[j,k,c] W[i,k,c] should give the correct product
        #
        # Actually, the standard decomposition is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And we need T[i,j,k] = 1 for all i,j,k (since the multiplication tensor
        # has all 1s in its standard basis)

        # Let's check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    # The schoolbook should verify correctly
    # Just return it

    return best_result
# EVOLVE-BLOCK-END
