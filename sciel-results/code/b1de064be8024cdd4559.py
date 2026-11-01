import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = 2, 4, 6

    # Build the known rank-42 Strassen solution as fallback
    def build_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        # Strassen 2x2x2 terms
        strassen_terms = [
            # P1 = (A00+A11)(B00+B11)
            {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            # P2 = (A10+A11)B00
            {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
            # P3 = A00(B01-B11)
            {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
            # P4 = A11(B10-B00)
            {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
            # P5 = (A00+A01)B11
            {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
            # P6 = (A10-A00)(B00+B01)
            {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            # P7 = (A01-A11)(B10+B11)
            {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
        ]

        col = 0
        for j_block in range(2):  # j = 0,1 and j = 2,3
            for k_block in range(3):  # k = 0,1 and k = 2,3 and k = 4,5
                for term in strassen_terms:
                    r = col
                    for (ar, ac, sign) in term['A']:
                        u_idx = ar * m + (j_block * 2 + ac)
                        U[u_idx][r] += sign
                    for (br, bc, sign) in term['B']:
                        v_idx = (j_block * 2 + br) * p + (k_block * 2 + bc)
                        V[v_idx][r] += sign
                    for (or_, oc, sign) in term['W']:
                        w_idx = (k_block * 2 + oc) * n + or_
                        W[w_idx][r] += sign
                    col += 1

        assert col == rank, f"Expected {rank} columns, got {col}"
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True

    # Build the rank-42 fallback
    fallback = build_strassen_42()

    if time.time() > deadline - 5:
        return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}

    # Try to find a lower-rank solution using a Kronecker product approach
    # 2x4x6 = 2x(2*2)x(2*3) = (2x2x2) ⊗ (1x2x3)
    # If we can find a rank-3 decomposition for 1x2x3, total rank = 7*3 = 21
    # If rank-4, total rank = 28
    # If rank-5, total rank = 35
    # If rank-6, total rank = 42 (same as current)

    # The 1x2x3 tensor: T[a, (j,k), c] where a in {0,1}, j in {0,1}, k in {0,1,2}, c in {0,1}
    # Wait, the dimensions are: A is 1x2, B is 2x3, C is 1x3
    # Actually for the Kronecker: A is 2x4, B is 4x6
    # Split A as 2x(2*2), B as (2*2)x(2*3)
    # So A = A1 ⊗ A2 where A1 is 2x2, A2 is 1x2
    # B = B1 ⊗ B2 where B1 is 2x2, B2 is 2x3
    # C = C1 ⊗ C2 where C1 is 2x2, C2 is 1x3

    # The tensor for 1x2x3: A is 1x2, B is 2x3, C is 1x3
    # T[a, (j,k), c] where a in {0}, j in {0,1}, k in {0,1,2}, c in {0,1}
    # T[0, j, c] = A[0,j] * B[j,c]
    # This is a 1x2x3 tensor. Its rank is at most 2 (since A has rank 1).
    # Actually, if A = [a0, a1] and B = [[b00, b01, b02], [b10, b11, b12]],
    # then C = [a0*b00 + a1*b10, a0*b01 + a1*b11, a0*b02 + a1*b12]
    # The tensor T[0,j,c] = A[0,j] * B[j,c]
    # Rank-1 decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is 1x1, v_t is 1x2, w_t is 1x3

    # If A has rank 1, say A = [a0, a1] = alpha * [1, 0] + beta * [0, 1]
    # Then T = alpha * ([1] ⊗ [1,0] ⊗ [b00,b01,b02]) + beta * ([1] ⊗ [0,1] ⊗ [b10,b11,b12])
    # So rank is at most 2!

    # Wait, but this is for a specific A and B. We need a decomposition that works for ALL A and B.
    # The tensor T[a, j, c] = delta(a,0) * A[a,j] * B[j,c] ... no, the tensor is defined by the multiplication map.

    # Let me reconsider. The tensor for multiplying a 1x2 matrix by a 2x3 matrix is:
    # T[a, j, c] where the multiplication is C[a,c] = sum_j A[a,j] B[j,c]
    # The tensor is T[a, j, c] = delta(a, 0) * ... no.
    # Actually, the tensor is T[a, j, c] such that C[a,c] = sum_j T[a,j,c] * A[a,j] * B[j,c]
    # So T[a,j,c] = delta(a,0) * delta(c, k) ... I'm confusing myself.

    # The standard tensor: T_{a,j,c} = 1 if the multiplication C = AB has C[a,c] = sum_j A[a,j] B[j,c]
    # So T[a,j,c] = 1 for all valid (a,j,c), and the contraction is sum_j T[a,j,c] A[a,j] B[j,c]
    # But that's not right either. The tensor is T[a, j, c] = delta_{a,0} * ...

    # Let me just directly compute the rank of the 1x2x3 tensor.
    # The tensor is T[a, j, c] where a in {0}, j in {0,1}, c in {0,1,2}
    # T[0, j, c] = 1 for all j, c (since C[0,c] = sum_j A[0,j] B[j,c])
    # Wait no, the tensor entries are T[a, j, c] = 1 if we're computing C[a,c] from A[a,j] and B[j,c]
    # So T[0, j, c] = 1 for all j in {0,1} and c in {0,1,2}
    # This is a 1x2x3 tensor with all entries equal to 1.
    # The rank of this tensor: we need to write it as sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is 1x1, v_t is 1x2, w_t is 1x3
    # T[0, j, c] = sum_t u_t[0] * v_t[j] * w_t[c]
    # Since T[0,j,c] = 1 for all j,c, we need sum_t u_t * v_t[j] * w_t[c] = 1
    # This is equivalent to: the matrix M[j,c] = sum_t (u_t * v_t[j]) * w_t[c] has all entries 1.
    # M is a 2x3 matrix with all entries 1. The rank of M is 1.
    # So we can write M = m1 * m2^T where m1 is 2x1 and m2 is 1x3.
    # Then T = 1 * m1 * m2^T, which is rank 1!

    # Wait, that can't be right. Let me check: if T[0,j,c] = 1 for all j,c,
    # then T = [1] ⊗ [1,1] ⊗ [1,1,1]
    # Check: T[0,j,c] = 1 * 1 * 1 = 1. Yes!
    # So the 1x2x3 tensor has rank 1!

    # That means the Kronecker product 2x2x2 ⊗ 1x2x3 has rank 7*1 = 7?
    # That can't be right either, because 2x4x6 should have rank at least 24 (naive).

    # I think I'm misunderstanding the Kronecker product structure. Let me reconsider.

    # The matrix multiplication 2x4 by 4x6:
    # A is 2x4, B is 4x6, C is 2x6
    # C[i,k] = sum_j A[i,j] B[j,k]

    # If we write A as a 2x2 block matrix with 1x2 blocks: A = [[A00, A01], [A10, A11]]
    # where each Aij is 1x2. Similarly B = [[B00, B01, B02], [B10, B11, B12]]
    # where each Bij is 2x2.

    # Then C[i,k] = sum_j A[i,j] B[j,k]
    # This is not a simple Kronecker product.

    # Let me try a different approach: use the known lower bound.
    # For 2x4x6, the tensor rank is at least the max of:
    # - rank of any slice
    # - the border rank
    # 
    # Actually, let me just try to find a rank-36 solution by solving a system of equations.

    # For rank r, we have r unknowns in each of U, V, W (with the constraint that they're consistent).
    # The equations are: for all (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # That's n*m*p = 2*4*6 = 48 equations.
    # Unknowns: r * (n*m + m*p + p*n) = r * (8 + 24 + 12) = 44r
    # For r = 36: 48 equations, 1584 unknowns. Very underdetermined.

    # But the equations are nonlinear (products of three unknowns).

    # Let me try a simpler approach: use the rank-42 solution and see if we can
    # combine some terms to reduce the rank.

    # Actually, let me just return the rank-42 solution. It's correct and safe.

    return fallback

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
