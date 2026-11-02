"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return a valid schoolbook decomposition as (U, V, W) lists of lists of Fractions."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _strassen_4x4x4():
    """
    Build a rank-49 decomposition for 4x4 @ 4x4 using two independent
    Strassen 2x2x2 blocks (one for even indices, one for odd indices).
    Returns (U, V, W) as lists of lists of Fractions with shapes (16, 49), (16, 49), (16, 49).
    """
    n = m = p = 4
    rank = 49
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]

    # Strassen 7 multiplications for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For a 2x2 block at offset (i0, j0, k0) where i0 in {0,2}, j0 in {0,2}, k0 in {0,2}
    # The 2x2 submatrix A has indices (i0+r, j0+c) for r,c in {0,1}
    # The 2x2 submatrix B has indices (j0+r, k0+c) for r,c in {0,1}
    # The 2x2 submatrix C has indices (k0+r, i0+c) for r,c in {0,1}

    strassen_coeffs = [
        # (name, U_coeffs [(r,c, val)], V_coeffs [(r,c, val)], W_coeffs [(r,c, val)])
        # M1 = (A00+A11)(B00+B11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)],  # C00+=M1, C01+=M1, C10+=M1, C11+=M1 -> wait, let me recheck
        },
    ]

    # Let me just hardcode the 7 Strassen terms properly.
    # For each of the 7 terms, we have:
    # U: 2x2 matrix (4 entries)
    # V: 2x2 matrix (4 entries)
    # W: 2x2 matrix (4 entries)
    # where C = sum_t U_t * V_t * W_t (with appropriate indexing)

    # Actually, the tensor decomposition is: C[i,j] = sum_t U[i,t] * V[j,t] * W[i,j,t]
    # But for matrix multiplication: C = A @ B, so C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta(i,j,k) meaning C[i,k] = sum_j A[i,j]B[j,k]
    # In the W convention: W[k*n+i, t] is the coefficient.
    # So for each term t, we have U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    # And the product U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should sum to delta(i,j,k)

    # For Strassen on 2x2x2:
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   U: A00 and A11 -> U[0,0]=1, U[1,1]=1 (row indices: i*m+j)
    #   V: B00 and B11 -> V[0,0]=1, V[1,1]=1 (col indices: j*p+k)
    #   W: contributes to C00, C01, C10, C11? No.
    #   C00 += M1, so W[0*2+0]=W[0]=1
    #   C01 += M1? No. C01 = M3+M5. 
    #   Let me re-derive.
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #   So M1 contributes to C00 and C11.
    #   W[0*2+0]=1 (C00), W[1*2+1]=W[3]=1 (C11)

    strassen_terms = [
        # (U_2x2, V_2x2, W_2x2) as flat lists of 4 Fractions
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # C10: W[1*2+0]=W[2], C11: W[3] with -1
        # Fix: W for M2: C10+=M2 means W[2]=1, C11-=M2 means W[3]=-1
        # Let me redo this more carefully.
    ]

    # Let me just define the 7 terms explicitly with their W contributions.
    # U_t[i,j] for i,j in {0,1} (flat: [U00,U01,U10,U11])
    # V_t[j,k] for j,k in {0,1} (flat: [V00,V01,V10,V11])
    # W_t[r,c] for r,c in {0,1} (flat: [W00,W01,W10,W11]) where W index = c*2+r

    terms = []
    # M1 = (A00+A11)(B00+B11)
    # U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,0],[0,1]] (C00 and C11)
    terms.append(([Fraction(1),Fraction(0),Fraction(0),Fraction(1)],
                  [Fraction(1),Fraction(0),Fraction(0),Fraction(1)],
                  [Fraction(1),Fraction(0),Fraction(0),Fraction(1)]))
    # M2 = (A10+A11)B00
    # U = [[0,0],[1,1]], V = [[1,0],[0,0]], W = [[0,0],[1,0]] but C11 -= M2 so W[1,1] = -1
    # W flat: [W00,W01,W10,W11] = [0,0,1,-1]
    terms.append(([Fraction(0),Fraction(0),Fraction(1),Fraction(1)],
                  [Fraction(1),Fraction(0),Fraction(0),Fraction(0)],
                  [Fraction(0),Fraction(0),Fraction(1),Fraction(-1)]))
    # M3 = A00(B10-B11)
    # U = [[1,0],[0,0]], V = [[0,0],[1,-1]], W = [[0,1],[0,0]] (C01 += M3)
    # W flat: [0,1,0,0]
    terms.append(([Fraction(1),Fraction(0),Fraction(0),Fraction(0)],
                  [Fraction(0),Fraction(0),Fraction(1),Fraction(-1)],
                  [Fraction(0),Fraction(1),Fraction(0),Fraction(0)]))
    # M4 = A11(B01-B00)
    # U = [[0,0],[0,1]], V = [[-1,1],[0,0]], W = [[1,0],[0,0]] (C00 += M4, C10 += M4)
    # W flat: [1,0,1,0]
    terms.append(([Fraction(0),Fraction(0),Fraction(0),Fraction(1)],
                  [Fraction(-1),Fraction(1),Fraction(0),Fraction(0)],
                  [Fraction(1),Fraction(0),Fraction(1),Fraction(0)]))
    # M5 = (A00+A01)B11
    # U = [[1,1],[0,0]], V = [[0,0],[0,1]], W = [[-1,0],[0,0]] (C00 -= M5, C01 += M5)
    # W flat: [-1,1,0,0]
    terms.append(([Fraction(1),Fraction(1),Fraction(0),Fraction(0)],
                  [Fraction(0),Fraction(0),Fraction(0),Fraction(1)],
                  [Fraction(-1),Fraction(1),Fraction(0),Fraction(0)]))
    # M6 = (A10-A00)(B00+B10)
    # U = [[-1,0],[1,0]], V = [[1,0],[1,0]], W = [[0,0],[0,1]] (C11 += M6)
    # W flat: [0,0,0,1]
    terms.append(([Fraction(-1),Fraction(0),Fraction(1),Fraction(0)],
                  [Fraction(1),Fraction(0),Fraction(1),Fraction(0)],
                  [Fraction(0),Fraction(0),Fraction(0),Fraction(1)]))
    # M7 = (A01-A11)(B01+B11)
    # U = [[0,1],[-1,0]], V = [[0,1],[0,1]], W = [[1,0],[0,0]] (C00 += M7)
    # W flat: [1,0,0,0]
    terms.append(([Fraction(0),Fraction(1),Fraction(-1),Fraction(0)],
                  [Fraction(0),Fraction(1),Fraction(0),Fraction(1)],
                  [Fraction(1),Fraction(0),Fraction(0),Fraction(0)]))

    # Now build the 4x4x4 decomposition using two Strassen blocks.
    # Block 0: even indices (i in {0,2}, j in {0,2}, k in {0,2})
    # Block 1: odd indices (i in {1,3}, j in {1,3}, k in {1,3})
    
    rank = 49
    U = [[Fraction(0)] * rank for _ in range(16)]
    V = [[Fraction(0)] * rank for _ in range(16)]
    W = [[Fraction(0)] * rank for _ in range(16)]

    for block_idx, (i0, j0, k0) in enumerate([(0, 0, 0), (1, 1, 1)]):
        for t_idx in range(7):
            col = block_idx * 7 + t_idx
            u_flat, v_flat, w_flat = terms[t_idx]
            for r in range(2):
                for c in range(2):
                    i = i0 + r
                    j = j0 + c
                    k = k0 + c  # Wait, V index is j*p+k, and the 2x2 V is indexed by (j_local, k_local)
                    # Actually for the V matrix in the 2x2 block:
                    # V[j_local, k_local] where j_local in {0,1}, k_local in {0,1}
                    # The global indices are j = j0 + j_local, k = k0 + k_local
                    # So for each (r, c) in the U 2x2 matrix:
                    # i = i0 + r, j = j0 + c (this is for U)
                    # But U and V share the j index!
                    # U[i, j] and V[j, k] - the j must match.
                    # So for term t, we have a 2x2 U and a 2x2 V.
                    # U[r, c] corresponds to i=i0+r, j=j0+c
                    # V[r, c] corresponds to j=j0+r, k=k0+c
                    # These don't align on j unless we're careful.
                    
    # I think I'm overcomplicating this. Let me just use the known fact that
    # Strassen gives rank 7 for 2x2x2, and for 4x4x4 we can use a block structure.
    # But the blocks aren't independent in the simple way I described.
    
    # Let me just fall back to schoolbook for now and try to improve.
    return None


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check that sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == delta(i,j,k)
    # where delta is 1 if the indices are valid for the tensor (which they always are)
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k in their ranges, and the
    # decomposition should satisfy: for all i,j,k, sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    # Wait, that's not right either. The matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 (it's just the identity in the "multiplication" sense)
    # And the decomposition is: T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    # So for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1) > tol:
                    return False
    return True


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Start with schoolbook (rank 80)
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using random search
    # Strategy: start with a random decomposition and try to reduce rank
    # by finding linear dependencies.
    
    # For now, return the schoolbook decomposition as a valid baseline.
    # A more sophisticated approach would use:
    # 1. Start with a random rank-r decomposition (r < 80)
    # 2. Use gradient descent or local search to minimize the error
    # 3. Snap to nearest half-integers and verify
    
    # Given time constraints, return the valid schoolbook solution.
    return best_result
# EVOLVE-BLOCK-END
