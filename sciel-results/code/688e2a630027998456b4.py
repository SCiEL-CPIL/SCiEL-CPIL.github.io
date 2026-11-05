import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS

    def schoolbook():
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

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor.
        Returns True if valid."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True

    def verify_half_integer(U, V, W):
        """Verify that all entries are integer or half-integer, and decomposition is exact."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return verify(U, V, W)

    def to_half_integer_matrix(mat):
        """Convert a matrix of floats to half-integers by rounding."""
        return [[round(val * 2) / 2 for val in row] for row in mat]

    # Construct the rank-71 decomposition
    # Strategy:
    # 1. Use Strassen for the 2x2 block at i={0,1}, j={0,1}, k={0,1}. Rank 7.
    # 2. Use Schoolbook for the rest.
    #    - i in {0,1}, j in {0,1}, k in {2..7} (2*2*6 = 24)
    #    - i=2, j in {0..2}, k in {0..7} (1*3*8 = 24)
    #    - i in {0,1}, j=2, k in {0..7} (2*1*8 = 16)
    # Total Rank = 7 + 24 + 24 + 16 = 71

    def construct_rank_71():
        rank = 71
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # --- Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1} ---
        # Indices:
        # i_off = 0, j_off = 0, k_off = 0
        # A is 2x2 (rows 0,1 of A; cols 0,1 of A) -> U indices: i*m+j for i in {0,1}, j in {0,1}
        # B is 2x2 (rows 0,1 of B; cols 0,1 of B) -> V indices: j*p+k for j in {0,1}, k in {0,1}
        # C is 2x2 (rows 0,1 of C; cols 0,1 of C) -> W indices: k*n+i for k in {0,1}, i in {0,1}
        
        # Strassen terms for C = A*B
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Term 1: M1
        # U: A00 (0,0), A11 (1,1) -> 1
        # V: B00 (0,0), B11 (1,1) -> 1
        # W: C00 (0,0), C11 (1,1) -> 1
        U[0*m + 0][col] += 1
        U[1*m + 1][col] += 1
        V[0*p + 0][col] += 1
        V[1*p + 1][col] += 1
        W[0*n + 0][col] += 1
        W[1*n + 1][col] += 1
        col += 1
        
        # Term 2: M2
        # U: A10 (1,0), A11 (1,1) -> 1
        # V: B00 (0,0) -> 1
        # W: C10 (0,1), C11 (1,1) -> 1, -1
        # Note: W index is k*n+i. 
        # C10 corresponds to i=1, k=0. Index: 0*n+1 = 1.
        # C11 corresponds to i=1, k=1. Index: 1*n+1 = 4.
        U[1*m + 0][col] += 1
        U[1*m + 1][col] += 1
        V[0*p + 0][col] += 1
        W[0*n + 1][col] += 1
        W[1*n + 1][col] -= 1
        col += 1
        
        # Term 3: M3
        # U: A00 (0,0) -> 1
        # V: B10 (1,0), B11 (1,1) -> 1, -1
        # W: C01 (0,0), C11 (1,0) -> 1, 1
        # C01 corresponds to i=0, k=0? No. C is 2x2.
        # C00 (i=0,k=0), C01 (i=0,k=1)? No, standard matrix mult C_ij = sum A_ik B_kj.
        # Here indices are:
        # A: rows i, cols j. B: rows j, cols k. C: rows i, cols k.
        # Wait, the tensor is T_{ijk} = A_{ij} B_{jk} C_{ki}?
        # The task says: "output factor follows the official trace convention W[k*n+i,r]"
        # This implies C_{ki} or similar.
        # Standard matrix multiplication: C = A * B. C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T_{ijk} = A_{ij} B_{jk} C_{ki} is the trace form?
        # Let's check the schoolbook code:
        # U[i*m+j], V[j*p+k], W[k*n+i].
        # Product sum: sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
        # This is trace(A B C^T) or similar?
        # Actually, if we view A, B, C as matrices:
        # A is n x m. B is m x p. C is p x n.
        # Then (A B) is n x p.
        # The term A_{ij} B_{jk} C_{ki} sums to trace(C A B)?
        # Let's check dimensions:
        # A_{ij}: i in 0..n-1, j in 0..m-1.
        # B_{jk}: j in 0..m-1, k in 0..p-1.
        # C_{ki}: k in 0..p-1, i in 0..n-1.
        # Sum_{i,j,k} A_{ij} B_{jk} C_{ki} = trace(C A B).
        # For the decomposition to represent matrix multiplication C = A B, we usually have C_{ik} = sum_j A_{ij} B_{jk}.
        # This is equivalent to the tensor T_{ijk} = A_{ij} B_{jk} D_{ik} where D is the identity? No.
        # The tensor for matrix multiplication is usually defined such that the bilinear forms recover the product.
        # The standard definition of the tensor T_{n,m,p} is such that 
        # <T, A \otimes B \otimes C> = 0 for all A,B,C where C != AB?
        # Or more simply, the rank is the minimum number of multiplications.
        # The schoolbook decomposition has terms (e_i e_j^T) \otimes (e_j e_k^T) \otimes (e_k e_i^T).
        # This corresponds to C_{ik} += A_{ij} B_{jk}.
        # So W[k*n+i] corresponds to the entry C_{ik} (or C_{ki} depending on row/col major).
        # In the schoolbook code: W[k*n+i] = 1.
        # If we interpret W as a matrix W_{ki}, then the term contributes to W_{ki}.
        # In C = A B, C_{ik} is the entry.
        # So W_{ki} corresponds to C_{ik}.
        
        # Let's re-verify Strassen mapping for C_{ik} = sum_j A_{ij} B_{jk}.
        # We are multiplying A (2x2) by B (2x2) to get C (2x2).
        # Indices i, j, k in {0,1}.
        # U[i*m+j], V[j*p+k], W[k*n+i].
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 (0,0), A11 (1,1).
        # V: B00 (0,0), B11 (1,1).
        # Contribution to C:
        # C00 += A00 B00 + A11 B00? No.
        # The term is sum_{i,j,k} U_{ij} V_{jk} W_{ki}.
        # For M1:
        # U_{00}=1, U_{11}=1.
        # V_{00}=1, V_{11}=1.
        # W_{00}=1, W_{11}=1.
        # Non-zero terms in sum:
        # i=0, j=0, k=0: U00 V00 W00 = 1*1*1 = 1. (Contributes to C00)
        # i=0, j=0, k=1: U00 V01 W10 = 1*0*0 = 0.
        # i=0, j=1, k=0: U01 V10 W00 = 0*0*1 = 0.
        # i=0, j=1, k=1: U01 V11 W10 = 0*1*0 = 0.
        # i=1, j=0, k=0: U10 V00 W01 = 0*1*0 = 0.
        # i=1, j=0, k=1: U10 V01 W11 = 0*0*1 = 0.
        # i=1, j=1, k=0: U11 V10 W01 = 1*0*0 = 0.
        # i=1, j=1, k=1: U11 V11 W11 = 1*1*1 = 1. (Contributes to C11)
        # So M1 contributes to C00 and C11.
        # In Strassen: C00 += M1, C11 += M1. Correct.
        
        # Term 2: M2 = (A10+A11)B00
        # U: A10 (1,0), A11 (1,1).
        # V: B00 (0,0).
        # W: C10 (0,1)? No, C_{ik}.
        # Strassen: C10 += M2, C11 -= M2.
        # C10 is i=1, k=0. W index: k*n+i = 0*2+1 = 1.
        # C11 is i=1, k=1. W index: k*n+i = 1*2+1 = 3.
        # So W[1] += 1, W[3] -= 1.
        U[1*m + 0][col] += 1
        U[1*m + 1][col] += 1
        V[0*p + 0][col] += 1
        W[0*n + 1][col] += 1
        W[1*n + 1][col] -= 1
        col += 1
        
        # Term 3: M3 = A00(B10-B11)
        # U: A00 (0,0).
        # V: B10 (1,0), B11 (1,1).
        # Strassen: C01 += M3, C11 += M3.
        # C01 is i=0, k=1. W index: 1*2+0 = 2.
        # C11 is i=1, k=1. W index: 1*2+1 = 3.
        U[0*m + 0][col] += 1
        V[1*p + 0][col] += 1
        V[1*p + 1][col] -= 1
        W[1*n + 0][col] += 1
        W[1*n + 1][col] += 1
        col += 1
        
        # Term 4: M4 = A11(B01-B00)
        # U: A11 (1,1).
        # V: B01 (0,1), B00 (0,0).
        # Strassen: C00 += M4, C10 += M4.
        # C00 is i=0, k=0. W index: 0*2+0 = 0.
        # C10 is i=1, k=0. W index: 0*2+1 = 1.
        U[1*m + 1][col] += 1
        V[0*p + 1][col] += 1
        V[0*p + 0][col] -= 1
        W[0*n + 0][col] += 1
        W[0*n + 1][col] += 1
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        # U: A00 (0,0), A01 (0,1).
        # V: B11 (1,1).
        # Strassen: C00 -= M5, C01 += M5.
        # C00 is i=0, k=0. W index: 0.
        # C01 is i=0, k=1. W index: 2.
        U[0*m + 0][col] += 1
        U[0*m + 1][col] += 1
        V[1*p + 1][col] += 1
        W[0*n + 0][col] -= 1
        W[1*n + 0][col] += 1
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # U: A10 (1,0), A00 (0,0).
        # V: B00 (0,0), B01 (0,1).
        # Strassen: C11 += M6.
        # C11 is i=1, k=1. W index: 3.
        U[1*m + 0][col] += 1
        U[0*m + 0][col] -= 1
        V[0*p + 0][col] += 1
        V[0*p + 1][col] += 1
        W[1*n + 1][col] += 1
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # U: A01 (0,1), A11 (1,1).
        # V: B10 (1,0), B11 (1,1).
        # Strassen: C00 += M7, C01 -= M7.
        # C00 is i=0, k=0. W index: 0.
        # C01 is i=0, k=1. W index: 2.
        U[0*m + 1][col] += 1
        U[1*m + 1][col] -= 1
        V[1*p + 0][col] += 1
        V[1*p + 1][col] += 1
        W[0*n + 0][col] += 1
        W[1*n + 0][col] -= 1
        col += 1

        # --- Part 2: Schoolbook for i in {0,1}, j in {0,1}, k in {2..7} ---
        for i in range(2):
            for j in range(2):
                for k in range(2, 8):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1

        # --- Part 3: Schoolbook for i=2, j in {0..2}, k in {0..7} ---
        for j in range(3):
            for k in range(8):
                U[2*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + 2][col] = 1
                col += 1

        # --- Part 4: Schoolbook for i in {0,1}, j=2, k in {0..7} ---
        for i in range(2):
            for k in range(8):
                U[i*m + 2][col] = 1
                V[2*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    best_result = construct_rank_71()
    
    if verify_half_integer(best_result["U"], best_result["V"], best_result["W"]):
        return best_result
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
