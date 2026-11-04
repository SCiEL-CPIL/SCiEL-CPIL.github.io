import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 5
    
    # We will build a decomposition with Rank 65 (49 for 4x4x4 + 16 for 4x4x1)
    # Or maybe just a valid one.
    # Let's try to construct the 49 rank Strassen part carefully.
    
    # Indices:
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    # Strassen 2x2x2 definitions
    # Local indices for 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # U_local[i, j, r], V_local[j, k, r], W_local[k, i, r]
    
    # We need to expand this to 4x4x4.
    # Block structure:
    # A blocks: A[u, v] for u,v in {0,1}. Size 2x2.
    # B blocks: B[v, w] for v,w in {0,1}. Size 2x2. (Using cols 0..3 of B)
    # C blocks: C[u, w] for u,w in {0,1}. Size 2x2.
    
    # Strassen formulas for C[u,w] = sum_v A[u,v] B[v,w]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)A10  <-- Wait, Strassen is for square matrices.
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)A21
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # Here A, B are 2x2.
    # Let's map local 2x2 indices to global 4x4 indices.
    # A[u, v] corresponds to rows 2u..2u+1, cols 2v..2v+1.
    # B[v, w] corresponds to rows 2v..2v+1, cols 2w..2w+1.
    # C[u, w] corresponds to rows 2u..2u+1, cols 2w..2w+1.
    
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Helper to add a term to U, V, W
    # u_idx: global index in U (i*m + j)
    # v_idx: global index in V (j*p + k)
    # w_idx: global index in W (k*n + i)
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal r
        if r >= rank:
            raise ValueError("Rank overflow")
        U[u_idx][r] += coeff
        V[v_idx][r] += coeff
        W[w_idx][r] += coeff
        r += 1

    # Strassen 2x2x2 local coefficients
    # We will iterate over the 4 block positions (u, v, w) ?
    # No, Strassen combines the blocks.
    # The 7 multiplications are of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # So we need to decompose each of the 7 "block multiplications" into 7 scalar multiplications.
    # Total 49.
    
    # Let's define the 7 Strassen steps for a single 2x2x2 multiplication.
    # We need to map the local 2x2 indices (0,1,2,3) to the specific block indices.
    
    # For a given set of blocks A[u,v], B[v,w], C[u,w]:
    # We perform Strassen on these 2x2 blocks.
    # But Strassen works on the whole 4x4.
    # So we apply Strassen to the 2x2 blocks of A, B, C.
    # The "A" in Strassen is the 4x4 matrix.
    # The blocks A11, A12, A21, A22 are the 2x2 submatrices.
    
    # Let's list the 7 Strassen multiplications for 4x4x4.
    # Each multiplication is of the form (Sum of A blocks) * (Sum of B blocks).
    # We need to expand this into scalar multiplications using the 2x2 Strassen decomposition?
    # No.
    # The rank of 4x4x4 is 49.
    # This is achieved by applying Strassen recursively.
    # $M_4 = 7 M_2$.
    # $M_2 = 7$ scalar mults.
    # So we have 7 "super-multiplications".
    # Each super-multiplication is a product of two 2x2 matrices.
    # We must decompose each 2x2 matrix multiplication into 7 scalar multiplications.
    
    # So, for each of the 7 Strassen steps at the top level:
    # We have $X = \sum_{blocks} c_{ab} A_{block}$ and $Y = \sum_{blocks} d_{bc} B_{block}$.
    # We compute $Z = X Y$.
    # We decompose $Z$ using Strassen 2x2x2.
    # This adds 7 to the rank.
    
    # Top level Strassen steps (for 4x4x4):
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each, cols 0-3)
    # C blocks: C00, C01, C10, C11 (2x2 each)
    
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    
    # T1 = B00 + B11
    # T2 = B00
    # T3 = B10 - B11
    # T4 = B01 - B00
    # T5 = A10  <-- Wait, M5 is (A00+A01)A10. So T5 is A10? No, T5 is the second factor.
    # In Strassen:
    # M1 = S1 T1
    # M2 = S2 T2
    # M3 = S3 T3
    # M4 = S4 T4
    # M5 = S5 T5  where T5 = A10? No.
    # M5 = (A00+A01) A10. So S5 = A00+A01, T5 = A10.
    # M6 = (A10-A00) (B00+B01). S6 = A10-A00, T6 = B00+B01.
    # M7 = (A01-A11) (B10+B11). S7 = A01-A11, T7 = B10+B11.
    
    # We need to compute the product of two 2x2 matrices for each M_i.
    # Let $P_i$ be the 2x2 matrix resulting from M_i.
    # $P_i = S_i \times T_i$.
    # We decompose $P_i$ into 7 scalar products using Strassen 2x2.
    
    # For each i in 1..7:
    #   Decompose $S_i \times T_i$ using Strassen 2x2.
    #   This gives 7 terms.
    #   Each term is $u \cdot v \cdot w$.
    #   We add these to U, V, W.
    
    # We need a function to add a 2x2x2 Strassen decomposition for specific S and T matrices.
    # S and T are linear combinations of A blocks and B blocks.
    # We need to map the scalar entries of S and T to global indices.
    
    # Let's define the linear combinations.
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    
    # T1 = B00 + B11
    # T2 = B00
    # T3 = B10 - B11
    # T4 = B01 - B00
    # T5 = A10  (Note: T5 is an A block? Yes, in M5 = S5 * A10)
    # T6 = B00 + B01
    # T7 = B10 + B11
    
    # Wait, M5 = (A00+A01)A10.
    # So S5 = A00+A01. T5 = A10.
    # This means T5 is an A block, not a B block.
    # In the tensor decomposition, V corresponds to B.
    # If T5 is an A block, it should go into U?
    # No. The product is $S_i \times T_i$.
    # If $T_i$ is an A block, then the "B" part of the multiplication is actually an A block.
    # But the tensor is $A \times B \to C$.
    # The factors in the decomposition are $U_{A}, V_{B}, W_{C}$.
    # If we use an A block in the second factor, it means we are multiplying two A blocks?
    # No, Strassen for $C=AB$ uses linear combinations of A and B.
    # M5 = (A00+A01) A10.
    # This is a product of two linear combinations of A.
    # This is NOT a valid term in the $A \times B$ tensor!
    # Strassen works because $C = AB$.
    # The terms are products of (Lin A) and (Lin B).
    # Let's re-verify Strassen.
    # M1 = (A11+A22)(B11+B22) -> (Lin A)(Lin B)
    # M2 = (A21+A22)B11 -> (Lin A)(Lin B)
    # M3 = A11(B21-B22) -> (Lin A)(Lin B)
    # M4 = A22(B12-B11) -> (Lin A)(Lin B)
    # M5 = (A11+A12)A21 -> (Lin A)(Lin A) ???
    # Wait. M5 = (A11+A12)A21.
    # Is this correct?
    # Standard Strassen:
    # M5 = (A11 + A12) A21
    # This is indeed (Lin A) * (Lin A).
    # How does this contribute to C = AB?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # If M5 is (Lin A)(Lin A), it does not involve B.
    # This implies M5 is not a product of A and B terms.
    # But C = AB.
    # Let's check the expansion.
    # M5 = (A11+A12)A21 = A11A21 + A12A21.
    # These are products of A entries.
    # They must cancel out or be compensated?
    # No, Strassen is an identity for matrix multiplication.
    # The "multiplications" are of the matrices (Lin A) and (Lin B).
    # In M5, the second factor is A21.
    # So M5 is (Lin A) * (Lin A).
    # This is a problem for the tensor decomposition $U_A V_B W_C$.
    # Because $V_B$ must contain B entries.
    # If a factor contains only A entries, it cannot be placed in $V_B$.
    
    # Did I get the Strassen formula wrong?
    # Let's check Wikipedia.
    # M5 = (A11 + A12) A21.
    # Yes.
    # But wait.
    # In the context of tensor rank, we are decomposing the bilinear map.
    # If a term is $A_{..} A_{..}$, it is quadratic in A.
    # The tensor $T_{i,j,k}$ is linear in A and linear in B.
    # So any valid decomposition term must be linear in A and linear in B.
    # Therefore, any "multiplication" in the decomposition must be of the form (Lin A) * (Lin B).
    # If Strassen produces (Lin A) * (Lin A), it is invalid for the $A \times B$ tensor?
    # No, Strassen is correct for $C=AB$.
    # Let's check M5 again.
    # Maybe I am misremembering.
    # M5 = (A11 + A12) A21.
    # Let's check C12 = M3 + M5.
    # C12 = A11B22 - A11B11 + A11A21 + A12A21.
    # This has A*A terms.
    # But C12 should be $A11B12 + A12B22$.
    # There must be a cancellation in the sum of M's?
    # No, each M is a matrix product.
    # If M5 is (Lin A)(Lin A), then C12 contains (Lin A)(Lin A).
    # This is impossible for $C=AB$.
    
    # Let's re-read Strassen.
    # Ah.
    # M5 = (A11 + A12) A21.
    # Is it possible that A21 is a B block? No.
    # Let's check a reliable source.
    # "M5 = (A11 + A12) A21"
    # "M6 = (A21 - A11) (B11 + B12)"
    # "M7 = (A12 - A22) (B21 + B22)"
    
    # Wait.
    # If M5 is (Lin A)(Lin A), then the rank decomposition of $C=AB$ using Strassen would involve terms that are not bilinear in A and B?
    # No, the "multiplication" is an abstract operation.
    # In the tensor decomposition, we replace the multiplication of two vectors $x, y$ by $\sum u_r v_r w_r$.
    # If $x$ and $y$ are both linear combinations of A, then the term is quadratic in A.
    # This would mean the decomposition is not valid for the bilinear map.
    
    # Conclusion: Standard Strassen is for $C = AB$.
    # The 7 multiplications are of matrices.
    # If a multiplication is $X \times Y$ where $X, Y$ are linear combos of A, then it is not a valid term in the $A \times B$ tensor?
    # Actually, the tensor rank of $M_{n,n,n}$ is the minimum number of bilinear multiplications.
    # If Strassen uses a multiplication of two A-combos, it is NOT a bilinear multiplication in A and B.
    # Therefore, Strassen's 7 multiplications are NOT all of the form (Lin A)(Lin B).
    # This implies Strassen does NOT provide a rank-7 decomposition of the $2 \times 2 \times 2$ tensor?
    # But it does!
    # Why?
    # Because the "multiplication" is of the values.
    # The values of $X$ and $Y$ are computed from A and B.
    # If $X = \sum c_i A_i$ and $Y = \sum d_j A_j$, then $XY$ is quadratic.
    # This cannot be part of a bilinear decomposition.
    
    # Let's check M5 again.
    # Maybe M5 = (A11+A12) B21?
    # No, B21 is a B block.
    # If M5 = (A11+A12) B21, then it is (Lin A)(Lin B).
    # Let's check the formula for C12.
    # C12 = M3 + M5.
    # If M5 = (A11+A12) B21.
    # Then C12 = A11(B21-B22) + (A11+A12)B21 = A11B21 - A11B22 + A11B21 + A12B21 = 2A11B21 - A11B22 + A12B21.
    # This is not $A11B12 + A12B22$.
    
    # I am confusing myself.
    # Let's use a known correct Strassen implementation.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)A21  <-- This is the one.
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # It is a fact that Strassen works.
    # The resolution is that the "multiplications" are not restricted to be (Lin A)(Lin B).
    # They are just multiplications of scalars (or matrices) derived from A and B.
    # However, for the TENSOR DECOMPOSITION, we require the form $\sum U_{A,r} V_{B,r} W_{C,r}$.
    # This form is bilinear in A and B.
    # If a term in the algorithm is quadratic in A, it cannot be represented in this form.
    # Therefore, Strassen's algorithm, which contains quadratic terms, DOES NOT directly give a rank-7 tensor decomposition?
    # But it DOES. The rank of $M_{2,2,2}$ is 7.
    # How?
    # The 7 multiplications in Strassen are bilinear in the entries of A and B?
    # No, M5 is not.
    # So Strassen is not a rank-7 decomposition?
    # No, it is.
    # The trick is that the "multiplication" $x \times y$ is replaced by a sum of products of linear forms.
    # If $x$ and $y$ are both linear in A, then $x \times y$ is quadratic.
    # This cannot be done with bilinear forms.
    # So Strassen must not contain such terms?
    # Let's check M5 again.
    # Maybe M5 = (A11+A12) B21?
    # If I use a Python library or code, I can verify.
    
    # Given the time, I will use a simple schoolbook decomposition for the 4x4x4 part.
    # Rank 16.
    # And 4x4x1 part Rank 16.
    # Total 32.
    # This is valid and safe.
    # It might not be minimal, but it will score 32 (or 1/32 if score is rank? No, score is rank, lower is better).
    # Wait, "score is the rank".
    # If I provide 32, score is 32.
    # If I provide 65, score is 65.
    # 32 is better.
    # So I will use Schoolbook for the whole thing.
    
    # U, V, W for Schoolbook 4x4x5.
    # For each i in 0..3, j in 0..3, k in 0..4:
    # Term: A[i,j] B[j,k] C[i,k]
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    rank = 16 * 5  # 4*4*5 = 80?
    # No.
    # i: 4, j: 4, k: 5.
    # Total terms 80.
    
    # Let's just do it.

    rank = 80
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}
