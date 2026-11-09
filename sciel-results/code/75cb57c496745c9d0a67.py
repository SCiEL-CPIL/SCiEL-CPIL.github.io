import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We split B (4x6) into B1 (4x4) and B2 (4x2)
    # C = A B1 + A B2
    
    # Part 1: A (4x4) * B1 (4x4) using Strassen (Rank 49)
    # Part 2: A (4x4) * B2 (4x2) using Schoolbook (Rank 32)
    
    # Total Rank = 49 + 32 = 81
    
    U_list = []
    V_list = []
    W_list = []
    
    # --- Part 1: Strassen for 4x4x4 ---
    # A is 4x4, B1 is 4x4.
    # We define the 7 Strassen multiplications.
    # Each is a 2x2x2 multiplication.
    # We need to map the 2x2x2 Strassen terms to the global 4x4x4 indices.
    
    # Strassen 2x2x2 terms (rank 7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # The output C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate the U, V, W factors for each of the 7 M_i terms.
    # Each M_i is a 2x2x2 multiplication.
    # The rank of 2x2x2 is 7.
    # We need the specific 7 terms for 2x2x2.
    
    # The 7 terms for 2x2x2 (Strassen):
    # t1 = (a00+a11)(b00+b11)
    # t2 = (a10+a11)b00
    # t3 = a00(b01-b11)
    # t4 = a11(b10-b00)
    # t5 = (a00+a01)b11
    # t6 = (a10-a00)(b00+b01)
    # t7 = (a01-a11)(b10+b11)
    
    # We need to express these as rank-1 tensors.
    # This is a known decomposition.
    # Instead of deriving it, we can use the fact that we can just use the schoolbook for 2x2x2 (rank 8) if we don't have the Strassen 2x2x2 terms handy?
    # No, Strassen 2x2x2 is rank 7.
    # If I use schoolbook for 2x2x2 (rank 8), then 4x4x4 would be 7*8 = 56.
    # 56 + 32 = 88. Still better than 96.
    # But Strassen 4x4x4 is 49.
    
    # Let's use the explicit Strassen 2x2x2 decomposition.
    # The 7 vectors for 2x2x2 are:
    # U, V, W for each term.
    
    # Term 1: (a00+a11)(b00+b11)
    # U: [1, 0, 1, 0] (a00, a01, a10, a11) -> indices 0, 2
    # V: [1, 0, 1, 0] (b00, b01, b10, b11) -> indices 0, 2
    # W: [1, 1, 1, 1] (c00, c01, c10, c11) -> indices 0, 1, 2, 3? No, W is for output.
    # The output of this term contributes to all 4 C entries?
    # No, in Strassen, the terms are combined.
    # The rank-1 decomposition of the Strassen algorithm is not just 7 rank-1 tensors for the whole 2x2x2.
    # The Strassen algorithm computes 7 products of sums.
    # Each product is a scalar multiplication of two 2x2 matrices? No, it's a scalar product of two 2x2 matrices?
    # No, $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$. This is a 2x2 matrix product.
    # So $M_1$ itself has rank 8 (schoolbook) or 7 (Strassen)?
    # No, $M_1$ is a single multiplication of two 2x2 matrices.
    # So to compute $M_1$, we need a decomposition of $2 \times 2 \times 2$.
    # So we need 7 rank-1 terms for $M_1$.
    # And we have 7 such $M_i$'s.
    # So total rank $7 \times 7 = 49$.
    
    # So we need the rank-7 decomposition of $2 \times 2 \times 2$.
    # Let's hardcode the 7 terms for $2 \times 2 \times 2$.
    
    # The 7 terms for 2x2x2 (Strassen):
    # 1. (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11?
    # Actually, the Strassen algorithm computes:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # Each pi is a scalar? No, pi is a 2x2 matrix?
    # No, a00, b00 are scalars in the 2x2x2 context?
    # If we are decomposing $2 \times 2 \times 2$, then a, b, c are 2x2 matrices.
    # The "strassen" terms are products of 2x2 matrices?
    # No, the standard Strassen algorithm for 2x2 matrices uses 7 multiplications of 2x2 matrices?
    # No, it uses 7 multiplications of scalars if we go down to 1x1?
    # No, Strassen's algorithm reduces the number of multiplications from 8 to 7 for 2x2 matrices.
    # The multiplications are of 2x2 matrices? No, they are of sub-matrices.
    # If the input is 2x2, the sub-matrices are 1x1 scalars.
    # So the 7 terms are scalar multiplications.
    # So the rank of $2 \times 2 \times 2$ is 7.
    # The 7 rank-1 terms are the 7 scalar products.
    
    # So for $4 \times 4 \times 4$, we have 7 Strassen terms $M_i$.
    # Each $M_i$ is a $2 \times 2 \times 2$ multiplication.
    # So each $M_i$ requires 7 rank-1 terms.
    # Total 49 rank-1 terms.
    
    # We need to map the scalar indices of the 2x2x2 decomposition to the 4x4x4 indices.
    
    # Let's define the 7 rank-1 terms for 2x2x2.
    # Indices for 2x2 A: 0,1,2,3 (row-major: 00, 01, 10, 11)
    # Indices for 2x2 B: 0,1,2,3
    # Indices for 2x2 C: 0,1,2,3
    
    # Term 1: (a00+a11)(b00+b11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 1, 1, 1] ? No.
    # The product is $a_{00}b_{00} + a_{00}b_{11} + a_{11}b_{00} + a_{11}b_{11}$.
    # This contributes to $c_{00}, c_{01}, c_{10}, c_{11}$?
    # In Strassen, $p1$ is added to $c_{00}$ and $c_{11}$, and subtracted from others?
    # No, $p1$ is a scalar. It is added to $c_{00}$ and $c_{11}$.
    # Wait, $c_{00} = p1 + p4 - p5 + p7$.
    # So $p1$ contributes +1 to $c_{00}$ and +1 to $c_{11}$.
    # So W for term 1 is [1, 0, 0, 1].
    
    # Let's list the 7 terms for 2x2x2 (Strassen):
    # 1. (a00+a11)(b00+b11) -> W: [1, 0, 0, 1]
    # 2. (a10+a11)b00 -> W: [0, 0, 1, 0] (c10)
    # 3. a00(b01-b11) -> W: [0, 1, 0, 0] (c01)
    # 4. a11(b10-b00) -> W: [1, 0, 1, 0] (c00, c10)
    # 5. (a00+a01)b11 -> W: [0, 1, 0, 0] (c01) ? No, c01 = p3+p5.
    # 6. (a10-a00)(b00+b01) -> W: [0, 0, 0, 1] (c11) ? No, c11 = p1-p2+p3+p6.
    # 7. (a01-a11)(b10+b11) -> W: [1, 0, 0, 0] (c00) ? No, c00 = p1+p4-p5+p7.
    
    # This is getting error-prone to derive from memory.
    # I will use the **Schoolbook** for the 2x2x2 parts to be safe, which gives rank 8 for 2x2x2.
    # Then 4x4x4 rank = 7 * 8 = 56.
    # Total rank = 56 + 32 = 88.
    # This is a safe improvement over 96.
    
    # Let's implement the Schoolbook for 2x2x2 within the Strassen structure.
    # Actually, if I use Schoolbook for 2x2x2, I don't need Strassen for 4x4x4.
    # I can just use Schoolbook for 4x4x4 (rank 64) and Schoolbook for 4x4x2 (rank 32).
    # Total 96.
    
    # To get 88, I need to use Strassen for 4x4x4 but Schoolbook for the 2x2x2 sub-problems?
    # No, Strassen for 4x4x4 relies on 7 multiplications of 2x2x2.
    # If I use Schoolbook (rank 8) for each 2x2x2, total rank 56.
    
    # So:
    # 1. Split B into B1 (4x4) and B2 (4x2).
    # 2. Compute A*B1 using Strassen 4x4x4 with Schoolbook 2x2x2 sub-steps. Rank 56.
    # 3. Compute A*B2 using Schoolbook 4x4x2. Rank 32.
    # Total Rank 88.
    
    # This is a valid improvement.
    
    # Implementation details:
    # We need to generate U, V, W for the 56 terms of the Strassen 4x4x4 part.
    # And 32 terms for the Schoolbook 4x4x2 part.
    
    # Let's build the 56 terms.
    # The 7 Strassen multiplications M1..M7.
    # Each Mi is a 2x2x2 multiplication.
    # For each Mi, we have 8 schoolbook terms.
    
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    return {"U": U_list, "V": V_list, "W": W_list}

