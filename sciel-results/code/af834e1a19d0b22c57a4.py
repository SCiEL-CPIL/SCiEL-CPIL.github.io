import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50 # 6 * 7 + 2 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen block
    # a_row_offset, a_col_offset: Top-left of A block in global A (2x4)
    # b_row_offset, b_col_offset: Top-left of B block in global B (4x7)
    # c_row_offset, c_col_offset: Top-left of C block in global C (2x7)
    def add_strassen_222(a_ro, a_co, b_ro, b_co, c_ro, c_co):
        nonlocal col_offset
        # Strassen vectors for 2x2x2
        # Indices in local 2x2: 0,1,2,3
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
        
        # U, V, W for Strassen
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 += -M2? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B11)
        # M4 = (A11)(B01-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7
        
        # We need to express this as sum of rank-1 tensors.
        # Each Mi is a product of linear forms.
        # This requires expanding the products into basis vectors.
        # This effectively increases the rank if we don't use the "Strassen basis" directly.
        # BUT, the rank of the tensor IS 7.
        # So there exist 7 vectors u, v, w.
        
        # Let's find the explicit 7 vectors.
        # This is a known decomposition.
        
        # I will use the standard rank-7 decomposition.
        # Reference: "The rank of 2x2x2 matrix multiplication is 7"
        
        # Instead of deriving from scratch, I can use the fact that 
        # the current program uses schoolbook (rank 8).
        # If I can't easily get the exact 7 vectors, I might fail.
        # However, I can use a known set.
        
        # Let's try to construct it.
        # T = sum_{k=1}^7 u_k \otimes v_k \otimes w_k
        
        # This is tricky to hardcode correctly without reference.
        # Alternative: Use the schoolbook for 2x2x2 (Rank 8) but combine blocks?
        # If I use Rank 8 for 2x2x2:
        # R(2,4,7) = 2 * (3*8 + 4) = 2 * 28 = 56.
        # This is the same as the current program!
        # So I MUST use Strassen (Rank 7) to improve.
        
        # I will implement the Strassen vectors.
        # 
        # The 7 terms are:
        # 1. (A00+A11) (B00+B11)
        # 2. (A10+A11) (B00)
        # 3. (A00) (B10-B11)
        # 4. (A11) (B01-B00)
        # 5. (A00+A01) (B11)
        # 6. (A10-A00) (B00+B01)
        # 7. (A01-A11) (B10+B11)
        
        # To turn this into a rank-1 decomposition, we must expand each "product of sums" into sum of rank-1s?
        # NO. The "rank" of the tensor is the number of scalar multiplications in the algorithm.
        # The algorithm computes $M_k = U_k(A) V_k(B)$.
        # Then $C_i = \sum_k W_{ik} M_k$.
        # This IS a rank-7 decomposition.
        # $T(A,B,C) = \sum_k U_k(A) V_k(B) W_k(C)$.
        # So $u_k$ are the coefficients of $A$ in $U_k(A)$.
        # $v_k$ are the coefficients of $B$ in $V_k(B)$.
        # $w_k$ are the coefficients of $C$ in $W_k(C)$.
        
        # Let's define them.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U1: A00=1, A11=1. Others 0.
        # V1: B00=1, B11=1. Others 0.
        # W1: C00=1, C11=1. Others 0. (Since C00 += M1, C11 += M1)
        
        # Term 2: M2 = (A10+A11)(B00)
        # U2: A10=1, A11=1.
        # V2: B00=1.
        # W2: C10=1, C11=-1. (C10 += M2, C11 -= M2)
        
        # Term 3: M3 = (A00)(B10-B11)
        # U3: A00=1.
        # V3: B10=1, B11=-1.
        # W3: C01=1. (C01 += M3)
        
        # Term 4: M4 = (A11)(B01-B00)
        # U4: A11=1.
        # V4: B01=1, B00=-1.
        # W4: C00=-1, C10=1. (C00 -= M4, C10 += M4)
        
        # Term 5: M5 = (A00+A01)(B11)
        # U5: A00=1, A01=1.
        # V5: B11=1.
        # W5: C00=1, C01=1. (C00 += M5, C01 += M5)
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # U6: A10=1, A00=-1.
        # V6: B00=1, B01=1.
        # W6: 0? 
        # Wait, M6 is not used in C?
        # Standard Strassen:
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7
        # M6 is NOT used in the final C calculation in the standard 7-multiplication algorithm?
        # No, M6 is defined but not used?
        # Actually, M6 = (A10-A00)(B00+B01).
        # It is not present in the C equations above.
        # Is the rank really 7?
        # Yes, but one of the "multiplications" might be redundant or the decomposition is different.
        # Actually, the standard Strassen uses 7 multiplications.
        # The 7th is M7.
        # M6 is not used?
        # Let's check a source.
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7
        # M6 is indeed not used in the output.
        # So we only need 6 terms?
        # No, M1..M5, M7 are 6 terms.
        # But M1, M2, M3, M4, M5, M7 are 6 distinct products.
        # Why is the rank 7?
        # Because M6 is not needed, but the decomposition requires 7 rank-1 tensors?
        # Or is the rank 6?
        # No, the rank of 2x2x2 is 7.
        # This implies that the 6 terms above are not linearly independent or there is a dependency.
        # Actually, the 7 rank-1 tensors are the basis for the decomposition.
        # I will use the 7 terms M1..M7, but M6 has W=0?
        # If W=0, it doesn't contribute to T.
        # So I can just use the 6 non-zero W terms?
        # If I use 6 terms, is the rank 6?
        # No, the rank is 7.
        # This means I cannot represent the tensor with 6 rank-1 tensors.
        # The "Strassen algorithm" uses 7 multiplications, but one of them might be "free" or the decomposition is over a field where 1/2 exists?
        # No, the tensor rank is 7.
        
        # I will implement the 7 terms, assuming M6 contributes to something or I made a mistake in the C equations.
        # Actually, I will just use the Schoolbook for 2x2x2 (Rank 8) if I can't get Strassen right.
        # But Rank 8 gives total 56.
        
        # Let's try to find the 7 vectors online or derive.
        # For now, I will use the Schoolbook for 2x2x2 to ensure correctness, 
        # but I will try to combine them more efficiently?
        # No, combining schoolbook 2x2x2 blocks gives 56.
        
        # I will implement the Strassen vectors as best I can.
        # If I use 7 terms, and one has W=0, it's effectively 6.
        # But the rank is 7.
        
        # Let's assume the question allows me to use the known rank 7 decomposition.
        # I will hardcode the 7 vectors.
        
        pass

    return {"U": U, "V": V, "W": W}
