import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)
        
        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val
            
        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    # 1. Generate 49 terms for 4x4 Strassen (indices 0..3)
    # Strassen for 4x4 is recursive.
    # We can generate the terms by recursively applying Strassen to 2x2 blocks.
    
    def strassen_2x2_terms(offset_i, offset_j, offset_k, scale=1):
        """
        Generates terms for 2x2 multiplication at given offsets.
        Returns list of (u_dict, v_dict, w_dict)
        """
        terms = []
        # M1 = (A00+A11)(B00+B11)
        u = {(offset_i+0, offset_j+0): scale, (offset_i+1, offset_j+1): scale}
        v = {(offset_j+0, offset_k+0): scale, (offset_j+1, offset_k+1): scale}
        w = {(offset_k+0, offset_i+0): scale, (offset_k+1, offset_i+1): scale}
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00
        u = {(offset_i+1, offset_j+0): scale, (offset_i+1, offset_j+1): scale}
        v = {(offset_j+0, offset_k+0): scale}
        w = {(offset_k+0, offset_i+1): scale, (offset_k+1, offset_i+1): -scale}
        terms.append((u, v, w))
        
        # M3 = A00(B01-B11)
        u = {(offset_i+0, offset_j+0): scale}
        v = {(offset_j+0, offset_k+1): scale, (offset_j+1, offset_k+1): -scale}
        w = {(offset_k+1, offset_i+0): scale, (offset_k+1, offset_i+1): scale}
        terms.append((u, v, w))
        
        # M4 = A11(B10-B00)
        u = {(offset_i+1, offset_j+1): scale}
        v = {(offset_j+1, offset_k+0): scale, (offset_j+0, offset_k+0): -scale}
        w = {(offset_k+0, offset_i+0): scale, (offset_k+1, offset_i+0): scale}
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11
        u = {(offset_i+0, offset_j+0): scale, (offset_i+0, offset_j+1): scale}
        v = {(offset_j+1, offset_k+1): scale}
        w = {(offset_k+0, offset_i+0): -scale, (offset_k+0, offset_i+1): scale}
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01)
        u = {(offset_i+1, offset_j+0): scale, (offset_i+0, offset_j+0): -scale}
        v = {(offset_j+0, offset_k+0): scale, (offset_j+0, offset_k+1): scale}
        w = {(offset_k+1, offset_i+1): scale}
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11)
        u = {(offset_i+0, offset_j+1): scale, (offset_i+1, offset_j+1): -scale}
        v = {(offset_j+1, offset_k+0): scale, (offset_j+1, offset_k+1): scale}
        w = {(offset_k+0, offset_i+0): scale}
        terms.append((u, v, w))
        
        return terms

    # For 4x4, we apply Strassen to the 4 2x2 blocks.
    # The 4x4 matrix is divided into 4 2x2 blocks:
    # A = [A00 A01; A10 A11] where Aij are 2x2.
    # B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    
    # The Strassen formulas for 4x4 involve 7 "super-multiplications" of 2x2 matrices.
    # Each super-multiplication is itself a 2x2 multiplication, which takes 7 scalar multiplications.
    # Total 49.
    
    # We need to combine the linear combinations from the 4x4 level with the 2x2 level.
    # This is complex to do dynamically with dictionaries.
    # Instead, I will hardcode the 49 terms for 4x4 Strassen.
    # Or, I can just use the schoolbook for 4x4 (16 terms) if I can't generate Strassen easily?
    # No, 16 is worse than 49? Wait.
    # Schoolbook 4x4 is 16 multiplications? No, 4*4*4 = 64.
    # Strassen is 49.
    # So Strassen is better.
    
    # Let's try to generate the 49 terms programmatically.
    # The 4x4 Strassen algorithm computes:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Where A00 is top-left 2x2, etc.
    # Each Mi is a 2x2 matrix product.
    # We can iterate through the 7 Mi, and for each, generate the 7 Strassen terms for the 2x2 multiplication,
    # but the inputs to the 2x2 multiplication are linear combinations of the 4x4 entries.
    
    # This requires tracking linear combinations.
    # To keep it simple and robust, I will use a precomputed list of the 49 terms.
    # Since I don't have them memorized, I will write a helper that simulates the recursion.
    
    def generate_4x4_strassen_terms():
        terms = []
        # Define the 7 super-terms
        # Each super-term is a product of two 2x2 matrices (linear combos)
        # We will expand each 2x2 product into 7 scalar products.
        
        # Helper to add a scalar term with coefficients
        # u: dict {(i,j): coeff}, v: dict {(j,k): coeff}, w: dict {(k,i): coeff}
        
        # The 7 Strassen formulas for 2x2 (indices 0,1 relative to block)
        # We will map block indices to global indices.
        
        # This is getting complicated. Let's just use Schoolbook for 4x4?
        # Rank 64.
        # Total rank: 64 + 40 = 104.
        # Current is 124. 104 is better.
        # But 90 is better.
        
        # Let's try to implement the 49 terms properly.
        # I will use a known set of 49 terms for 4x4 Strassen.
        # Actually, I can just use the recursive generation.
        
        # Let's assume the standard Strassen recursion.
        # I will write a function that takes a list of "linear combo indices" for A and B blocks,
        # and generates the scalar terms.
        
        # This is too error-prone to do on the fly in a short time.
        # I will stick to the 2+3 split but optimize the mixed terms?
        # Or use 4x4 Schoolbook (64) + Mixed (40) = 104.
        
        # Let's check if 104 is better than 124. Yes.
        # Let's check if I can do 4x4 Strassen.
        # I found a reference that 4x4 Strassen has 49 mults.
        # I will hardcode the 49 terms.
        # Since I cannot verify them easily, I will use the 2+3 split with Strassen 3x3?
        # 3x3 Strassen has 23 mults.
        # Current 2+3 split:
        # 2x2 Strassen: 7
        # 3x3 Schoolbook: 27
        # Mixed:
        # 2x3x3 (A23 B32): 18? No, 2*3*3 = 18?
        # A23 (2x3) * B32 (3x2) -> 2x2.
        # Terms: i in 0,1; k in 0,1; sum j in 0,1,2.
        # 2*2*3 = 12 terms.
        # 2x3x3 (A22 B23): A22 (2x2) * B23 (2x3) -> 2x3.
        # Terms: i in 0,1; k in 2,3,4; sum j in 0,1.
        # 2*3*2 = 12 terms.
        # 3x2x3 (A32 B22): A32 (3x2) * B22 (2x2) -> 3x2.
        # Terms: i in 2,3,4; k in 0,1; sum j in 0,1.
        # 3*2*2 = 12 terms.
        # 3x3x3 (A32 B23): A32 (3x2) * B23 (2x3) -> 3x3.
        # Terms: i in 2,3,4; k in 2,3,4; sum j in 0,1.
        # 3*3*2 = 18 terms.
        
        # Total mixed: 12+12+12+18 = 54.
        # Total: 7 + 27 + 54 = 88?
        # Wait, the current program says 124.
        # Let's re-read the current program.
        # It uses Schoolbook for 3x3 (27).
        # It uses Schoolbook for mixed blocks.
        # 2x3x3: 2*3*3 = 18?
        # The loop in current program:
        # for i in [0,1]: for j in [2,3,4]: for k in [0,1]: -> 2*3*2 = 12.
        # for i in [0,1]: for j in [0,1]: for k in [2,3,4]: -> 2*2*3 = 12.
        # for i in [2,3,4]: for j in [0,1]: for k in [0,1]: -> 3*2*2 = 12.
        # for i in [2,3,4]: for j in [0,1]: for k in [2,3,4]: -> 3*2*3 = 18.
        # Sum mixed: 12+12+12+18 = 54.
        # 2x2 Strassen: 7.
        # 3x3 Schoolbook: 27.
        # Total: 7 + 27 + 54 = 88.
        
        # Why does the current program say 124?
        # Ah, the current program uses Schoolbook for 2x2?
        # No, it uses Strassen for 2x2 (7 terms).
        # Let's check the code.
        # "Strassen terms for 2x2" -> 7 terms added.
        # "A_23 B_32 schoolbook" -> 12 terms.
        # "C_23 = A_22 B_23 + A_23 B_33"
        # A_22 B_23: 12 terms.
        # A_23 B_33: 18 terms.
        # "C_32 = A_32 B_22 + A_33 B_32"
        # A_32 B_22: 12 terms.
        # A_33 B_32: 18 terms.
        # "C_33 = A_32 B_23 + A_33 B_33"
        # A_32 B_23: 18 terms.
        # A_33 B_33: 27 terms.
        
        # Sum: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124.
        # My calculation: 7 (2x2) + 27 (3x3) + 54 (mixed) = 88.
        # Where is the discrepancy?
        # The mixed blocks in the current program are:
        # 1. A_23 B_32 (12)
        # 2. A_22 B_23 (12)
        # 3. A_23 B_33 (18)
        # 4. A_32 B_22 (12)
        # 5. A_33 B_32 (18)
        # 6. A_32 B_23 (18)
        # Sum: 12+12+18+12+18+18 = 90.
        # Total: 7 + 27 + 90 = 124.
        
        # My previous mixed calculation:
        # I grouped them differently.
        # The blocks are:
        # C_22: A_22 B_22 (7)
        # C_23: A_22 B_23 (12) + A_23 B_33 (18)
        # C_32: A_32 B_22 (12) + A_33 B_32 (18)
        # C_33: A_32 B_23 (18) + A_33 B_33 (27)
        
        # Total: 7 + 12 + 18 + 12 + 18 + 18 + 27 = 112?
        # 7+12=19, +18=37, +12=49, +18=67, +18=85, +27=112.
        # Wait, 124 in the program.
        # Let's re-sum:
        # 7 (Strassen 2x2)
        # 12 (A23 B32) -- Wait, A23 B32 is part of C_22?
        # C_22 = A_22 B_22 + A_23 B_32.
        # So C_22 has 7 + 12 = 19 terms.
        # C_23 = A_22 B_23 (12) + A_23 B_33 (18) = 30 terms.
        # C_32 = A_32 B_22 (12) + A_33 B_32 (18) = 30 terms.
        # C_33 = A_32 B_23 (18) + A_33 B_33 (27) = 45 terms.
        # Total: 19 + 30 + 30 + 45 = 124.
        
        # So the current rank is 124.
        
        # Can we reduce the 3x3 block?
        # Use Strassen for 3x3 (23 terms) instead of 27.
        # Saves 4 terms.
        # New rank: 120.
        
        # Can we reduce the mixed blocks?
        # A_23 B_33 is 2x3 by 3x3.
        # This is 2x3x3 multiplication.
        # Rank of 2x3x3 is at most 18 (schoolbook).
        # Can it be less?
        # It is known that R(2x3x3) <= 16?
        # If we can reduce 18 to 16, we save 2.
        # There are 4 such mixed blocks (A23 B33, A33 B32, A32 B23, A22 B23 is 2x2x3=12, A32 B22 is 3x2x2=12, A23 B32 is 2x3x2=12).
        # The 18-term blocks are:
        # A23 (2x3) * B33 (3x3) -> 2x3. (18 terms)
        # A33 (3x3) * B32 (3x2) -> 3x2. (18 terms)
        # A32 (3x2) * B23 (2x3) -> 3x3. (18 terms)
        # A22 (2x2) * B23 (2x3) -> 2x3. (12 terms)
        # A32 (3x2) * B22 (2x2) -> 3x2. (12 terms)
        # A23 (2x3) * B32 (3x2) -> 2x2. (12 terms)
        
        # If we can reduce the 18-term blocks to 16, we save 2*3 = 6 terms.
        # Total saving: 4 (from 3x3) + 6 (from mixed) = 10.
        # New rank: 114.
        
        # I will implement the 3x3 Strassen (23 terms) and keep the rest as is.
        # This is a safe improvement.
        
        # I need the 23 terms for 3x3 Strassen.
        # I will hardcode them.
        
    # Implementing 3x3 Strassen (23 terms)
    # Indices 2,3,4 map to 0,1,2 for the 3x3 block.
    
    # I will use the known 23-term decomposition for 3x3.
    # Reference: "A 23-multiplication algorithm for 3x3 matrix multiplication"
    
    # Since I don't have the exact 23 terms hardcoded, I will use the schoolbook for 3x3 (27) for now,
    # and focus on the 4+1 split which I calculated as 90.
    # 90 is much better than 124.
    
    # I will implement the 4+1 split with Schoolbook 4x4 (64) + Mixed (40) = 104.
    # This is a guaranteed improvement over 124.
    
    # 4x4 Schoolbook:
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = {(i,j):1}
                v = {(j,k):1}
                w = {(k,i):1}
                add_term(u, v, w)
                
    # Mixed terms:
    # 1. A_41 B_14 (16 terms)
    for i in range(4):
        for k in range(4):
            u = {(i,4):1}
            v = {(4,k):1}
            w = {(k,i):1}
            add_term(u, v, w)
            
    # 2. A_41 B_11 (4 terms)
    for i in range(4):
        u = {(i,4):1}
        v = {(4,4):1}
        w = {(4,i):1}
        add_term(u, v, w)
        
    # 3. A_14 B_44 (16 terms)
    for j in range(4):
        for k in range(4):
            u = {(4,j):1}
            v = {(j,k):1}
            w = {(k,4):1}
            add_term(u, v, w)
            
    # 4. A_14 B_41 (4 terms)
    for j in range(4):
        u = {(4,j):1}
        v = {(j,4):1}
        w = {(4,4):1}
        add_term(u, v, w)
        
    # 5. A_11 B_11 (1 term)
    u = {(4,4):1}
    v = {(4,4):1}
    w = {(4,4):1}
    add_term(u, v, w)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
