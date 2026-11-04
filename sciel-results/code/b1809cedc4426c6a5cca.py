import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the factors U, V, W as lists of lists.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We define a helper to add a sub-decomposition
    # sub_U, sub_V, sub_W are local factors for a sub-tensor of size (sn, sm, sp)
    # offset_i, offset_j, offset_k are the starting indices in the global n, m, p
    # We append the columns to the global U, V, W
    
    global_U = []
    global_V = []
    global_W = []
    
    # Helper to initialize empty lists if needed, but we are appending columns.
    # It's easier to build a list of tuples (i, j, k, u_val, v_val, w_val) for each rank term
    # and then convert to matrices.
    
    terms = []
    
    def add_subtensor(sn, sm, sp, off_i, off_j, off_k, factors):
        # factors is a list of (u_vec, v_vec, w_vec) where u_vec is length sn*sm, etc.
        # Actually, it's easier to pass the local U, V, W matrices.
        pass

    # Let's just generate the terms directly.
    # A term is defined by (u_indices, v_indices, w_indices)
    # where u_indices is a dict mapping local (i,j) to value
    # This is getting complicated to manage offsets.
    
    # Alternative: Build the full U, V, W matrices of size (n*m, total_rank), etc.
    # We don't know total_rank in advance.
    
    # Let's use a list of "components".
    # Each component is (U_comp, V_comp, W_comp)
    # U_comp is a sparse representation or a small matrix.
    
    components = []
    
    def add_schoolbook(sn, sm, sp, off_i, off_j, off_k):
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    # Local indices
                    li, lj, lk = i, j, k
                    # Global indices
                    gi = off_i + i
                    gj = off_j + j
                    gk = off_k + k
                    
                    u = {}
                    v = {}
                    w = {}
                    u[(gi, gj)] = 1
                    v[(gj, gk)] = 1
                    w[(gk, gi)] = 1
                    components.append((u, v, w))

    def add_strassen(sn, sm, sp, off_i, off_j, off_k):
        # Only valid for 2x2x2
        if sn != 2 or sm != 2 or sp != 2:
            return add_schoolbook(sn, sm, sp, off_i, off_j, off_k)
        
        # Strassen factors for 2x2x2
        # Local indices 0,1
        # a00, a01, a10, a11
        # b00, b01, b10, b11
        # c00, c01, c10, c11
        
        # 1. a00 * (b00 + b01)
        u = {(0,0): 1}
        v = {(0,0): 1, (0,1): 1}
        w = {(0,0): 1, (1,0): 1, (0,1): -1, (1,1): -1} # c00 +=, c01 +=, c10 -=, c11 -= ?
        # Strassen:
        # M1 = A00 * (B00 + B01)
        # M2 = A10 * B00
        # M3 = A00 * B11
        # M4 = A11 * (B10 - B11)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 + A11) * (B10 - B11)
        
        # C00 = M5 + M4 - M3 - M7
        # C01 = M1 + M3
        # C10 = M2 + M4
        # C11 = M5 - M1 + M2 + M7
        
        # We need to map these to U, V, W.
        # U has entries for A, V for B, W for C.
        
        # Term 1: M1 = A00 * (B00 + B01)
        # U: A00 -> 1
        # V: B00 -> 1, B01 -> 1
        # W: C01 -> 1 (from M1)
        # Wait, W is for the output C.
        # The decomposition is sum U V W = T.
        # T_{i,j,k} is 1 if i,j,k correspond to a product term?
        # No, T is the matrix multiplication tensor.
        # The bilinear form is sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} C_{ki} = tr(ABC).
        # So W_{k,i} should be the coefficient of C_{ki} in the term.
        
        # Let's define the terms for Strassen:
        # 1. U: {00:1}, V: {00:1, 01:1}, W: {10:1, 11:1} ? No.
        # M1 contributes to C01 and C11?
        # C01 = M1 + M3. So M1 contributes 1 to C01.
        # C11 = M5 - M1 + M2 + M7. So M1 contributes -1 to C11.
        # So W: {10:1, 11:-1} ? Indices for W are (k,i). C01 is k=0, i=1?
        # C_{ik}. C01 is i=0, k=1.
        # W index is k*n + i.
        # For 2x2, n=2.
        # C00: k=0, i=0 -> idx 0.
        # C01: k=1, i=0 -> idx 1.
        # C10: k=0, i=1 -> idx 2.
        # C11: k=1, i=1 -> idx 3.
        
        # M1 = A00 (B00 + B01).
        # A00 is i=0, j=0. U index 0.
        # B00 is j=0, k=0. V index 0.
        # B01 is j=0, k=1. V index 1.
        # Contributes to C01 (i=0, k=1) with +1. W index 1.
        # Contributes to C11 (i=1, k=1) with -1. W index 3.
        
        # So:
        # 1. U: {0:1}, V: {0:1, 1:1}, W: {1:1, 3:-1}
        
        # 2. M2 = A10 B00
        # A10: i=1, j=0 -> U idx 2.
        # B00: j=0, k=0 -> V idx 0.
        # C10: i=1, k=0 -> W idx 2.
        # C11: i=1, k=1 -> W idx 3.
        # C10 = M2 + M4. M2 contributes +1 to C10.
        # C11 = M5 - M1 + M2 + M7. M2 contributes +1 to C11.
        # So: U: {2:1}, V: {0:1}, W: {2:1, 3:1}
        
        # 3. M3 = A00 B11
        # A00: U idx 0.
        # B11: j=1, k=1 -> V idx 3.
        # C01: i=0, k=1 -> W idx 1.
        # C00: i=0, k=0 -> W idx 0.
        # C01 = M1 + M3. M3 contributes +1 to C01.
        # C00 = M5 + M4 - M3 - M7. M3 contributes -1 to C00.
        # So: U: {0:1}, V: {3:1}, W: {1:1, 0:-1}
        
        # 4. M4 = A11 (B10 - B11)
        # A11: i=1, j=1 -> U idx 3.
        # B10: j=1, k=0 -> V idx 2.
        # B11: j=1, k=1 -> V idx 3.
        # C10: i=1, k=0 -> W idx 2.
        # C11: i=1, k=1 -> W idx 3.
        # C00: i=0, k=0 -> W idx 0.
        # C10 = M2 + M4. M4 contributes +1 to C10.
        # C11 = M5 - M1 + M2 + M7. M4 does not appear?
        # Wait, C11 formula: M5 - M1 + M2 + M7.
        # C00 = M5 + M4 - M3 - M7. M4 contributes +1 to C00.
        # So: U: {3:1}, V: {2:1, 3:-1}, W: {2:1, 0:1}
        
        # 5. M5 = (A00 + A11) (B00 + B11)
        # A00: U idx 0. A11: U idx 3.
        # B00: V idx 0. B11: V idx 3.
        # C00: W idx 0.
        # C11: W idx 3.
        # C00 = M5 + ... M5 contributes +1.
        # C11 = M5 - ... M5 contributes +1.
        # So: U: {0:1, 3:1}, V: {0:1, 3:1}, W: {0:1, 3:1}
        
        # 6. M6 = (A10 - A00) (B00 + B01)
        # A10: U idx 2. A00: U idx 0.
        # B00: V idx 0. B01: V idx 1.
        # C01: W idx 1.
        # C10: W idx 2.
        # C01 = M1 + M3. M6 not in C01?
        # C10 = M2 + M4. M6 not in C10?
        # Wait, Strassen formulas:
        # C00 = M5 + M4 - M3 - M7
        # C01 = M1 + M3
        # C10 = M2 + M4
        # C11 = M5 - M1 + M2 + M7
        # Where is M6?
        # M6 is not used in the standard Strassen output?
        # Ah, M6 is used in some variants or I missed it.
        # Actually, M6 = (A10-A00)(B00+B01) is not in the standard 7-term Strassen for 2x2.
        # The 7 terms are M1..M7.
        # M1 = A00(B00+B01)
        # M2 = A10 B00
        # M3 = A00 B11
        # M4 = A11(B10-B11)
        # M5 = (A00+A11)(B00+B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01+A11)(B10-B11)
        
        # Let's re-verify C formulas with M6.
        # C00 = M5 + M4 - M3 - M7
        # C01 = M1 + M3
        # C10 = M2 + M4
        # C11 = M5 - M1 + M2 + M7
        # M6 is not used?
        # Actually, M6 is used if we want to compute something else?
        # No, Strassen uses 7 multiplications.
        # Maybe M6 is not needed for C?
        # Let's check.
        # If M6 is not used, then rank is 7.
        # But M6 is defined.
        # Maybe I have the wrong formulas.
        # Correct Strassen:
        # M1 = A00 * (B00 + B01)
        # M2 = (A10 + A11) * B00
        # M3 = (A00 + A01) * B11
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M5 + M4 - M3 + M7
        # C01 = M1 + M3
        # C10 = M2 + M4
        # C11 = M5 + M1 - M2 + M6
        
        # Let's use this set.
        
        # 1. M1: U{0:1}, V{0:1, 1:1}, W{1:1} (C01)
        #    Wait, C01 = M1 + M3.
        #    C11 = M5 + M1 - M2 + M6. M1 contributes +1 to C11.
        #    So W: {1:1, 3:1}
        
        # 2. M2: U{2:1, 3:1}, V{0:1}, W{2:1} (C10)
        #    C10 = M2 + M4.
        #    C11 = ... - M2 ... M2 contributes -1 to C11.
        #    So W: {2:1, 3:-1}
        
        # 3. M3: U{0:1, 1:1}, V{3:1}, W{1:1} (C01)
        #    C01 = M1 + M3.
        #    C00 = M5 + M4 - M3 + M7. M3 contributes -1 to C00.
        #    So W: {1:1, 0:-1}
        
        # 4. M4: U{3:1}, V{2:1, 0:-1}, W{2:1} (C10)
        #    C10 = M2 + M4.
        #    C00 = ... + M4 ... M4 contributes +1 to C00.
        #    So W: {2:1, 0:1}
        
        # 5. M5: U{0:1, 3:1}, V{0:1, 3:1}, W{0:1} (C00)
        #    C00 = M5 + ...
        #    C11 = M5 + ... M5 contributes +1 to C11.
        #    So W: {0:1, 3:1}
        
        # 6. M6: U{2:1, 0:-1}, V{0:1, 1:1}, W{3:1} (C11)
        #    C11 = ... + M6 ...
        #    So W: {3:1}
        
        # 7. M7: U{1:1, 3:-1}, V{2:1, 3:1}, W{0:1} (C00)
        #    C00 = ... + M7 ...
        #    So W: {0:1}
        
        # I will implement these 7 terms for 2x2x2.
        pass

    # I will implement the assembly in the code.
    # For now, I will just return the schoolbook 90 to ensure it runs, then refine.
    
    # Actually, I will implement the 88 solution (schoolbook for 3x3x3, optimized for 3x2x3).
    # This is safer.
    
    return {"U": [], "V": [], "W": []} # Placeholder

# I will write the full code in the final block.
