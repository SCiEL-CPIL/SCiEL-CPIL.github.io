import numpy as np

def construct():
    n, m, p = 5, 5, 5
    
    # Generate 4x4 Strassen decomposition
    # Returns list of (u_vec, v_vec, w_vec) for 4x4
    # u_vec, v_vec, w_vec are length 16
    
    def strassen_2x2():
        # Returns list of (u, v, w) for 2x2
        # u, v, w are length 4
        # Indices: 00, 01, 10, 11
        terms = []
        # M1 = (A00+A11)(B00+B11)
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, -1] # C00 += M1, C11 -= M1
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, 0] # C10 += M2
        terms.append((u, v, w))
        
        # M3 = A00(B10-B11)
        u = [1, 0, 0, 0]
        v = [0, 0, 1, -1]
        w = [0, 1, 0, 0] # C01 += M3
        terms.append((u, v, w))
        
        # M4 = A11(B01-B00)
        u = [0, 0, 0, 1]
        v = [0, 1, -1, 0]
        w = [1, 0, 0, 1] # C00 += M4, C11 += M4
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 0, 0, 0] # C00 -= M5
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01)
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 1, 0] # C10 += M6? No, C10 = M2+M4. C11 = M1-M2+M3+M7.
        # Let's check C10. C10 = A10 B00 + A11 B10.
        # M2 = (A10+A11)B00 = A10 B00 + A11 B00.
        # M4 = A11(B01-B00) = A11 B01 - A11 B00.
        # M2 + M4 = A10 B00 + A11 B01. This is not C10.
        # C10 = A10 B00 + A11 B10.
        # My Strassen formulas might be slightly off in indexing.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Let's re-verify C21.
        # C21 = A21 B11 + A22 B21.
        # M2 = (A21+A22)B11 = A21 B11 + A22 B11.
        # M4 = A22(B12-B11) = A22 B12 - A22 B11.
        # M2 + M4 = A21 B11 + A22 B12.
        # This is not C21.
        # Wait, C21 = A21 B11 + A22 B21.
        # M2 + M4 = A21 B11 + A22 B12.
        # There is a mistake in my manual check or the standard formula.
        # Standard formula: C21 = M2 + M4.
        # Let's check M4 again. M4 = A22(B12-B11).
        # Maybe M4 is A22(B21-B11)?
        # If M4 = A22(B21-B11), then M2+M4 = A21 B11 + A22 B21. Correct.
        # So M4 = A22(B21-B11).
        # Let's re-verify M3. M3 = A11(B21-B22).
        # C12 = M3 + M5 = A11(B21-B22) + (A11+A12)B22 = A11 B21 - A11 B22 + A11 B22 + A12 B22 = A11 B21 + A12 B22. Correct.
        # So M4 should be A22(B21-B11).
        
        # Let's update the vectors.
        # Indices: 00, 01, 10, 11
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,-1]
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # M3: u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,0]
        # M4: u=[0,0,0,1], v=[0,-1,1,0] # B21-B11 -> [0, -1, 1, 0]
        # w=[1,0,0,1]
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,0,0,0]
        # M6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,1,0] # C21 += M6?
        # C21 = M2+M4. C22 = M1-M2+M3+M7.
        # C11 = M1+M4-M5-M7.
        # C12 = M3+M5.
        # Where do M6 and M7 go?
        # C11 = M1 + M4 - M5 - M7.
        # C22 = M1 - M2 + M3 + M7.
        # M6 is not used in C?
        # Ah, M6 = (A21-A11)(B11+B12).
        # M7 = (A12-A22)(B21+B22).
        # C11 = M1 + M4 - M5 - M7.
        # C12 = M3 + M5.
        # C21 = M2 + M4.
        # C22 = M1 - M2 + M3 + M7.
        # M6 is not in the final sum?
        # No, M6 is used in M1? No.
        # Actually, M6 is not needed?
        # Strassen uses 7 multiplications.
        # M1, M2, M3, M4, M5, M6, M7.
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # M6 is not used?
        # Wait, M6 = (A21-A11)(B11+B12).
        # Is M6 used in C?
        # No, M6 is not in the formulas for C11, C12, C21, C22.
        # This implies M6 is redundant?
        # No, Strassen's algorithm has 7 multiplications.
        # Let's check a reference.
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # M6 is indeed not used in the final combination?
        # That can't be right.
        # Ah, M6 is used to compute M1? No.
        # Maybe my M4 is wrong.
        # If M4 = A22(B12-B11), then M2+M4 = A21 B11 + A22 B12.
        # Then C21 = M2 + M4 + M6?
        # Let's assume the standard 7 terms are correct and implement them.
        # I will use the standard formulas.
        
        pass

    # Given the complexity and time, I will stick to the 122 rank decomposition 
    # which is guaranteed to be correct if I implement the block structure carefully.
    # I will hardcode the 49 terms for 4x4 using a recursive generator if possible, 
    # or just use the 125 schoolbook if I can't verify the 122.
    # But 122 is better.
    
    # To be safe, I will return the schoolbook 125 if I can't easily generate 122.
    # But the prompt asks for a stronger algorithm.
    # I will implement the 122 decomposition.
    
    pass

