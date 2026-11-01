import time

def construct():
    deadline = time.time() + 100
    n, m, p = 2, 2, 9
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Strassen 2x2x2
    # A indices: 0,1,2,3
    # B block indices (local): 0,1,2,3
    # C block indices (local): 0,1,2,3
    
    # The 7 terms of Strassen
    # u: coeffs for A
    # v: coeffs for B (local)
    # w: coeffs for C (local)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B00) -> Incorrect.
    
    # I will use the explicit 7 terms:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B01
    # 3. A00(B10-B11)
    # 4. A11(B00-B00)
    
    # Correct Strassen terms:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B01
    # M3 = A00(B10 - B11)
    # M4 = A11(B00 - B00)
    
    # I'll just use the 4 schoolbook terms for 2x2x2? No, rank 7 is better than 8.
    # Wait, schoolbook for 2x2x2 is 8 terms?
    # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$
    # $C_{01} = A_{00}B_{01} + A_{01}B_{11}$
    # $C_{10} = A_{10}B_{00} + A_{11}B_{10}$
    # $C_{11} = A_{10}B_{01} + A_{11}B_{11}$
    # Yes, 8 terms.
    # Strassen is 7 terms.
    
    # I will implement the 7 Strassen terms.
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]), # M1
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, -1]), # M3
        ([0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]),   # M4 (A11 * B00)
        ([0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),   # M5 (A01 * B01)
        ([1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]),   # M6 (A00+A01)*B10
        ([0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]),   # M7 (A10*B11)
    ]
    
    # Verify M1..M7 reconstruction:
    # C00 = M1 + M4 - M2 - M3
    # C01 = M2 + M5
    # C10 = M3 + M6
    # C11 = M1 - M4 + M5 - M6 - M7 + M7?
    
    # I will use a known correct set.
    
    # For now, I'll just use the 8 schoolbook terms for 2x2x2 to be safe, which gives rank 4*8 + 4 = 36.
    # But I want 32.
    
    # I'll use the 7 Strassen terms.
    
    pass

