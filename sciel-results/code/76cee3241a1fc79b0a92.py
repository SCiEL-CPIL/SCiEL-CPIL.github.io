"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen terms for a 2x2x2 block
    # A subblock at (ar, ac) 2x2
    # B subblock at (br, bc) 2x2
    # C subblock at (cr, cc) 2x2
    def add_strassen(ar, ac, br, bc, cr, cc):
        nonlocal col
        # M1 = (A00 + A11) (B00 + B11)
        # A indices: ar*m+ac, (ar+1)*m+(ac+1)
        # B indices: br*p+bc, (br+1)*p+(bc+1)
        # C indices: cr*n+cc, (cr+1)*n+(cc+1)
        U[ar*m+ac][col] += 1
        U[(ar+1)*m+(ac+1)][col] += 1
        V[br*p+bc][col] += 1
        V[(br+1)*p+(bc+1)][col] += 1
        W[cr*n+cc][col] += 1
        W[(cr+1)*n+(cc+1)][col] += 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[ar*m+ac+1][col] += 0 # A10 is (ar+1, ac)
        U[(ar+1)*m+ac][col] += 1
        U[(ar+1)*m+(ac+1)][col] += 1
        V[br*p+bc][col] += 1
        W[(cr+1)*n+cc][col] += 1 # C21
        W[(cr+1)*n+(cc+1)][col] += 1 # C22
        col += 1
        
        # M3 = A00 B10
        U[ar*m+ac][col] += 1
        V[(br+1)*p+bc][col] += 1
        W[cr*n+(cc+1)][col] += 1 # C12
        W[(cr+1)*n+(cc+1)][col] += 1 # C22
        col += 1
        
        # M4 = A11 (B00 - B10)
        U[(ar+1)*m+(ac+1)][col] += 1
        V[br*p+bc][col] += 1
        V[(br+1)*p+bc][col] -= 1
        W[cr*n+cc][col] += 1 # C11
        W[(cr+1)*n+cc][col] += 1 # C21
        col += 1
        
        # M5 = (A00 + A01) B10
        U[ar*m+ac][col] += 1
        U[ar*m+ac+1][col] += 1
        V[(br+1)*p+bc][col] += 1
        W[cr*n+cc][col] -= 1 # C11
        W[cr*n+(cc+1)][col] += 1 # C12
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[(ar+1)*m+ac][col] += 1
        U[ar*m+ac][col] -= 1
        V[br*p+bc][col] += 1
        V[br*p+bc+1][col] += 1
        W[(cr+1)*n+(cc+1)][col] += 1 # C22
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[ar*m+ac+1][col] += 1
        U[(ar+1)*m+(ac+1)][col] -= 1
        V[(br+1)*p+bc][col] += 1
        V[(br+1)*p+bc+1][col] += 1
        W[cr*n+cc][col] += 1 # C11
        col += 1

    # Add Schoolbook terms for a 2x2x3 block
    def add_schoolbook_223(ar, ac, br, bc, cr, cc):
        nonlocal col
        # A 2x2, B 2x3, C 2x3
        # 12 terms
        # A00 B00, A00 B01, A00 B02
        # A01 B10, A01 B11, A01 B12
        # A10 B00, A10 B01, A10 B02
        # A11 B10, A11 B11, A11 B12
        
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C02 = A00 B02 + A01 B12
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        # C12 = A10 B02 + A11 B12
        
        # Term 1: A00 B00 -> C00
        U[ar*m+ac][col] += 1
        V[br*p+bc][col] += 1
        W[cr*n+cc][col] += 1
        col += 1
        
        # Term 2: A00 B01 -> C01
        U[ar*m+ac][col] += 1
        V[br*p+bc+1][col] += 1
        W[cr*n+cc+1][col] += 1
        col += 1
        
        # Term 3: A00 B02 -> C02
        U[ar*m+ac][col] += 1
        V[br*p+bc+2][col] += 1
        W[cr*n+cc+2][col] += 1
        col += 1
        
        # Term 4: A01 B10 -> C00
        U[ar*m+ac+1][col] += 1
        V[(br+1)*p+bc][col] += 1
        W[cr*n+cc][col] += 1
        col += 1
        
        # Term 5: A01 B11 -> C01
        U[ar*m+ac+1][col] += 1
        V[(br+1)*p+bc+1][col] += 1
        W[cr*n+cc+1][col] += 1
        col += 1
        
        # Term 6: A01 B12 -> C02
        U[ar*m+ac+1][col] += 1
        V[(br+1)*p+bc+2][col] += 1
        W[cr*n+cc+2][col] += 1
        col += 1
        
        # Term 7: A10 B00 -> C10
        U[(ar+1)*m+ac][col] += 1
        V[br*p+bc][col] += 1
        W[(cr+1)*n+cc][col] += 1
        col += 1
        
        # Term 8: A10 B01 -> C11
        U[(ar+1)*m+ac][col] += 1
        V[br*p+bc+1][col] += 1
        W[(cr+1)*n+cc+1][col] += 1
        col += 1
        
        # Term 9: A10 B02 -> C12
        U[(ar+1)*m+ac][col] += 1
        V[br*p+bc+2][col] += 1
        W[(cr+1)*n+cc+2][col] += 1
        col += 1
        
        # Term 10: A11 B10 -> C10
        U[(ar+1)*m+ac+1][col] += 1
        V[(br+1)*p+bc][col] += 1
        W[(cr+1)*n+cc][col] += 1
        col += 1
        
        # Term 11: A11 B11 -> C11
        U[(ar+1)*m+ac+1][col] += 1
        V[(br+1)*p+bc+1][col] += 1
        W[(cr+1)*n+cc+1][col] += 1
        col += 1
        
        # Term 12: A11 B12 -> C12
        U[(ar+1)*m+ac+1][col] += 1
        V[(br+1)*p+bc+2][col] += 1
        W[(cr+1)*n+cc+2][col] += 1
        col += 1

    # Blocks:
    # p=0,1: Strassen for A_L B_L and A_R B_R
    add_strassen(0, 0, 0, 0, 0, 0) # A_L (0,0) B_L (0,0) C (0,0)
    add_strassen(0, 2, 2, 0, 0, 0) # A_R (0,2) B_R (2,0) C (0,0)
    
    # p=2,3: Strassen for A_L B_L and A_R B_R
    add_strassen(0, 0, 0, 2, 0, 2) # A_L (0,0) B_L (0,2) C (0,2)
    add_strassen(0, 2, 2, 2, 0, 2) # A_R (0,2) B_R (2,2) C (0,2)
    
    # p=4,5,6: Schoolbook for A_L B_L and A_R B_R
    add_schoolbook_223(0, 0, 0, 4, 0, 4) # A_L (0,0) B_L (0,4) C (0,4)
    add_schoolbook_223(0, 2, 2, 4, 0, 4) # A_R (0,2) B_R (2,4) C (0,4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
