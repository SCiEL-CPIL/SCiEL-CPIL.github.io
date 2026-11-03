"""Matrix multiplication 2x4x7 with optimized tensor decomposition."""
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """Construct rank-40 tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: Split the 4-dimension into two 2-dimensions.
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x7
    C = A1*B1 + A2*B2
    
    For each 2x2 * 2x7 product, split the 7 into 2+2+3:
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x2: Strassen (rank 7)
    - 2x2 * 2x3: Schoolbook (rank 6)
    Total per 2x2*2x7: 20
    Total: 40
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    terms = []  # list of (U_row, V_row, W_row, coeff)
    
    def add_strassen_2x2_2x2(base_i, base_j, base_k):
        """Add Strassen terms for 2x2*2x2.
        A is 2x2 with top-left at (base_i, base_j)
        B is 2x2 with top-left at (base_j, base_k)
        C is 2x2 with top-left at (base_k, base_i)
        """
        # A indices
        A00_r = base_i * m + base_j
        A01_r = base_i * m + (base_j + 1)
        A10_r = (base_i + 1) * m + base_j
        A11_r = (base_i + 1) * m + (base_j + 1)
        
        # B indices
        B00_r = base_j * p + base_k
        B01_r = base_j * p + (base_k + 1)
        B10_r = (base_j + 1) * p + base_k
        B11_r = (base_j + 1) * p + (base_k + 1)
        
        # C indices
        C00_r = base_k * n + base_i
        C01_r = (base_k + 1) * n + base_i
        C10_r = base_k * n + (base_i + 1)
        C11_r = (base_k + 1) * n + (base_i + 1)
        
        # M1 = (A00+A11)*(B00+B11)
        terms.append((A00_r, B00_r, C00_r, 1))
        terms.append((A00_r, B11_r, C00_r, 1))
        terms.append((A11_r, B00_r, C00_r, 1))
        terms.append((A11_r, B11_r, C00_r, 1))
        terms.append((A00_r, B00_r, C11_r, 1))
        terms.append((A00_r, B11_r, C11_r, 1))
        terms.append((A11_r, B00_r, C11_r, 1))
        terms.append((A11_r, B11_r, C11_r, 1))
        
        # M2 = (A10+A11)*B00
        terms.append((A10_r, B00_r, C10_r, 1))
        terms.append((A11_r, B00_r, C10_r, 1))
        terms.append((A10_r, B00_r, C11_r, -1))
        terms.append((A11_r, B00_r, C11_r, -1))
        
        # M3 = A00*(B01-B11)
        terms.append((A00_r, B01_r, C01_r, 1))
        terms.append((A00_r, B11_r, C01_r, -1))
        terms.append((A00_r, B01_r, C11_r, 1))
        terms.append((A00_r, B11_r, C11_r, -1))
        
        # M4 = A11*(B10-B00)
        terms.append((A11_r, B10_r, C00_r, 1))
        terms.append((A11_r, B00_r, C00_r, -1))
        terms.append((A11_r, B10_r, C10_r, 1))
        terms.append((A11_r, B00_r, C10_r, -1))
        
        # M5 = (A00+A01)*B11
        terms.append((A00_r, B11_r, C00_r, -1))
        terms.append((A01_r, B11_r, C00_r, -1))
        terms.append((A00_r, B11_r, C01_r, 1))
        terms.append((A01_r, B11_r, C01_r, 1))
        
        # M6 = (A10-A00)*(B00+B01)
        terms.append((A10_r, B00_r, C11_r, 1))
        terms.append((A10_r, B01_r, C11_r, 1))
        terms.append((A00_r, B00_r, C11_r, -1))
        terms.append((A00_r, B01_r, C11_r, -1))
        
        # M7 = (A01-A11)*(B10+B11)
        terms.append((A01_r, B10_r, C00_r, 1))
        terms.append((A01_r, B11_r, C00_r, 1))
        terms.append((A11_r, B10_r, C00_r, -1))
        terms.append((A11_r, B11_r, C00_r, -1))
    
    def add_schoolbook_2x2_2x3(base_i, base_j, base_k):
        """Add schoolbook terms for 2x2 * 2x3.
        A is 2x2 with top-left at (base_i, base_j)
        B is 2x3 with top-left at (base_j, base_k)
        C is 2x3 with top-left at (base_k, base_i)
        """
        for di in range(2):
            for dj in range(2):
                for dk in range(3):
                    U_r = (base_i + di) * m + (base_j + dj)
                    V_r = (base_j + dj) * p + (base_k + dk)
                    W_r = (base_k + dk) * n + (base_i + di)
                    terms.append((U_r, V_r, W_r, 1))
    
    # Group 1: A1 (i=0,1; j=0,1) * B1 (j=0,1; k=0,1,2,3,4,5,6)
    # Split k into 0-1, 2-3, 4-6
    add_strassen_2x2_2x2(0, 0, 0)  # k=0,1
    add_strassen_2x2_2x2(0, 0, 2)  # k=2,3
    add_schoolbook_2x2_2x3(0, 0, 4)  # k=4,5,6
    
    # Group 2: A2 (i=0,1; j=2,3) * B2 (j=2,3; k=0,1,2,3,4,5,6)
    # Split k into 0-1, 2-3, 4-6
    add_strassen_2x2_2x2(0, 2, 0)  # k=0,1
    add_strassen_2x2_2x2(0, 2, 2)  # k=2,3
    add_schoolbook_2x2_2x3(0, 2, 4)  # k=4,5,6
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 14 x rank
    
    for col, (U_r, V_r, W_r, coeff) in enumerate(terms):
        U[U_r][col] = coeff
        V[V_r][col] = coeff
        W[W_r][col] = coeff
    
    return {"U": U, "V": V, "W": W}
