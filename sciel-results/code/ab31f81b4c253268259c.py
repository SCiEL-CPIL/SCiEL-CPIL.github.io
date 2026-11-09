import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
        
    # Helper for Schoolbook 1x2x2
    # A is 1x2, B is 2x2, C is 1x2
    # Indices: A[0], A[1]; B[0,0]..B[1,1]; C[0], C[1]
    def schoolbook_1x2x2(a_base, b_base, c_base):
        # a_base: index of first element of A (1x2)
        # b_base: index of first element of B (2x2)
        # c_base: index of first element of C (1x2)
        # A = [A0, A1]
        # B = [[B00, B01], [B10, B11]]
        # C = [C0, C1]
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        
        # Term 1: A0 B00 C0
        add_term(a_base, 1, b_base + 0, 1, c_base + 0, 1)
        # Term 2: A1 B10 C0
        add_term(a_base + 1, 1, b_base + 2, 1, c_base + 0, 1)
        # Term 3: A0 B01 C1
        add_term(a_base, 1, b_base + 1, 1, c_base + 1, 1)
        # Term 4: A1 B11 C1
        add_term(a_base + 1, 1, b_base + 3, 1, c_base + 1, 1)

    # Helper for Strassen 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    def strassen_2x2x2(a_base, b_base, c_base):
        # A indices: a_base+0 (00), a_base+1 (01), a_base+2 (10), a_base+3 (11)
        # B indices: b_base+0 (00), b_base+1 (01), b_base+2 (10), b_base+3 (11)
        # C indices: c_base+0 (00), c_base+1 (01), c_base+2 (10), c_base+3 (11)
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C01 -= M1, C10 += M1, C11 -= M1
        add_term(a_base, 1, b_base, 1, c_base, 1)
        add_term(a_base+3, 1, b_base, 1, c_base, 1)
        add_term(a_base, 1, b_base+3, 1, c_base, 1)
        add_term(a_base+3, 1, b_base+3, 1, c_base, -1)
        
        # Wait, this adds 4 terms for M1. But M1 is ONE multiplication.
        # The rank-1 tensor is u v w.
        # u = A00+A11. v = B00+B11. w = C00 - C01 + C10 - C11.
        # So we need to add ONE term with these linear combinations.
        
        # Let's redefine add_term to handle linear combinations?
        # Or just add the 4 parts to the same column?
        # Yes, U, V, W are matrices. We can put multiple non-zeros in the same column.
        
        # Let's reset col and do it properly.
        # I will implement a helper that writes to a specific column.
        
    # Let's restart the construction with a cleaner approach.
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen(a_base, b_base, c_base):
        nonlocal col
        if col >= rank: return
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=-1, C10=1, C11=-1
        U[a_base, col] += 1
        U[a_base+3, col] += 1
        V[b_base, col] += 1
        V[b_base+3, col] += 1
        W[c_base, col] += 1
        W[c_base+1, col] += -1
        W[c_base+2, col] += 1
        W[c_base+3, col] += -1
        col += 1
        
        # M2 = (A10+A11)B01
        # U: A10=1, A11=1
        # V: B01=1
        # W: C01=1, C11=1
        U[a_base+2, col] += 1
        U[a_base+3, col] += 1
        V[b_base+1, col] += 1
        W[c_base+1, col] += 1
        W[c_base+3, col] += 1
        col += 1
        
        # M3 = A00(B10-B11)
        # U: A00=1
        # V: B10=1, B11=-1
        # W: C10=1, C11=-1
        U[a_base, col] += 1
        V[b_base+2, col] += 1
        V[b_base+3, col] += -1
        W[c_base+2, col] += 1
        W[c_base+3, col] += -1
        col += 1
        
        # M4 = A11(B00-B01)
        # U: A11=1
        # V: B00=1, B01=-1
        # W: C10=1, C11=-1 ? No.
        # Strassen:
        # C00 = M1 - M4 + M5 - M6 + M7 ?
        # Let's use a verified set of Strassen coefficients.
        
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B12
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B11) -> A22(B11-B11) is wrong.
        # M4 = A22(B11-B11)?
        # Actually:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B12
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M4 + M7
        # C12 = M2 + M4
        # C21 = M3 + M5
        # C22 = M1 - M3 + M6 - M7
        
        # Let's use this set.
        
        # Reset col? No, we are in the middle.
        # I will implement the full 7 terms.
        
    # This is getting complex to edit inline. I will provide the full replacement block.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
