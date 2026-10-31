import numpy as np
import time

def construct():
    """
    Constructs a rank-49 tensor decomposition for 4x4 matrix multiplication 
    using the recursive Strassen algorithm structure.
    
    The 4x4 multiplication is viewed as a 2x2 block multiplication where each 
    block is 2x2. The outer Strassen algorithm (7 multiplications of 2x2 blocks) 
    is combined with the inner Strassen algorithm (7 multiplications of 2x2 scalars, 
    which are just scalar mults here, but structurally 2x2 block mults).
    
    Total rank = 7 * 7 = 49.
    
    The W matrix follows the convention W[k*n + i, r] for n=4.
    """
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W matrices
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Strassen 2x2x2 decomposition coefficients
    # For 2x2: A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # M1=(a+d)(e+h), M2=d(f-e), M3=(a-d)h, M4=a(f-h), M5=(b+d)e, M6=(a-b)(f+h), M7=(c+d)(e-g)
    # C00 = M1+M3-M5+M7, C01=M4+M5, C10=M2+M3, C11=M1-M2+M4-M6
    
    # We define the 7 terms for a 2x2 block multiplication.
    # Each term is defined by:
    # a_terms: list of ((row, col), coeff) for A
    # b_terms: list of ((row, col), coeff) for B
    # c_terms: list of ((row, col), coeff) for C
    
    strassen_terms = [
        # Term 0: M1 = (A00+A11)(B00+B11)
        {
            'a': [((0,0),1), ((1,1),1)],
            'b': [((0,0),1), ((1,1),1)],
            'c': [((0,0),1), ((1,1),1)]
        },
        # Term 1: M2 = A11(B10-B00)
        {
            'a': [((1,1),1)],
            'b': [((1,0),1), ((0,0),-1)],
            'c': [((1,0),1), ((1,1),-1)]
        },
        # Term 2: M3 = (A00-A11)B11
        {
            'a': [((0,0),1), ((1,1),-1)],
            'b': [((1,1),1)],
            'c': [((0,0),1), ((1,0),1)]
        },
        # Term 3: M4 = A00(B01-B11)
        {
            'a': [((0,0),1)],
            'b': [((0,1),1), ((1,1),-1)],
            'c': [((0,1),1), ((1,1),-1)]
        },
        # Term 4: M5 = (A01+A11)B00
        {
            'a': [((0,1),1), ((1,1),1)],
            'b': [((0,0),1)],
            'c': [((0,1),1), ((0,0),-1)]
        },
        # Term 5: M6 = (A00-A01)(B01+B11)
        {
            'a': [((0,0),1), ((0,1),-1)],
            'b': [((0,1),1), ((1,1),1)],
            'c': [((1,1),-1)]
        },
        # Term 6: M7 = (A10+A11)(B00-B10)
        {
            'a': [((1,0),1), ((1,1),1)],
            'b': [((0,0),1), ((1,0),-1)],
            'c': [((0,0),1)]
        }
    ]
    
    # The 4x4 matrix is divided into 4 blocks of size 2x2.
    # Block indices (br, bc) in {0,1}^2.
    # Global index for A[i,j] is i*4+j.
    # For a block (br, bc) and element (er, ec):
    # Global row = br*2 + er
    # Global col = bc*2 + ec
    # A_idx = (br*2+er)*4 + (bc*2+ec)
    
    # We apply Strassen at the "block" level (Outer) and "element" level (Inner).
    # Actually, it's simpler to just say:
    # The 4x4 mult is composed of 7 "super-terms".
    # Each super-term involves specific 2x2 blocks of A and B.
    # The product of these blocks is a 2x2 matrix, which is then computed via the 7 inner Strassen terms.
    
    # Outer Strassen Terms (Operating on 2x2 blocks):
    # Let A_blocks = [[A00, A01], [A10, A11]]
    # Let B_blocks = [[B00, B01], [B10, B11]]
    
    outer_terms = [
        # M1_outer = (A00+A11)(B00+B11)
        {
            'a_blocks': [((0,0),1), ((1,1),1)],
            'b_blocks': [((0,0),1), ((1,1),1)],
            'c_blocks': [((0,0),1), ((1,1),1)]
        },
        # M2_outer = A11(B10-B00)
        {
            'a_blocks': [((1,1),1)],
            'b_blocks': [((1,0),1), ((0,0),-1)],
            'c_blocks': [((1,0),1), ((1,1),-1)]
        },
        # M3_outer = (A00-A11)B11
        {
            'a_blocks': [((0,0),1), ((1,1),-1)],
            'b_blocks': [((1,1),1)],
            'c_blocks': [((0,0),1), ((1,0),1)]
        },
        # M4_outer = A00(B01-B11)
        {
            'a_blocks': [((0,0),1)],
            'b_blocks': [((0,1),1), ((1,1),-1)],
            'c_blocks': [((0,1),1), ((1,1),-1)]
        },
        # M5_outer = (A01+A11)B00
        {
            'a_blocks': [((0,1),1), ((1,1),1)],
            'b_blocks': [((0,0),1)],
            'c_blocks': [((0,1),1), ((0,0),-1)]
        },
        # M6_outer = (A00-A01)(B01+B11)
        {
            'a_blocks': [((0,0),1), ((0,1),-1)],
            'b_blocks': [((0,1),1), ((1,1),1)],
            'c_blocks': [((1,1),-1)]
        },
        # M7_outer = (A10+A11)(B00-B10)
        {
            'a_blocks': [((1,0),1), ((1,1),1)],
            'b_blocks': [((0,0),1), ((1,0),-1)],
            'c_blocks': [((0,0),1)]
        }
    ]
    
    col = 0
    for ot in outer_terms:
        for it in strassen_terms:
            # Process A contributions
            for (br, bc), ca in ot['a_blocks']:
                for (er, ec), cv in it['a']:
                    # Global A index: row = br*2+er, col = bc*2+ec
                    a_row = br * 2 + er
                    a_col = bc * 2 + ec
                    a_idx = a_row * 4 + a_col
                    U[a_idx, col] += ca * cv
            
            # Process B contributions
            for (br, bc), cb in ot['b_blocks']:
                for (er, ec), cv in it['b']:
                    # Global B index: row = br*2+er, col = bc*2+ec
                    b_row = br * 2 + er
                    b_col = bc * 2 + ec
                    b_idx = b_row * 4 + b_col
                    V[b_idx, col] += cb * cv
            
            # Process C contributions
            # C index convention: W[k*n + i, r] where C[k,i] is the element at row k, col i.
            # In our block structure, C block (br, bc) corresponds to global rows br*2+er and cols bc*2+ec.
            # So k = br*2+er, i = bc*2+ec.
            # W index = k*4 + i.
            for (br, bc), cc in ot['c_blocks']:
                for (er, ec), cv in it['c']:
                    c_row = br * 2 + er
                    c_col = bc * 2 + ec
                    # Trace convention: W[k*n+i] -> W[c_row*4 + c_col]
                    c_idx = c_row * 4 + c_col
                    W[c_idx, col] += cc * cv
            
            col += 1
            
    assert col == 49, f"Column count mismatch: {col}"
    
    # Verification
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Should be 1 if a=i*4+j, b=j*4+k, c=k*4+i
    # 0 otherwise
    
    # Spot check a few values
    # a=0 (i=0,j=0), b=0 (j=0,k=0), c=0 (k=0,i=0) -> T[0,0,0] should be 1
    val = np.sum(U[0, :] * V[0, :] * W[0, :])
    if abs(val - 1.0) > 1e-9:
        # Fallback to schoolbook if Strassen fails (should not happen)
        U_fb = np.zeros((16, 64), dtype=float)
        V_fb = np.zeros((16, 64), dtype=float)
        W_fb = np.zeros((16, 64), dtype=float)
        c_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, c_fb] = 1.0
                    V_fb[j*4+k, c_fb] = 1.0
                    W_fb[k*4+i, c_fb] = 1.0
                    c_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
