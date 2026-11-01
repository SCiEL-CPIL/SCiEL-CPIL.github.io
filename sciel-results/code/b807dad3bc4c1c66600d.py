"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # We'll construct a rank-20 decomposition by splitting into two 2x2 x 2x3 blocks
    # For each 2x2 x 2x3 block, we need rank 10
    
    # For 2x2 x 2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Let's find a rank-10 decomposition
    
    # Variables: A00, A01, A10, A11, B00, B01, B02, B10, B11, B12
    # Outputs: C00, C01, C02, C10, C11, C12
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Rank r decomposition: M_c = (sum_{i,j} U[i,j,c]*A[i,j]) * (sum_{j,k} V[j,k,c]*B[j,k])
    # C[i,k] = sum_c W[i,k,c] * M_c
    
    # Let me try to construct this explicitly.
    # Use Strassen-like ideas.
    
    # For 2x2 x 2x2, Strassen gives rank 7:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6
    
    # For 2x2 x 2x3, we need to extend this.
    # The third column adds: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
    
    # We can add 3 more terms for the third column (schoolbook for that column):
    # M8 = A00*B02, M9 = A01*B12, M10 = ... wait, that's only 2 terms for column 2
    # Actually for column 2: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
    # Schoolbook for column 2 needs 4 terms, but we can share with existing structure.
    
    # Let me try: use Strassen for columns 0,1 (7 terms) + 3 more for column 2 = 10 terms
    # For column 2, we need: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
    # We can use: M8 = (A00+A01+A10+A11)*B02 ... no, that doesn't work directly.
    
    # Alternative: M8 = A00*B02, M9 = A01*B12, M10 = (A10+A11)*B02
    # C02 = M8 + M9
    # C12 = M10 + A11*B02 - A10*B02 + A11*B12 ... this is getting complicated.
    
    # Let me just try to solve it numerically for rank 10.
    
    deadline = time.time() + 110
    
    def solve_block(rank):
        """Find rank decomposition for 2x2 x 2x3"""
        np.random.seed(123)
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                return None
            
            # Initialize
            U = np.random.randn(4, rank)  # A indices: 0=A00, 1=A01, 2=A10, 3=A11
            V = np.random.randn(6, rank)  # B indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
            W = np.random.randn(6, rank)  # C indices: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
            
            # Gradient descent on the quadratic loss
            lr = 0.005
            for step in range(10000):
                if time.time() > deadline - 5:
                    return None
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # For each output (i,k) and each input pair (j, jp):
                # residual_{i,k,j,jp} = sum_c W[i,k,c]*U[i,j,c]*V[jp,k,c] - delta_{j,jp}
                # where i in {0,1}, k in {0,1,2}, j in {0,1}, jp in {0,1}
                
                for i in range(2):
                    for k in range(3):
                        for j in range(2):
                            for jp in range(2):
                                target = 1.0 if j == jp else 0.0
                                res = 0.0
                                for c in range(rank):
                                    res += W[i*3+k, c] * U[i*2+j, c] * V[jp*3+k, c]
                                res -= target
                                
                                for c in range(rank):
                                    grad_U[i*2+j, c] += 2*res * W[i*3+k, c] * V[jp*3+k, c]
                                    grad_V[jp*3+k, c] += 2*res * W[i*3+k, c] * U[i*2+j, c]
                                    grad_W[i*3+k, c] += 2*res * U[i*2+j, c] * V[jp*3+k, c]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if step % 1000 == 0:
                    # Check
                    max_res = 0
                    for i in range(2):
                        for k in range(3):
                            for j in range(2):
                                for jp in range(2):
                                    res = 0
                                    for c in range(rank):
                                        res += W[i*3+k, c] * U[i*2+j, c] * V[jp*3+k, c]
                                    res -= (1.0 if j == jp else 0.0)
                                    max_res = max(max_res, abs(res))
                    if max_res < 1e-8:
                        return (U, V, W)
        
        return None
    
    # Try to find rank-10 decomposition for the 2x2 x 2x3 block
    result = solve_block(10)
    
    if result is not None:
        U_block, V_block, W_block = result
        # Extend to 2x2 x 2x6 by repeating for the second half
        r = 20
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(12)]
        W = [[0]*r for _ in range(12)]
        
        for c in range(10):
            for i in range(4):
                U[i][c] = U_block[i, c]
                U[i][c+10] = U_block[i, c]
            for j in range(6):
                V[j][c] = V_block[j, c]
                V[j+6][c+10] = V_block[j, c]
            for k in range(6):
                W[k][c] = W_block[k, c]
                W[k+6][c+10] = W_block[k, c]
        
        # Verify
        valid = True
        for i in range(2):
            for k in range(6):
                for j in range(2):
                    for jp in range(2):
                        total = 0
                        for c in range(r):
                            total += W[i*6+k][c] * U[i*2+j][c] * V[jp*6+k][c]
                        expected = 1 if j == jp else 0
                        if abs(total - expected) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to fractions
            U_frac = [[Fraction(U[i][c]).limit_denominator(100) for c in range(r)] for i in range(4)]
            V_frac = [[Fraction(V[i][c]).limit_denominator(100) for c in range(r)] for i in range(12)]
            W_frac = [[Fraction(W[i][c]).limit_denominator(100) for c in range(r)] for i in range(12)]
            
            # Final verification with fractions
            valid_frac = True
            for i in range(2):
                for k in range(6):
                    for j in range(2):
                        for jp in range(2):
                            total = 0
                            for c in range(r):
                                total += W_frac[i*6+k][c] * U_frac[i*2+j][c] * V_frac[jp*6+k][c]
                            expected = 1 if j == jp else 0
                            if total != expected:
                                valid_frac = False
                                break
                        if not valid_frac:
                            break
                    if not valid_frac:
                        break
                if not valid_frac:
                    break
            if valid_frac:
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fallback: rank-21 Strassen-based
    # Let's construct a known rank-21 solution
    # For 2x2 x 2x6, use Strassen for 2x2 x 2x2 (7 terms) applied to each pair of columns
    # Columns 0,1: 7 terms; columns 2,3: 7 terms; columns 4,5: 7 terms = 21 terms
    
    r = 21
    U = [[0]*r for _ in range(4)]
    V = [[0]*r for _ in range(12)]
    W = [[0]*r for _ in range(12)]
    
    # Strassen for 2x2 x 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6
    
    strassen_U = [
        [1, 0, 1, 0, 0],  # A00: M1, M3, M5, M6
        [0, 1, 0, 0, 0],  # A01: M5, M7
        [0, 1, 0, 1, 0],  # A10: M2, M4, M6
        [1, 1, 0, 1, 1]   # A11: M1, M2, M4, M7
    ]
    # Wait, let me be more careful.
    # M1 = (A00+A11)*(B00+B11): U_M1 = [1,0,0,1], V_M1 = [1,0,0,1]
    # M2 = (A10+A11)*B00: U_M2 = [0,0,1,1], V_M2 = [1,0,0,0]
    # M3 = A00*(B01-B11): U_M3 = [1,0,0,0], V_M3 = [0,1,0,-1]
    # M4 = A11*(B10-B00): U_M4 = [0,0,0,1], V_M4 = [0,0,1,0]... wait B10-B00 = [-1,0,1,0]
    # Let me redo: B indices for 2x2: B00, B01, B10, B11
    # M1 = (A00+A11)*(B00+B11): U=[1,0,0,1], V=[1,0,0,1]
    # M2 = (A10+A11)*B00: U=[0,0,1,1], V=[1,0,0,0]
    # M3 = A00*(B01-B11): U=[1,0,0,0], V=[0,1,0,-1]
    # M4 = A11*(B10-B00): U=[0,0,0,1], V=[-1,0,1,0]
    # M5 = (A00+A01)*B11: U=[1,1,0,0], V=[0,0,0,1]
    # M6 = (A10-A00)*(B00+B01): U=[-1,0,1,0], V=[1,1,0,0]
    # M7 = (A01-A11)*(B10+B11): U=[0,1,0,-1], V=[0,0,1,1]
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6
    
    # For 2x2 x 2x6, split into 3 blocks of 2 columns each.
    # Block 0: columns 0,1 of B; Block 1: columns 2,3; Block 2: columns 4,5
    
    for block in range(3):
        base = block * 7
        for c in range(7):
            col = base + c
            # U
            U[0][col] = strassen_U[0][c] if c < len(strassen_U[0]) else 0
            # This is getting messy. Let me just use schoolbook as fallback.
    
    # Just use schoolbook rank-24 as guaranteed valid fallback
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(n*p)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
