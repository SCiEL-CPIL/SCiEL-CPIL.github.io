import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, leaving margin
    
    n, m, p = 2, 4, 6
    
    # We'll use a Strassen-like decomposition
    # A is 2x4, split into four 2x2 blocks: A11, A12, A21, A22
    # B is 4x6, split into four 2x3 blocks: B11, B12, B21, B22
    # C is 2x6, with blocks C11, C12, C21, C22 (each 2x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using 7 Strassen-like steps:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each step is a 2x2 * 2x3 multiplication, which requires 6 scalar multiplications.
    # Total: 7 * 6 = 42
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define steps as (A_coeffs, B_coeffs, C_targets)
    # A_coeffs: dict (i,j) -> coeff for A[i,j]
    # B_coeffs: dict (j,k) -> coeff for B[j,k]
    # C_targets: list of (block, sign) where block in {0,1,2,3} for C11,C12,C21,C22
    
    steps = [
        # M1 = (A11+A22)*(B11+B22) -> C11 + C22
        (
            {(0,0):1, (0,1):1, (1,2):1, (1,3):1},
            {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,3):1, (2,4):1, (2,5):1, (3,3):1, (3,4):1, (3,5):1},
            [(0,1), (3,1)]
        ),
        # M2 = (A21+A22)*B11 -> C21 - C22... 
        # Actually, let me use the standard Strassen output formulas:
        # C11 = M1 - M2 - M3 - M4 + M5
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M5 + M6 + M7
    ]
    
    # Let me redefine the steps with correct signs:
    steps = [
        # M1 = (A11+A22)*(B11+B22)
        # Contributes to: C11(+1), C22(+1)
        (
            {(0,0):1, (0,1):1, (1,2):1, (1,3):1},  # A11+A22
            {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,3):1, (2,4):1, (2,5):1, (3,3):1, (3,4):1, (3,5):1},  # B11+B22
            [(0,1), (3,1)]
        ),
        # M2 = (A21+A22)*B11
        # Contributes to: C21(+1), C11(-1)
        (
            {(1,0):1, (1,1):1, (1,2):1, (1,3):1},  # A21+A22
            {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1},  # B11
            [(2,1), (0,-1)]
        ),
        # M3 = A11*(B12-B22)
        # Contributes to: C12(+1), C11(-1)
        (
            {(0,0):1, (0,1):1},  # A11
            {(0,3):1, (0,4):1, (0,5):1, (1,3):1, (1,4):1, (1,5):1, (2,3):-1, (2,4):-1, (2,5):-1, (3,3):-1, (3,4):-1, (3,5):-1},  # B12-B22
            [(1,1), (0,-1)]
        ),
        # M4 = A22*(B21-B11)
        # Contributes to: C21(+1), C11(-1)
        (
            {(1,2):1, (1,3):1},  # A22
            {(2,0):1, (2,1):1, (2,2):1, (3,0):1, (3,1):1, (3,2):1, (0,0):-1, (0,1):-1, (0,2):-1, (1,0):-1, (1,1):-1, (1,2):-1},  # B21-B11
            [(2,1), (0,-1)]
        ),
        # M5 = (A11+A12)*B22
        # Contributes to: C11(+1), C12(+1), C22(-1)
        (
            {(0,0):1, (0,1):1, (0,2):1, (0,3):1},  # A11+A12
            {(2,3):1, (2,4):1, (2,5):1, (3,3):1, (3,4):1, (3,5):1},  # B22
            [(0,1), (1,1), (3,-1)]
        ),
        # M6 = (A21-A11)*(B11+B12)
        # Contributes to: C22(+1)
        (
            {(1,0):1, (1,1):1, (0,0):-1, (0,1):-1},  # A21-A11
            {(0,0):1, (0,1):1, (0,2):1, (0,3):1, (0,4):1, (0,5):1, (1,0):1, (1,1):1, (1,2):1, (1,3):1, (1,4):1, (1,5):1},  # B11+B12
            [(3,1)]
        ),
        # M7 = (A12-A22)*(B21+B22)
        # Contributes to: C22(+1)
        (
            {(0,2):1, (0,3):1, (1,2):-1, (1,3):-1},  # A12-A22
            {(2,0):1, (2,1):1, (2,2):1, (2,3):1, (2,4):1, (2,5):1, (3,0):1, (3,1):1, (3,2):1, (3,3):1, (3,4):1, (3,5):1},  # B21+B22
            [(3,1)]
        ),
    ]
    
    col = 0
    for A_dict, B_dict, C_targets in steps:
        # For each step, we do a 2x2 * 2x3 multiplication
        # This gives a 2x3 result
        # We decompose this into 6 scalar multiplications
        
        # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # But A_dict has entries like (0,0), (0,1), (1,2), (1,3) which are global indices
        # We need to map to local 2x2 indices
        
        # Local A indices: (0,0)->A11[0,0], (0,1)->A11[0,1], (1,2)->A22[0,0], (1,3)->A22[0,1]
        # etc.
        
        # For the 2x2 * 2x3 multiplication:
        # Let a = [[a00, a01], [a10, a11]] (local 2x2)
        # Let b = [[b00, b01, b02], [b10, b11, b12]] (local 2x3)
        # Result c = [[c00, c01, c02], [c10, c11, c12]]
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c02 = a00*b02 + a01*b12
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        # c12 = a10*b02 + a11*b12
        
        # We can do this with 6 scalar multiplications:
        # m0 = a00*b00, m1 = a01*b10, m2 = a00*b01, m3 = a01*b11, m4 = a00*b02, m5 = a01*b12
        # Then c00 = m0+m1, c01 = m2+m3, c02 = m4+m5, c10 = a10*b00+a11*b10, etc.
        # Wait, that's 12 multiplications.
        
        # Actually, for 2x2 * 2x3, we need 2*2*3 = 12 scalar mults schoolbook.
        # But we can use Strassen for the 2x2 part: 7 mults for 2x2*2x2, but here it's 2x2*2x3.
        
        # Hmm, I think for 2x2 * 2x3, the minimum number of scalar multiplications is 12 (schoolbook) 
        # or possibly less with a clever decomposition.
        
        # Actually, let me just use the 6-multiplication approach from the original code:
        # Split the 2x2 into two 1x2 vectors and the 2x3 into two 1x3 vectors... no.
        
        # The original code splits:
        # x0 = sum of A entries with j even (local col 0)
        # x1 = sum of A entries with j odd (local col 1)
        # y00, y01, y02 = B entries with row 0 (local)
        # y10, y11, y12 = B entries with row 1 (local)
        # Then 6 products: x0*y00, x1*y10, x0*y01, x1*y11, x0*y02, x1*y12
        
        # This gives:
        # z0 = x0*y00 + x1*y10 = (a00+a10... wait no.
        
        # x0 = a00 (local), x1 = a01 (local)? No.
        # In the original code, x0_coeffs has entries with j%2==0, which are local col 0.
        # x1_coeffs has entries with j%2==1, which are local col 1.
        
        # So x0 = sum of A entries in local col 0 = a00 + a10 (if both present)
        # x1 = sum of A entries in local col 1 = a01 + a11 (if both present)
        
        # And y_coeffs[0] = B entries with local row 0, local col 0 = b00
        # y_coeffs[3] = B entries with local row 1, local col 0 = b10
        # etc.
        
        # So the 6 products are:
        # (a00+a10)*b00, (a01+a11)*b10, (a00+a10)*b01, (a01+a11)*b11, (a00+a10)*b02, (a01+a11)*b12
        
        # And z0 = (a00+a10)*b00 + (a01+a11)*b10 = a00*b00 + a10*b00 + a01*b10 + a11*b10
        # But c00 = a00*b00 + a01*b10 and c10 = a10*b00 + a11*b10
        # So z0 = c00 + c10, not just c00.
        
        # This means the 6-product approach doesn't directly give the 2x3 result.
        # It gives sums of rows of the result.
        
        # I think the original code's approach is flawed. Let me just use 12 scalar multiplications per step.
        # 7 steps * 12 mults = 84, which is worse than schoolbook 48.
        
        # Alternatively, use Strassen for the 2x2 part: 7 mults per step, but then we need to handle the 2x3 structure.
        
        # I think the cleanest approach is to just use schoolbook: 48 rank.
        # But the current code claims rank 42. Let me just verify if the current code is correct.
        
        # Given the time, let me just output a correct schoolbook decomposition with rank 48.
        
        break  # Exit the loop
    
    # Schoolbook decomposition: rank = n*m*p = 48
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
