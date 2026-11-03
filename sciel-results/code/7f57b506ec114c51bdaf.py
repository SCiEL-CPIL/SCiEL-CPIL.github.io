"""Matrix multiplication 3x4x4: find rank-minimal tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def _validate(U, V, W):
    """Check if U, V, W exactly decompose the 3x4x4 matmul tensor."""
    r = len(U[0])
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_strassen_4x4x4():
    """
    Construct the rank-49 decomposition for 4x4x4 matmul using Strassen's algorithm.
    Returns U, V, W of shapes (16, 49), (16, 49), (16, 49).
    Note: For 4x4x4, n=m=p=4.
    U index: i*4+j (16)
    V index: j*4+k (16)
    W index: k*4+i (16)
    """
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 4x4x4, we split into 2x2 blocks.
    # This results in 7 * 7 = 49 multiplications.
    
    # We define the 7 rank-1 tensors for 2x2x2.
    # Indices for 2x2: i,j,k in {0,1}.
    # U2: (4, 7), V2: (4, 7), W2: (4, 7).
    
    # Strassen's 7 terms:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)(b00)
    # m2 = (a00)(b10+b11)
    # m3 = (a11)(b01)
    # m4 = (a00+a01)(b11)
    # m5 = (a10-a11)(b01+b11)
    # m6 = (a01-a00)(b00+b10)
    
    # Let's build the 2x2 decomposition matrices.
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c].
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=0: m0 = (a00+a11)(b00+b11)
    # A part: a00=1, a11=1 -> U2[0,0]=1, U2[3,0]=1
    # B part: b00=1, b11=1 -> V2[0,0]=1, V2[3,0]=1
    # C part: c00 += m0, c11 += m0 -> W2[0,0]=1, W2[3,0]=1
    # Wait, W index is k*2+i.
    # c00: i=0, k=0 -> W2[0,0]. c11: i=1, k=1 -> W2[3,0].
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # c=1: m1 = (a10+a11)(b00)
    # A: a10=1, a11=1 -> U2[2,1]=1, U2[3,1]=1
    # B: b00=1 -> V2[0,1]=1
    # C: c10 += m1, c11 += m1?
    # c10: i=1, k=0 -> W2[2,1]. c11: i=1, k=1 -> W2[3,1].
    # Wait, standard Strassen:
    # C00 = m0 - m1 + m2 + m4
    # C01 = m0 + m3 - m4 - m5
    # C10 = m1 + m2 - m3 + m5
    # C11 = m0 + m1 + m2 + m6
    # Let's use this standard form.
    
    # Reset U2, V2, W2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # m0 = (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 += m0, C11 += m0
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # m1 = (a10+a11)(b00)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 += m1, C11 += m1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # m2 = (a00)(b10+b11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = 1
    # C00 += m2, C10 += m2
    W2[0, 2] = 1; W2[2, 2] = 1
    
    # m3 = (a11)(b01)
    U2[3, 3] = 1
    V2[1, 3] = 1
    # C01 -= m3, C10 -= m3
    W2[1, 3] = -1; W2[2, 3] = -1
    
    # m4 = (a00+a01)(b11)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C00 += m4, C01 -= m4
    W2[0, 4] = 1; W2[1, 4] = -1
    
    # m5 = (a10-a11)(b01+b11)
    U2[2, 5] = 1; U2[3, 5] = -1
    V2[1, 5] = 1; V2[3, 5] = 1
    # C01 -= m5, C10 += m5
    W2[1, 5] = -1; W2[2, 5] = 1
    
    # m6 = (a01-a00)(b00+b10)
    U2[1, 6] = 1; U2[0, 6] = -1
    V2[0, 6] = 1; V2[2, 6] = 1
    # C11 += m6
    W2[3, 6] = 1
    
    # Now build 4x4x4 from 2x2x2.
    # The 4x4 matrices are 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # We apply Strassen to the block multiplication.
    # This requires 7 "multiplications" of 2x2 matrices.
    # Each 2x2 multiplication requires 7 scalar multiplications.
    # Total 49.
    
    # The indices for 4x4:
    # i, j, k in {0,1,2,3}.
    # Block indices I, J, K in {0,1}.
    # i = I*2 + i', j = J*2 + j', k = K*2 + k'.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # For each of the 7 Strassen terms (s=0..6):
    # We need to compute M_s = (sum u_s A) (sum v_s B).
    # The result contributes to C blocks.
    
    # The 7 terms for block matmul:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B10+B11)
    # M3 = (A11)(B01)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A11)(B01+B11)
    # M6 = (A01-A00)(B00+B10)
    
    # C00 = M0 - M1 + M2 + M4
    # C01 = M0 + M3 - M4 - M5
    # C10 = M1 + M2 - M3 + M5
    # C11 = M0 + M1 + M2 + M6
    
    # For each s, we have a 2x2 matmul decomposition.
    # Let U2_s, V2_s, W2_s be the rank-1 components for the s-th block multiplication.
    # Actually, we can just combine the indices.
    
    # Block s corresponds to a specific combination of A and B blocks.
    # Let's define the block coefficients for A and B for each s.
    # A_s = sum_{I,J} a_{s,I,J} A_{I,J}
    # B_s = sum_{J,K} b_{s,J,K} B_{J,K}
    # C_block_contribution = W_s * (A_s B_s)
    
    # We iterate over s=0..6.
    # For each s, we iterate over the 7 scalar multiplications of the 2x2 matmul.
    # Let t=0..6 be the scalar multiplication index.
    
    for s in range(7):
        # Determine A_s and B_s blocks
        # a_s is a 2x2 matrix of coefficients for A blocks
        # b_s is a 2x2 matrix of coefficients for B blocks
        # w_s is a 2x2 matrix of coefficients for C blocks
        
        if s == 0:
            a_s = np.array([[1, 0], [0, 1]])
            b_s = np.array([[1, 0], [1, 1]]) # B00+B11? No.
            # M0 = (A00+A11)(B00+B11)
            # A part: A00, A11. B part: B00, B11.
            # This is not a simple product of sums of blocks.
            # Strassen's algorithm computes M_s as a product of two linear combinations.
            # M0 = (A00+A11) * (B00+B11).
            # So A_s = A00+A11. B_s = B00+B11.
            # a_s = [[1, 0], [0, 1]] (coeffs for A00, A01, A10, A11)
            # b_s = [[1, 0], [1, 1]]? No.
            # B00 is at (0,0), B11 is at (1,1).
            # So b_s = [[1, 0], [0, 1]].
            # Wait, B00+B11.
            # b_s[0,0]=1, b_s[1,1]=1.
            b_s = np.array([[1, 0], [0, 1]])
            # C contribution: C00 += M0, C11 += M0.
            w_s = np.array([[1, 0], [0, 1]])
            
        elif s == 1:
            # M1 = (A10+A11)(B00)
            a_s = np.array([[0, 0], [1, 1]])
            b_s = np.array([[1, 0], [0, 0]])
            # C10 += M1, C11 += M1
            w_s = np.array([[0, 0], [1, 1]])
            
        elif s == 2:
            # M2 = (A00)(B10+B11)
            a_s = np.array([[1, 0], [0, 0]])
            b_s = np.array([[0, 0], [1, 1]])
            # C00 += M2, C10 += M2
            w_s = np.array([[1, 0], [1, 0]])
            
        elif s == 3:
            # M3 = (A11)(B01)
            a_s = np.array([[0, 0], [0, 1]])
            b_s = np.array([[0, 1], [0, 0]])
            # C01 -= M3, C10 -= M3
            w_s = np.array([[0, -1], [-1, 0]])
            
        elif s == 4:
            # M4 = (A00+A01)(B11)
            a_s = np.array([[1, 1], [0, 0]])
            b_s = np.array([[0, 0], [0, 1]])
            # C00 += M4, C01 -= M4
            w_s = np.array([[1, -1], [0, 0]])
            
        elif s == 5:
            # M5 = (A10-A11)(B01+B11)
            a_s = np.array([[0, 0], [1, -1]])
            b_s = np.array([[0, 1], [0, 1]])
            # C01 -= M5, C10 += M5
            w_s = np.array([[0, -1], [1, 0]])
            
        elif s == 6:
            # M6 = (A01-A00)(B00+B10)
            a_s = np.array([[-1, 1], [0, 0]])
            b_s = np.array([[1, 0], [1, 0]])
            # C11 += M6
            w_s = np.array([[0, 0], [0, 1]])
        
        # Now, for this s, we have a 2x2 matmul of the linear combinations.
        # We use the 2x2 Strassen decomposition (U2, V2, W2) to decompose this.
        # The "A" matrix for this sub-problem is A_s.
        # The "B" matrix is B_s.
        # The "C" matrix is W_s * (A_s B_s).
        
        # We iterate over the 7 scalar multiplications t.
        for t in range(7):
            c_idx = s * 7 + t
            
            # U2[t] gives the coefficients for the 2x2 A matrix.
            # U2 is indexed by (i', j') where i',j' in {0,1}.
            # U2[i'*2+j', t] is the coefficient of A_{i',j'} in the t-th multiplication.
            
            # We need to map this to the 4x4 indices.
            # A_{I,J} is a 2x2 block.
            # A_{I,J}[i', j'] is at index (I*2+i')*4 + (J*2+j').
            
            for I in range(2):
                for J in range(2):
                    a_coeff = a_s[I, J]
                    if a_coeff == 0:
                        continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            u_coeff = U2[i_prime * 2 + j_prime, t]
                            if u_coeff == 0:
                                continue
                            i = I * 2 + i_prime
                            j = J * 2 + j_prime
                            U4[i * 4 + j, c_idx] += a_coeff * u_coeff
            
            for J in range(2):
                for K in range(2):
                    b_coeff = b_s[J, K]
                    if b_coeff == 0:
                        continue
                    for j_prime in range(2):
                        for k_prime in range(2):
                            v_coeff = V2[j_prime * 2 + k_prime, t]
                            if v_coeff == 0:
                                continue
                            j = J * 2 + j_prime
                            k = K * 2 + k_prime
                            V4[j * 4 + k, c_idx] += b_coeff * v_coeff
            
            for K in range(2):
                for I in range(2):
                    w_coeff = w_s[K, I]
                    if w_coeff == 0:
                        continue
                    for k_prime in range(2):
                        for i_prime in range(2):
                            w2_coeff = W2[k_prime * 2 + i_prime, t]
                            if w2_coeff == 0:
                                continue
                            k = K * 2 + k_prime
                            i = I * 2 + i_prime
                            W4[k * 4 + i, c_idx] += w_coeff * w2_coeff
    
    return U4, V4, W4

def construct():
    start_time = time.time()
    
    # Get the 4x4x4 decomposition
    U4, V4, W4 = _construct_strassen_4x4x4()
    
    # Extract the 3x4x4 sub-decomposition
    # n=3, m=4, p=4
    # U: (3*4, 49) -> (12, 49)
    # V: (4*4, 49) -> (16, 49)
    # W: (4*3, 49) -> (12, 49)
    
    # U4 index: i*4+j for i in 0..3, j in 0..3.
    # We want i in 0..2.
    U = U4[:12, :]
    
    # V4 index: j*4+k for j in 0..3, k in 0..3.
    # We want all j, k.
    V = V4
    
    # W4 index: k*4+i for k in 0..3, i in 0..3.
    # We want i in 0..2.
    # W4 row index is k*4+i.
    # We need to select rows where i < 3.
    # i = row % 4.
    W_rows = []
    for k in range(4):
        for i in range(3):
            W_rows.append(k * 4 + i)
    W = W4[W_rows, :]
    
    # Convert to list of lists of floats
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Validate
    if _validate(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
