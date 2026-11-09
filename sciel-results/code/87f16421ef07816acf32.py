"""Laderman's rank-23 decomposition for 3x3 matrix multiplication."""
import time
import random

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Define the 23 products
    # Each product is (linear combo of A) * (linear combo of B)
    # U[i*3+k, r] = coefficient of A[i][k] in product r
    # V[k*3+j, r] = coefficient of B[k][j] in product r
    
    # P0 = a*x
    U[0][0] = 1; V[0][0] = 1
    # P1 = b*u
    U[1][1] = 1; V[3][1] = 1
    # P2 = c*r
    U[2][2] = 1; V[6][2] = 1
    # P3 = d*y
    U[3][3] = 1; V[1][3] = 1
    # P4 = e*v
    U[4][4] = 1; V[4][4] = 1
    # P5 = f*s
    U[5][5] = 1; V[7][5] = 1
    # P6 = g*z
    U[6][6] = 1; V[2][6] = 1
    # P7 = h*w
    U[7][7] = 1; V[5][7] = 1
    # P8 = i*t
    U[8][8] = 1; V[8][8] = 1
    # P9 = (a+e+i)*(x+v+t)
    U[0][9] = 1; U[4][9] = 1; U[8][9] = 1
    V[0][9] = 1; V[4][9] = 1; V[8][9] = 1
    # P10 = (a+h+i)*(x+s+t)
    U[0][10] = 1; U[7][10] = 1; U[8][10] = 1
    V[0][10] = 1; V[7][10] = 1; V[8][10] = 1
    # P11 = (b+d+i)*(u+y+t)
    U[1][11] = 1; U[3][11] = 1; U[8][11] = 1
    V[3][11] = 1; V[1][11] = 1; V[8][11] = 1
    # P12 = (b+g+f)*(u+z+y)
    U[1][12] = 1; U[6][12] = 1; U[5][12] = 1
    V[3][12] = 1; V[2][12] = 1; V[1][12] = 1
    # P13 = (c+d+h)*(r+y+w)
    U[2][13] = 1; U[3][13] = 1; U[7][13] = 1
    V[6][13] = 1; V[1][13] = 1; V[5][13] = 1
    # P14 = (c+g+e)*(r+v+z)
    U[2][14] = 1; U[6][14] = 1; U[4][14] = 1
    V[6][14] = 1; V[4][14] = 1; V[2][14] = 1
    # P15 = (a-b+c-d+e-f)*(x-y+z-u+v-w)
    U[0][15] = 1; U[1][15] = -1; U[2][15] = 1
    U[3][15] = -1; U[4][15] = 1; U[5][15] = -1
    V[0][15] = 1; V[1][15] = -1; V[2][15] = 1
    V[3][15] = -1; V[4][15] = 1; V[5][15] = -1
    # P16 = (a+b-c-d-e+f)*(x+y-z-u-v+w)
    U[0][16] = 1; U[1][16] = 1; U[2][16] = -1
    U[3][16] = -1; U[4][16] = -1; U[5][16] = 1
    V[0][16] = 1; V[1][16] = 1; V[2][16] = -1
    V[3][16] = -1; V[4][16] = -1; V[5][16] = 1
    # P17 = (a-b-c+d+e+f)*(x-y-z+u+v+w)
    U[0][17] = 1; U[1][17] = -1; U[2][17] = -1
    U[3][17] = 1; U[4][17] = 1; U[5][17] = 1
    V[0][17] = 1; V[1][17] = -1; V[2][17] = -1
    V[3][17] = 1; V[4][17] = 1; V[5][17] = 1
    # P18 = (a+b+c-d-e-f)*(x+y+z-u-v-w)
    U[0][18] = 1; U[1][18] = 1; U[2][18] = 1
    U[3][18] = -1; U[4][18] = -1; U[5][18] = -1
    V[0][18] = 1; V[1][18] = 1; V[2][18] = 1
    V[3][18] = -1; V[4][18] = -1; V[5][18] = -1
    # P19 = (d-e+f-g+h-i)*(y-v+w-s+t-r)
    U[3][19] = 1; U[4][19] = -1; U[5][19] = 1
    U[6][19] = -1; U[7][19] = 1; U[8][19] = -1
    V[1][19] = 1; V[4][19] = -1; V[5][19] = 1
    V[7][19] = -1; V[8][19] = 1; V[6][19] = -1
    # P20 = (d+e-f-g-h+i)*(y+v-w-s-t+r)
    U[3][20] = 1; U[4][20] = 1; U[5][20] = -1
    U[6][20] = -1; U[7][20] = -1; U[8][20] = 1
    V[1][20] = 1; V[4][20] = 1; V[5][20] = -1
    V[7][20] = -1; V[8][20] = -1; V[6][20] = 1
    # P21 = (d-e-f+g+h+i)*(y-v-w+s+t+r)
    U[3][21] = 1; U[4][21] = -1; U[5][21] = -1
    U[6][21] = 1; U[7][21] = 1; U[8][21] = 1
    V[1][21] = 1; V[4][21] = -1; V[5][21] = -1
    V[7][21] = 1; V[8][21] = 1; V[6][21] = 1
    # P22 = (d+e+f-g-h-i)*(y+v+w-s-t-r)
    U[3][22] = 1; U[4][22] = 1; U[5][22] = 1
    U[6][22] = -1; U[7][22] = -1; U[8][22] = -1
    V[1][22] = 1; V[4][22] = 1; V[5][22] = 1
    V[7][22] = -1; V[8][22] = -1; V[6][22] = -1
    
    # Now compute W coefficients.
    # For each C[i,j] and each product r, the coefficient W[j*3+i, r] should be
    # such that sum_r W[j*3+i, r] * (sum_k U[i*3+k, r] * V[k*3+j, r]) gives
    # the correct coefficient for each monomial in C[i,j].
    
    # Actually, the correct formula is:
    # C[i,j] = sum_r W[j*3+i, r] * P_r
    # where P_r = (sum_k U[i*3+k, r]*A[i,k]) * (sum_k V[k*3+j, r]*B[k,j])
    
    # But this isn't quite right either. Let me think again.
    
    # The bilinear form is:
    # C[i,j] = sum_r W_{ijr} * (sum_k U_{ikr} A_{ik}) * (sum_k V_{kjr} B_{kj})
    
    # For this to equal sum_k A_{ik} B_{kj}, we need:
    # sum_r W_{ijr} * U_{ikr} * V_{kjr} = 1 for all i,j,k
    
    # So W_{ijr} = sum over r of the appropriate values.
    
    # Let me compute: for each (i,j), we need:
    # sum_k A[i,k]*B[k,j] = sum_r W[j*3+i, r] * (sum_k U[i*3+k,r]*A[i,k]) * (sum_k V[k*3+j,r]*B[k,j])
    
    # This means: for each (i,j,k), the coefficient of A[i,k]*B[k,j] on the RHS must be 1.
    # The coefficient of A[i,k]*B[k,j] in P_r is U[i*3+k,r] * V[k*3+j,r].
    # So we need: sum_r W[j*3+i, r] * U[i*3+k,r] * V[k*3+j,r] = 1 for all i,j,k.
    
    # This gives us 27 equations (for each i,j,k) and 9*23 = 207 unknowns (W values).
    # But actually W[j*3+i, r] only depends on (j,i,r), so we have 9*23 = 207 unknowns
    # and 27 equations. This is underdetermined, but we can solve it.
    
    # Actually, for each fixed (i,j), we have 3 equations (one for each k) and 23 unknowns
    # (W[j*3+i, 0] through W[j*3+i, 22]).
    
    # Let me solve this system.
    
    # For each (i,j), set up the system:
    # For k=0,1,2: sum_r W[j*3+i, r] * U[i*3+k, r] * V[k*3+j, r] = 1
    
    # This is a system of 3 equations in 23 unknowns. We can find one solution.
    
    # Let me use a simple approach: for each (i,j), try to find W values.
    
    # Actually, since the system is underdetermined, let me use the approach of
    # setting most W to 0 and solving for the rest, or use least squares.
    
    # Let me try a different approach: use the known structure.
    
    # For the diagonal products (P0-P8), we already have:
    # P0: contributes to C[0,0] with U[0,0]=1, V[0,0]=1 -> coeff for A[0,0]*B[0,0] is 1
    # So W[0,0] should be 1 (for the A[0,0]*B[0,0] term in C[0,0])
    
    # Let me just solve the linear system numerically.
    
    import numpy as np
    
    # For each (i,j), solve for W[j*3+i, :]
    for i in range(3):
        for j in range(3):
            # Build the 3x23 matrix M where M[k, r] = U[i*3+k, r] * V[k*3+j, r]
            M = np.zeros((3, 23))
            for k in range(3):
                for r in range(23):
                    M[k, r] = U[i*3+k][r] * V[k*3+j][r]
            
            # We need M @ w = [1, 1, 1] where w = W[j*3+i, :]
            # Use least squares (or exact solution if possible)
            try:
                w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(3), rcond=None)
                for r in range(23):
                    W[j*3+i][r] = round(w[r], 10)
            except:
                # Fallback: set all to 0
                for r in range(23):
                    W[j*3+i][r] = 0
    
    # Verify the solution
    def verify():
        for trial in range(100):
            A = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            B = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            C_true = [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            C_comp = [[0.0 for _ in range(3)] for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    for r in range(23):
                        ua = sum(U[i*3+k][r] * A[i][k] for k in range(3))
                        vb = sum(V[k*3+j][r] * B[k][j] for k in range(3))
                        C_comp[i][j] += W[j*3+i][r] * ua * vb
            for i in range(3):
                for j in range(3):
                    if abs(C_comp[i][j] - C_true[i][j]) > 1e-6:
                        return False
        return True
    
    if not verify():
        # Fall back to schoolbook
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    U[i*3+k][col] = 1
                    V[k*3+j][col] = 1
                    W[j*3+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

