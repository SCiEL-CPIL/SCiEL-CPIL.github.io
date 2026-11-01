"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that the factors exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.default_rng(42)
    for _ in range(5):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    acc += W[kk * n + ii, cc] * sum(U[ii * m + jj, cc] * A[ii, jj] for jj in range(m)) * sum(V[jj * p + kk, cc] * B[jj, kk] for jj in range(m))
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=tol):
            return False
    return True

def _strassen_2x2(U_base, V_base, W_base, n, m, p):
    """
    Build Strassen decomposition for 2x2 block multiplication.
    A is (n x m) = (2 x 4), B is (m x p) = (4 x 4).
    Split into 2x2 blocks where each block is (1x2) x (2x2) -> (1x2).
    Actually, let's think of this differently.
    
    A = [[A11, A12], [A21, A22]] where each Aij is 1x2
    B = [[B11, B12], [B21, B22]] where each Bji is 2x2
    C = [[C11, C12], [C21, C22]] where each Cki is 1x2
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each Cki = sum of two (1x2)*(2x2) products = (1x2) result.
    Each such product has rank 2 (since (1x2)*(2x2) = sum of 2 rank-1 terms).
    So naively each Cki needs rank 2, total 8*2 = 16? No wait...
    
    Actually, let me reconsider. The tensor for (1x2)*(2x2)->(1x2) has:
    C[k] = sum_j A[j]*B[j,k] for j=0,1 and k=0,1
    This is just a 2-dimensional output with 2*2=4 input dimensions.
    The rank of this small tensor is 2 (it's just two independent dot products of length 2).
    
    So for each of the 4 output blocks, we need rank 2, total rank 8.
    But that's not right either because the blocks share structure.
    
    Let me just build it directly.
    """
    pass

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Start with the structural rank decomposition (rank = n*m*p = 32)
    rank = n * m * p
    idx = np.arange(rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    i = (idx // (m * p)) % n
    j = (idx // p) % m
    k = idx % p
    U[i * m + j, idx] = 1
    V[j * p + k, idx] = 1
    W[k * n + i, idx] = 1
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try Strassen-based decomposition
    # A is 2x4, B is 4x4, C is 2x4
    # Split rows of A into 2 groups of 1, columns of B into 2 groups of 2
    # A = [A0; A1] where A0, A1 are 1x4
    # B = [B00 B01; B10 B11] where each Bji is 2x2
    # C = [C0; C1] where C0, C1 are 1x4
    # C0 = A0*B00 + A0*B10 ... wait, that's not right.
    # C0[k] = sum_j A0[j]*B[j,k] for j=0..3
    # = sum_j A0[j]*B[j,k]
    # If we split B into top 2 rows and bottom 2 rows:
    # C0[k] = sum_{j=0}^{1} A0[j]*B[j,k] + sum_{j=2}^{3} A0[j]*B[j,k]
    # = A0_top * B_top[:,k] + A0_bot * B_bot[:,k]
    
    # Let me think of it as: A is 2x4, split into left 2 cols and right 2 cols
    # A = [A_L | A_R] where A_L, A_R are 2x2
    # B is 4x4, split into top 2 rows and bottom 2 rows
    # B = [B_T; B_B] where B_T, B_B are 2x4
    # C = A_L * B_T + A_R * B_B, where each product is 2x4
    
    # Now A_L * B_T is (2x2)*(2x4) -> 2x4
    # This can be done with Strassen on the 2x2 part:
    # Let A_L = [[a,b],[c,d]], B_T = [b11 b12 b13 b14; b21 b22 b23 b24]
    # Result row 0: a*b11 + b*b21, a*b12 + b*b22, a*b13 + b*b23, a*b14 + b*b24
    # Result row 1: c*b11 + d*b21, c*b12 + d*b22, c*b13 + d*b23, c*b14 + d*b24
    
    # Using Strassen for the 2x2 matrix multiply part, but B_T has 4 columns.
    # We can apply Strassen to each pair of columns, but that's not efficient.
    # Instead, note that A_L * B_T = [a*b11+b*b21, a*b12+b*b22, a*b13+b*b23, a*b14+b*b24;
    #                                 c*b11+d*b21, c*b12+d*b22, c*b13+d*b23, c*b14+d*b24]
    
    # The 7 Strassen products for 2x2*2x2 are:
    # M1 = (a+b)*(e+f) where e,f are from B
    # But here B has 4 columns, so let's define:
    # For the top 2 rows of B (which is 2x4), let's call the columns b1, b2, b3, b4 (each 2x1)
    # A_L * B_T gives a 2x4 matrix.
    
    # Actually, let's just use the standard rank-7 Strassen decomposition for 2x2*2x2.
    # A_L is 2x2, and we multiply by each 2x2 submatrix of B_T (there are 2 such: columns 0-1 and 2-3).
    # So A_L * B_T = [A_L * B_T[:,0:2] | A_L * B_T[:,2:4]]
    # Each of these is a 2x2*2x2 multiply, which has rank 7.
    # So A_L * B_T has rank 7+7=14? No, the 7 products can be shared if we're clever.
    
    # Actually, the 7 Strassen products are:
    # M1 = (a+b)*(col1+col2) where col1, col2 are the two columns of the 2x2 B block
    # But since B_T has 4 columns, we have two separate 2x2 blocks.
    # So we need 7 products for the first block and 7 for the second = 14.
    # Similarly for A_R * B_B = 14.
    # Total = 28. That matches the expected improvement!
    
    # Let's build this.
    # A = [A_L | A_R], A_L = [[a,b],[c,d]], A_R = [[e,f],[g,h]]
    # B = [B_T; B_B], B_T = 2x4, B_B = 2x4
    # B_T = [t11 t12 t13 t14; t21 t22 t23 t24]
    # B_B = [b11 b12 b13 b14; b21 b22 b23 b24]
    
    # C = A_L*B_T + A_R*B_B
    
    # A_L*B_T using Strassen (7 products):
    # M1 = (a+b)*(t11+t21) ... wait, Strassen is for full 2x2*2x2.
    # Let B_T_left = [t11 t12; t21 t22] (columns 0,1)
    # B_T_right = [t13 t14; t23 t24] (columns 2,3)
    # A_L * B_T_left: 7 Strassen products
    # A_L * B_T_right: 7 Strassen products
    
    # Standard Strassen for [[a,b],[c,d]] * [[x,y],[z,w]] = [[p,q],[r,s]]:
    # M1 = (a+b)*(z+w)
    # M2 = (c+d)*x
    # M3 = a*(y-w)
    # M4 = d*(z+x)
    # M5 = (a+c)*w
    # M6 = (b-d)*(x+z)
    # M7 = (c-a)*(y+w)
    # p = M1+M4-M5+M7
    # q = M3+M5
    # r = M2+M4
    # s = M1-M2+M3+M6
    
    # For A_L * B_T_left where B_T_left = [[t11,t12],[t21,t22]]:
    # a=t11? No, a,b,c,d are from A_L.
    # A_L = [[a,b],[c,d]], B_T_left = [[t11,t12],[t21,t22]]
    # M1 = (a+b)*(t21+t22)
    # M2 = (c+d)*t11
    # M3 = a*(t12-t22)
    # M4 = d*(t21+t11)
    # M5 = (a+c)*t22
    # M6 = (b-d)*(t11+t21)
    # M7 = (c-a)*(t12+t22)
    # Result[0,0] = M1+M4-M5+M7
    # Result[0,1] = M3+M5
    # Result[1,0] = M2+M4
    # Result[1,1] = M1-M2+M3+M6
    
    # Similarly for A_L * B_T_right with B_T_right = [[t13,t14],[t23,t24]]:
    # M8 = (a+b)*(t23+t24)
    # M9 = (c+d)*t13
    # M10 = a*(t14-t24)
    # M11 = d*(t23+t13)
    # M12 = (a+c)*t24
    # M13 = (b-d)*(t13+t23)
    # M14 = (c-a)*(t14+t24)
    # Result[0,2] = M8+M11-M12+M14
    # Result[0,3] = M10+M12
    # Result[1,2] = M9+M11
    # Result[1,3] = M8-M9+M10+M13
    
    # For A_R * B_B where A_R = [[e,f],[g,h]], B_B = [[b11,b12],[b21,b22]] (left) and [[b13,b14],[b23,b24]] (right):
    # M15 = (e+f)*(b21+b22)
    # M16 = (g+h)*b11
    # M17 = e*(b12-b22)
    # M18 = h*(b21+b11)
    # M19 = (e+g)*b22
    # M20 = (f-h)*(b11+b21)
    # M21 = (g-e)*(b12+b22)
    # Result[0,0] += M15+M18-M19+M21
    # Result[0,1] += M17+M19
    # Result[1,0] += M16+M18
    # Result[1,1] += M15-M16+M17+M20
    
    # M22 = (e+f)*(b23+b24)
    # M23 = (g+h)*b13
    # M24 = e*(b14-b24)
    # M25 = h*(b23+b13)
    # M26 = (e+g)*b24
    # M27 = (f-h)*(b13+b23)
    # M28 = (g-e)*(b14+b24)
    # Result[0,2] += M22+M25-M26+M28
    # Result[0,3] += M24+M26
    # Result[1,2] += M23+M25
    # Result[1,3] += M22-M23+M24+M27
    
    # Total: 28 products. Now I need to express each product as U*V*W factors.
    
    # U factors (for A): each M involves a linear combination of A entries.
    # V factors (for B): each M involves a linear combination of B entries.
    # W factors (for C): each C entry is a linear combination of M's.
    
    # Let me define the 28 products and their U, V, W coefficients.
    # Product c has U_c (2x4 vector), V_c (4x4 vector), W_c (4x2 vector)
    # where U_c[i*4+j] = coefficient of A[i,j], V_c[j*4+k] = coefficient of B[j,k], W_c[k*2+i] = coefficient of C[i,k]
    
    # Actually, the convention is:
    # C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # with n=2, m=4, p=4.
    # So U has shape (2*4, 28) = (8, 28)
    # V has shape (4*4, 28) = (16, 28)
    # W has shape (4*2, 28) = (8, 28)
    
    # Let me build this systematically.
    
    U_str = np.zeros((8, 28), dtype=float)
    V_str = np.zeros((16, 28), dtype=float)
    W_str = np.zeros((8, 28), dtype=float)
    
    # Helper: set U for a product that uses A entries
    # A indices: A[0,0]=a, A[0,1]=b, A[0,2]=e, A[0,3]=f, A[1,0]=c, A[1,1]=d, A[1,2]=g, A[1,3]=h
    # U[i*4+j, c] = coeff of A[i,j]
    
    # Helper: set V for a product that uses B entries
    # B indices: B[0,0]=t11, B[0,1]=t12, B[0,2]=t13, B[0,3]=t14, B[1,0]=t21, B[1,1]=t22, B[1,2]=t23, B[1,3]=t24, B[2,0]=b11, B[2,1]=b12, B[2,2]=b13, B[2,3]=b14, B[3,0]=b21, B[3,1]=b22, B[3,2]=b23, B[3,3]=b24
    # V[j*4+k, c] = coeff of B[j,k]
    
    # Helper: set W for output C entries
    # C indices: C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], C[1,1], C[1,2], C[1,3]
    # W[k*2+i, c] = coeff of C[i,k]
    
    # Product 1: M1 = (a+b)*(t21+t22)
    # U: a=1, b=1 -> U[0*4+0]=1, U[0*4+1]=1
    U_str[0, 0] = 1; U_str[1, 0] = 1
    # V: t21=1, t22=1 -> V[1*4+0]=1, V[1*4+1]=1
    V_str[4, 0] = 1; V_str[5, 0] = 1
    # W: contributes to C[0,0] and C[1,1]
    # C[0,0] += M1, C[1,1] += M1
    W_str[0*2+0, 0] = 1  # C[0,0]
    W_str[1*2+1, 0] = 1  # C[1,1]
    
    # Product 2: M2 = (c+d)*t11
    U_str[4, 1] = 1; U_str[5, 1] = 1  # c=A[1,0], d=A[1,1]
    V_str[0, 1] = 1  # t11=B[0,0]
    W_str[1*2+0, 1] = 1  # C[1,0] += M2
    W_str[1*2+1, 1] = -1  # C[1,1] -= M2
    
    # Product 3: M3 = a*(t12-t22)
    U_str[0, 2] = 1  # a=A[0,0]
    V_str[1, 2] = 1; V_str[5, 2] = -1  # t12=B[0,1], t22=B[1,1]
    W_str[0*2+1, 2] = 1  # C[0,1] += M3
    W_str[1*2+1, 2] = 1  # C[1,1] += M3
    
    # Product 4: M4 = d*(t21+t11)
    U_str[5, 3] = 1  # d=A[1,1]
    V_str[4, 3] = 1; V_str[0, 3] = 1  # t21=B[1,0], t11=B[0,0]
    W_str[0*2+0, 3] = 1  # C[0,0] += M4
    W_str[1*2+0, 3] = 1  # C[1,0] += M4
    
    # Product 5: M5 = (a+c)*t22
    U_str[0, 4] = 1; U_str[4, 4] = 1  # a=A[0,0], c=A[1,0]
    V_str[5, 4] = 1  # t22=B[1,1]
    W_str[0*2+0, 4] = -1  # C[0,0] -= M5
    W_str[0*2+1, 4] = 1  # C[0,1] += M5
    
    # Product 6: M6 = (b-d)*(t11+t21)
    U_str[1, 5] = 1; U_str[5, 5] = -1  # b=A[0,1], d=A[1,1]
    V_str[0, 5] = 1; V_str[4, 5] = 1  # t11=B[0,0], t21=B[1,0]
    W_str[1*2+1, 5] = 1  # C[1,1] += M6
    
    # Product 7: M7 = (c-a)*(t12+t22)
    U_str[4, 6] = 1; U_str[0, 6] = -1  # c=A[1,0], a=A[0,0]
    V_str[1, 6] = 1; V_str[5, 6] = 1  # t12=B[0,1], t22=B[1,1]
    W_str[0*2+0, 6] = 1  # C[0,0] += M7
    
    # Product 8: M8 = (a+b)*(t23+t24)
    U_str[0, 7] = 1; U_str[1, 7] = 1
    V_str[6, 7] = 1; V_str[7, 7] = 1  # t23=B[1,2], t24=B[1,3]
    W_str[0*2+2, 7] = 1  # C[0,2] += M8
    W_str[1*2+3, 7] = 1  # C[1,3] += M8
    
    # Product 9: M9 = (c+d)*t13
    U_str[4, 8] = 1; U_str[5, 8] = 1
    V_str[2, 8] = 1  # t13=B[0,2]
    W_str[1*2+2, 8] = 1  # C[1,2] += M9
    W_str[1*2+3, 8] = -1  # C[1,3] -= M9
    
    # Product 10: M10 = a*(t14-t24)
    U_str[0, 9] = 1
    V_str[3, 9] = 1; V_str[7, 9] = -1  # t14=B[0,3], t24=B[1,3]
    W_str[0*2+3, 9] = 1  # C[0,3] += M10
    W_str[1*2+3, 9] = 1  # C[1,3] += M10
    
    # Product 11: M11 = d*(t23+t13)
    U_str[5, 10] = 1
    V_str[6, 10] = 1; V_str[2, 10] = 1  # t23=B[1,2], t13=B[0,2]
    W_str[0*2+2, 10] = 1  # C[0,2] += M11
    W_str[1*2+2, 10] = 1  # C[1,2] += M11
    
    # Product 12: M12 = (a+c)*t24
    U_str[0, 11] = 1; U_str[4, 11] = 1
    V_str[7, 11] = 1  # t24=B[1,3]
    W_str[0*2+2, 11] = -1  # C[0,2] -= M12
    W_str[0*2+3, 11] = 1  # C[0,3] += M12
    
    # Product 13: M13 = (b-d)*(t13+t23)
    U_str[1, 12] = 1; U_str[5, 12] = -1
    V_str[2, 12] = 1; V_str[6, 12] = 1  # t13=B[0,2], t23=B[1,2]
    W_str[1*2+3, 12] = 1  # C[1,3] += M13
    
    # Product 14: M14 = (c-a)*(t14+t24)
    U_str[4, 13] = 1; U_str[0, 13] = -1
    V_str[3, 13] = 1; V_str[7, 13] = 1  # t14=B[0,3], t24=B[1,3]
    W_str[0*2+2, 13] = 1  # C[0,2] += M14
    
    # Product 15: M15 = (e+f)*(b21+b22)
    U_str[2, 14] = 1; U_str[3, 14] = 1  # e=A[0,2], f=A[0,3]
    V_str[12, 14] = 1; V_str[13, 14] = 1  # b21=B[3,0], b22=B[3,1]
    W_str[0*2+0, 14] = 1  # C[0,0] += M15
    W_str[1*2+1, 14] = 1  # C[1,1] += M15
    
    # Product 16: M16 = (g+h)*b11
    U_str[6, 15] = 1; U_str[7, 15] = 1  # g=A[1,2], h=A[1,3]
    V_str[8, 15] = 1  # b11=B[2,0]
    W_str[1*2+0, 15] = 1  # C[1,0] += M16
    W_str[1*2+1, 15] = -1  # C[1,1] -= M16
    
    # Product 17: M17 = e*(b12-b22)
    U_str[2, 16] = 1
    V_str[9, 16] = 1; V_str[13, 16] = -1  # b12=B[2,1], b22=B[3,1]
    W_str[0*2+1, 16] = 1  # C[0,1] += M17
    W_str[1*2+1, 16] = 1  # C[1,1] += M17
    
    # Product 18: M18 = h*(b21+b11)
    U_str[7, 17] = 1
    V_str[12, 17] = 1; V_str[8, 17] = 1  # b21=B[3,0], b11=B[2,0]
    W_str[0*2+0, 17] = 1  # C[0,0] += M18
    W_str[1*2+0, 17] = 1  # C[1,0] += M18
    
    # Product 19: M19 = (e+g)*b22
    U_str[2, 18] = 1; U_str[6, 18] = 1
    V_str[13, 18] = 1  # b22=B[3,1]
    W_str[0*2+0, 18] = -1  # C[0,0] -= M19
    W_str[0*2+1, 18] = 1  # C[0,1] += M19
    
    # Product 20: M20 = (f-h)*(b11+b21)
    U_str[3, 19] = 1; U_str[7, 19] = -1
    V_str[8, 19] = 1; V_str[12, 19] = 1  # b11=B[2,0], b21=B[3,0]
    W_str[1*2+1, 19] = 1  # C[1,1] += M20
    
    # Product 21: M21 = (g-e)*(b12+b22)
    U_str[6, 20] = 1; U_str[2, 20] = -1
    V_str[9, 20] = 1; V_str[13, 20] = 1  # b12=B[2,1], b22=B[3,1]
    W_str[0*2+0, 20] = 1  # C[0,0] += M21
    
    # Product 22: M22 = (e+f)*(b23+b24)
    U_str[2, 21] = 1; U_str[3, 21] = 1
    V_str[14, 21] = 1; V_str[15, 21] = 1  # b23=B[3,2], b24=B[3,3]
    W_str[0*2+2, 21] = 1  # C[0,2] += M22
    W_str[1*2+3, 21] = 1  # C[1,3] += M22
    
    # Product 23: M23 = (g+h)*b13
    U_str[6, 22] = 1; U_str[7, 22] = 1
    V_str[10, 22] = 1  # b13=B[2,2]
    W_str[1*2+2, 22] = 1  # C[1,2] += M23
    W_str[1*2+3, 22] = -1  # C[1,3] -= M23
    
    # Product 24: M24 = e*(b14-b24)
    U_str[2, 23] = 1
    V_str[11, 23] = 1; V_str[15, 23] = -1  # b14=B[2,3], b24=B[3,3]
    W_str[0*2+3, 23] = 1  # C[0,3] += M24
    W_str[1*2+3, 23] = 1  # C[1,3] += M24
    
    # Product 25: M25 = h*(b23+b13)
    U_str[7, 24] = 1
    V_str[14, 24] = 1; V_str[10, 24] = 1  # b23=B[3,2], b13=B[2,2]
    W_str[0*2+2, 24] = 1  # C[0,2] += M25
    W_str[1*2+2, 24] = 1  # C[1,2] += M25
    
    # Product 26: M26 = (e+g)*b24
    U_str[2, 25] = 1; U_str[6, 25] = 1
    V_str[15, 25] = 1  # b24=B[3,3]
    W_str[0*2+2, 25] = -1  # C[0,2] -= M26
    W_str[0*2+3, 25] = 1  # C[0,3] += M26
    
    # Product 27: M27 = (f-h)*(b13+b23)
    U_str[3, 26] = 1; U_str[7, 26] = -1
    V_str[10, 26] = 1; V_str[14, 26] = 1  # b13=B[2,2], b23=B[3,2]
    W_str[1*2+3, 26] = 1  # C[1,3] += M27
    
    # Product 28: M28 = (g-e)*(b14+b24)
    U_str[6, 27] = 1; U_str[2, 27] = -1
    V_str[11, 27] = 1; V_str[15, 27] = 1  # b14=B[2,3], b24=B[3,3]
    W_str[0*2+2, 27] = 1  # C[0,2] += M28
    
    # Verify the Strassen decomposition
    if _verify_factors(U_str, V_str, W_str, n, m, p, tol=1e-9):
        best = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
        best_rank = 28
    
    return best
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
