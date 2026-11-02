"""Program for 3x5x5 matrix multiplication tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a tuple of 7 (i, j, k, coeff) triples.
    Using trace convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    For 2x2x2: n=m=p=2, so U row = i*2+j, V row = j*2+k, W row = k*2+i.
    """
    # Strassen algorithm for C = A * B where A, B, C are 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # c00 = (a00+a11)(b00+b11) - (a10+a11)b10 - (a00-a10)b00
    # c01 = a11*(b01-b10) + (a00+a11)*b10
    # c10 = (a10-a01)*b01 + (a00+a01)*a11... 
    # Let me use the standard 7 multiplications:
    # p0 = (a00+a11)(b00+b11)
    # p1 = (a10+a11)b00
    # p2 = a00(b01-b10)
    # p3 = a11(b10-b00)
    # p4 = (a00+a01)b11
    # p5 = (a10-a01)(b00+b01)
    # p6 = (a01-a11)(b10+b11)
    
    # C[0,0] = p0 + p3 - p1 - p6
    # C[0,1] = p2 + p4
    # C[1,0] = p1 + p5
    # C[1,1] = p0 + p2 - p4 - p5
    
    terms = []
    # p0 = (a00+a11)(b00+b11)
    # U: a00, a11 -> U rows 0, 3 ; V: b00, b11 -> V rows 0, 3
    terms.append([
        (0, 0, 0, 1),    # a00*b00 -> c00
        (0, 0, 1, -1),   # a00*b00 -> c01 ... wait, this isn't right
    ])
    
    # Let me think more carefully. In the tensor decomposition for 2x2x2:
    # T[i,j,k] = C[i,k] where C = A * B, i.e., C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] which equals 1 if we think of it as the identity
    # decomposition, but for rank-1 terms:
    # T = sum_r u_r ⊗ v_r ⊗ w_r where u_r[i] = A[i], v_r[j] = B[j], w_r[k] = C[k]
    
    # Actually, the tensor decomposition for matrix multiplication M_{n,m,p} has:
    # M[i,j,k] = δ_{ik} for the schoolbook (one term per (i,j,k))
    # For rank-1 term r: contribution is U[i,j] * V[j,k] * W[k,i]
    
    # So for 2x2x2, schoolbook has 8 terms: (i,j,k) for all combinations.
    # Strassen reduces this to 7 terms with some having coefficient -1 or +1.
    
    # Let me just use a known explicit decomposition.
    # The 7 Strassen factors for 2x2x2 (with n=m=p=2):
    # Each factor r has nonzero entries at specific (i,j,k) positions.
    
    # Using the convention T = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # and the tensor T[i,j,k] = 1 (it's the identity in the sense that
    # the contraction sum_j T[i,j,k] B[j,k] gives C[i,k] when T is the standard matmul tensor)
    
    # Actually T[i,j,k] should be such that C = A*B is recovered.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that would give sum_j A[i,j]*B[j,k]*1 which is correct!
    # Wait, the tensor for matrix multiplication is indeed all ones? No...
    # T[i,j,k] = 1 means C[i,k] = sum_j A[i,j]*B[j,k]*1 = sum_j A[i,j]*B[j,k]. Yes!
    
    # So the standard tensor M_{n,m,p}[i,j,k] = 1 for all i,j,k.
    # And we want to write it as sum_r U_r[i,j]*V_r[j,k]*W_r[k,i]
    
    # Schoolbook: rank = n*m*p, each term r=(i,j,k) has U[i,j]=1, V[j,k]=1, W[k,i]=1, others 0.
    
    # Strassen for 2x2x2 finds 7 rank-1 terms with coefficients in {-1, 0, 1}.
    
    # Let me look up or derive the Strassen decomposition.
    # For 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-g)
    # p4 = d(g-h)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # C[0,0] = p1 + p4 - p2 - p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p6
    # C[1,1] = p1 + p3 - p5 - p6
    
    # Now let's verify: 
    # C[0,0] should be ae+bg
    # p1 + p4 - p2 - p7 = (a+d)(e+h) + d(g-h) - (c+d)e - (b-d)(g+h)
    # = (ae+ah+de+dh) + (dg-dh) - (ce+de) - (bg+bh-dg-dh)
    # = ae+ah+de+dh+dg-dh-ce-de-bg-bh+dg+dh
    # = ae+ah+dg-dh-ce+dg+dh-bg-bh  ... let me redo this carefully
    
    # (a+d)(e+h) = ae + ah + de + dh
    # d(g-h) = dg - dh
    # (c+d)e = ce + de
    # (b-d)(g+h) = bg + bh - dg - dh
    
    # Sum: ae + ah + de + dh + dg - dh - ce - de - bg - bh + dg + dh
    # = ae + ah + (de - de) + (dh - dh + dh) + (dg + dg) - ce - bg - bh
    # = ae + ah + dh + 2dg - ce - bg - bh
    
    # This should equal ae + bg. Let me check: ae + ah + dh + 2dg - ce - bg - bh
    # Hmm, this doesn't look right. Let me use the standard Strassen formulas more carefully.
    
    # Standard Strassen (from Wikipedia):
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-g)
    # p4 = d(g-h)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # 
    # c11 = p1 + p4 - p2 - p7
    # c12 = p3 + p5
    # c21 = p2 + p6
    # c22 = p1 + p3 - p5 - p6
    
    # Verify c11: p1 + p4 - p2 - p7
    # = (a+d)(e+h) + d(g-h) - (c+d)e - (b-d)(g+h)
    # = (ae+ah+de+dh) + (dg-dh) - (ce+de) - (bg+bh-dg-dh)
    # = ae+ah+de+dh+dg-dh-ce-de-bg-bh+dg+dh
    # = ae + ah + (de-de) + (dh-dh+dh) + (dg+dg) - ce - bg - bh
    # = ae + ah + dh + 2dg - ce - bg - bh
    
    # This should equal ae+bg. Let's test with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # ae+bg = 5+14 = 19
    # Formula: ae + ah + dh + 2dg - ce - bg - bh = 5 + 8 + 32 + 2*28 - 15 - 14 - 16 = 5+8+32+56-15-14-16 = 56
    # That's not 19. Something is wrong with my expansion.
    
    # Let me redo: (b-d)(g+h) = (2-4)(7+8) = -2*15 = -30
    # p7 = -30, so -p7 = 30
    # c11 = p1+p4-p2-p7 = (1+4)(5+8) + 4*(7-8) - (3+4)*5 - (-30)
    # = 5*13 + 4*(-1) - 7*5 + 30 = 65 - 4 - 35 + 30 = 56
    # But actual c11 = a*e + b*g = 1*5 + 2*7 = 5+14 = 19
    # 56 ≠ 19. The formula I'm using is incorrect.
    
    # Let me use a different formulation. The issue is I may have the wrong Strassen formulas.
    # Let me just use a verified decomposition programmatically.
    # 
    # For now, I'll skip Strassen and use a better approach: 
    # Use rank-23 for 3x3x3 (if I can find it) and schoolbook for the rest.
    # Actually, let me just focus on getting a valid low-rank decomposition.
    
    return None

def construct():
    n, m, p = 3, 5, 5
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook decomposition
    rank = n * m * p  # 75
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify
    A = np.random.rand(3, 5)
    B = np.random.rand(5, 5)
    C_expected = A @ B
    C_calc = np.zeros((3, 5))
    for r in range(rank):
        for i in range(3):
            for k in range(5):
                s = 0
                for j in range(5):
                    s += U[i*5+j][r] * V[j*5+k][r]
                C_calc[i,k] += s * W[k*3+i][r]
    
    if not np.allclose(C_calc, C_expected):
        raise ValueError("Decomposition failed verification")
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # Would have been caught by deadline check
    
    return {"U": U, "V": V, "W": W}
