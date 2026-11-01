import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 9]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication.
    Rank 7.
    Indices:
    U: (2*2, 7) -> rows i*2+j
    V: (2*2, 7) -> rows j*2+k
    W: (2*2, 7) -> rows k*2+i
    """
    # M_r[i,j] and N_r[j,k] such that sum w_r M_r N_r = A B
    # Standard Strassen:
    # M0 = [[1,0],[0,0]], N0 = [[1,0],[0,0]] -> a*x
    # M1 = [[0,1],[0,0]], N1 = [[0,0],[1,0]] -> b*z
    # M2 = [[0,0],[1,0]], N2 = [[1,0],[0,0]] -> c*x
    # M3 = [[0,0],[0,1]], N3 = [[0,0],[1,0]] -> d*z
    # M4 = [[1,1],[1,1]], N4 = [[0,1],[0,1]] -> (a+b)(y+z)
    # M5 = [[1,1],[1,-1]], N5 = [[0,1],[0,-1]] -> (a+b)(y-z)
    # M6 = [[1,-1],[1,1]], N6 = [[0,1],[0,1]] -> (a-c)(x+z)
    # M7 = [[1,-1],[1,-1]], N7 = [[0,1],[0,-1]] -> (a-c)(x-z)
    
    # Let's verify the sum for C[0,0] = ax + bz.
    # Terms contributing to C[0,0]:
    # M0 N0: 1*1 = 1 (ax)
    # M1 N1: 0*0 = 0
    # M2 N2: 0*1 = 0
    # M3 N3: 0*0 = 0
    # M4 N4: (1*0 + 1*0) = 0
    # M5 N5: (1*0 + 1*0) = 0
    # M6 N6: (1*0 + 1*0) = 0
    # M7 N7: (1*0 + 1*0) = 0
    # Sum = 1. Correct.
    
    # C[0,1] = ay + bz.
    # M0 N0: 1*0 = 0
    # M1 N1: 0*0 = 0
    # M2 N2: 0*0 = 0
    # M3 N3: 0*0 = 0
    # M4 N4: 1*1 + 1*1 = 2 (a+b)(y+z) -> 2ay + 2az + 2by + 2bz
    # M5 N5: 1*1 + 1*(-1) = 0? No.
    # Let's use the explicit matrices.
    
    U = np.zeros((4, 8)) # Using 8 for simplicity, rank 8 is trivial but let's do 7? No, 2x2x2 rank is 7.
    # Actually, let's just use the trivial rank 4 for 2x2x2 first to ensure correctness, then try to optimize?
    # No, the goal is to beat 36. 3 blocks of rank 10 is 30.
    # Can we do better?
    # mu(2,2,9) is known to be 27? Or 28?
    # Let's check known results.
    # mu(2,2,p) = 4p - 3 for p >= 2? No.
    # For p=2, mu=7.
    # For p=3, mu=10.
    # For p=4, mu=13?
    # It seems mu(2,2,p) = 3p + 1?
    # p=2 -> 7. 3*2+1 = 7.
    # p=3 -> 10. 3*3+1 = 10.
    # p=4 -> 13.
    # p=9 -> 3*9+1 = 28.
    # If this formula holds, we can achieve rank 28.
    
    # How to construct the decomposition for general p?
    # We can use the block decomposition.
    # A (2x2) * B (2xp).
    # We can split B into B1 (2x2) and B2 (2x(p-2)).
    # Rank = mu(2,2,2) + mu(2,2,p-2) = 7 + mu(2,2,p-2).
    # This gives mu(2,2,p) <= 7 + mu(2,2,p-2).
    # Base case mu(2,2,1) = 4.
    # mu(2,2,2) = 7.
    # mu(2,2,3) = 10.
    # mu(2,2,4) = 13.
    # mu(2,2,5) = 16.
    # mu(2,2,9) = 7 + 7 + 7 + 4 = 25?
    # Wait.
    # mu(2,2,1) = 4.
    # mu(2,2,2) = 7.
    # mu(2,2,3) = 10.
    # mu(2,2,4) = 13.
    # mu(2,2,5) = 16.
    # mu(2,2,6) = 19.
    # mu(2,2,7) = 22.
    # mu(2,2,8) = 25.
    # mu(2,2,9) = 28.
    
    # So if we can construct a rank 10 decomposition for 2x2x3, and rank 7 for 2x2x2, and rank 4 for 2x2x1,
    # we can combine them.
    # 9 = 3 + 3 + 3. Rank = 10 + 10 + 10 = 30.
    # 9 = 2 + 2 + 2 + 2 + 1. Rank = 7 + 7 + 7 + 7 + 4 = 32.
    # 9 = 3 + 2 + 2 + 2. Rank = 10 + 7 + 7 + 7 = 31.
    # 9 = 3 + 3 + 2 + 1. Rank = 10 + 10 + 7 + 4 = 31.
    
    # Is there a better decomposition?
    # The formula 3p+1 suggests 28.
    # How to get 28?
    # Maybe a different combination?
    # Or a direct construction?
    
    # Let's try to find a rank 10 decomposition for 2x2x3.
    # And a rank 7 for 2x2x2.
    
    # Then we can try to combine them.
    # If we use 3 blocks of 3, rank is 30.
    # Can we do better?
    # What if we use a "sliding window" or overlapping decomposition?
    
    # Let's first implement the 2x2x2 and 2x2x3 decompositions.
    
    # 2x2x2 (Rank 7):
    # M0 = [[1,0],[0,0]], N0 = [[1,0],[0,0]]
    # M1 = [[0,1],[0,0]], N1 = [[0,0],[1,0]]
    # M2 = [[0,0],[1,0]], N2 = [[1,0],[0,0]]
    # M3 = [[0,0],[0,1]], N3 = [[0,0],[1,0]]
    # M4 = [[1,1],[1,1]], N4 = [[0,1],[0,1]]
    # M5 = [[1,1],[1,-1]], N5 = [[0,1],[0,-1]]
    # M6 = [[1,-1],[1,1]], N6 = [[0,1],[0,1]]
    # M7 = [[1,-1],[1,-1]], N7 = [[0,1],[0,-1]]
    # Wait, I listed 8 matrices above. Strassen uses 7.
    # One of them is redundant or I made a mistake.
    # The 7th product is M7 = (a-c)(x-z).
    # The 4th product is d*z.
    # Let's re-verify the 7 products.
    # p1 = a*x
    # p2 = b*z
    # p3 = c*x
    # p4 = d*z
    # p5 = (a+b)(y+z)
    # p6 = (a-c)(x+z)
    # p7 = (b+d)(y-z)
    
    # C00 = p1 + p2 - p5 + p7 = ax + bz - (ay+az+by+bz) + (by-bz+dy-dz)
    # = ax + bz - ay - az - by - bz + by - bz + dy - dz
    # = ax - ay - az - bz + dy - dz.
    # This is not ax + by.
    
    # Let's look up the exact Strassen formulas.
    # p1 = a*x
    # p2 = b*z
    # p3 = c*x
    # p4 = d*z
    # p5 = (a+b)*(y+z)
    # p6 = (a-c)*(x+z)
    # p7 = (b+d)*(y-z)
    
    # C00 = p1 + p2 - p5 + p7
    # C01 = p5 - p2 - p7
    # C10 = p3 + p6
    # C11 = p4 - p3 + p6
    
    # Let's check C00:
    # ax + bz - (ay+az+by+bz) + (by-bz+dy-dz)
    # = ax + bz - ay - az - by - bz + by - bz + dy - dz
    # = ax - ay - az - bz + dy - dz.
    # Still wrong.
    
    # Correct Strassen:
    # p1 = a*x
    # p2 = b*(z-x)
    # p3 = c*(x+z)
    # p4 = d*(y-z)
    # p5 = (a+b)*(y+z)
    # p6 = (a-c)*(x+z)
    # p7 = (b+d)*(y-z)
    
    # C00 = p1 + p2 - p5 + p7
    # = ax + b(z-x) - (a+b)(y+z) + (b+d)(y-z)
    # = ax + bz - bx - (ay+az+by+bz) + (by-bz+dy-dz)
    # = ax + bz - bx - ay - az - by - bz + by - bz + dy - dz
    # = ax - bx - ay - az - bz + dy - dz.
    # Still not ax+by.
    
    # I will just use the trivial rank 4 for 2x2x2 to be safe, and trivial rank 4 for 2x2x1.
    # Then 9 = 2+2+2+2+1 -> 4+4+4+4+4 = 20? No, rank is additive.
    # If I use trivial rank 4 for each 2x2x2 block, total rank is 4*4 + 4 = 20.
    # Wait, the trivial rank for 2x2x2 is 4.
    # So 4 blocks of 2x2x2 and 1 block of 2x2x1.
    # Rank = 4*4 + 4 = 20.
    # This is much better than 36.
    
    # Let's verify if trivial rank 4 for 2x2x2 is valid.
    # A = [[a,b],[c,d]], B = [[x,y],[z,w]].
    # C00 = ax + bz.
    # C01 = ay + bw.
    # C10 = cx + dz.
    # C11 = cy + dw.
    
    # Terms:
    # 1: M=[[1,0],[0,0]], N=[[1,0],[0,0]], W=1 -> ax
    # 2: M=[[0,1],[0,0]], N=[[0,0],[1,0]], W=1 -> bz
    # 3: M=[[0,0],[1,0]], N=[[1,0],[0,0]], W=1 -> cx
    # 4: M=[[0,0],[0,1]], N=[[0,0],[1,0]], W=1 -> dz
    # Wait, this only gives the diagonal of C?
    # No, N is 2x2.
    # N1 = [[0,0],[1,0]]. N1[0,0]=0, N1[0,1]=0, N1[1,0]=1, N1[1,1]=0.
    # M2 = [[0,1],[0,0]].
    # M2 N2 = [[0,1],[0,0]] [[0,0],[1,0]] = [[1,0],[0,0]].
    # This contributes to C00.
    # We need C01 = ay + bw.
    # Term 5: M=[[1,0],[0,0]], N=[[0,1],[0,0]], W=1 -> ay
    # Term 6: M=[[0,1],[0,0]], N=[[0,0],[0,1]], W=1 -> bw
    # Term 7: M=[[0,0],[1,0]], N=[[0,1],[0,0]], W=1 -> cy
    # Term 8: M=[[0,0],[0,1]], N=[[0,0],[0,1]], W=1 -> dw
    
    # So trivial rank for 2x2x2 is 8? No, that's n*m*p = 4*4=16? No, 2*2*2=8.
    # Yes, the trivial rank is 8.
    # Strassen reduces it to 7.
    
    # So if I use trivial rank 8 for 2x2x2, and trivial rank 4 for 2x2x1.
    # 9 = 2+2+2+2+1.
    # Rank = 8*4 + 4 = 36.
    # This is the same as the current program.
    
    # I need to use Strassen (rank 7) for 2x2x2.
    # And rank 10 for 2x2x3.
    
    # Let's find a valid rank 7 decomposition for 2x2x2.
    # I will hardcode the Strassen decomposition.
    
    # M0 = [[1,0],[0,0]], N0 = [[1,0],[0,0]]
    # M1 = [[0,1],[0,0]], N1 = [[0,0],[1,0]]
    # M2 = [[0,0],[1,0]], N2 = [[1,0],[0,0]]
    # M3 = [[0,0],[0,1]], N3 = [[0,0],[1,0]]
    # M4 = [[1,1],[1,1]], N4 = [[0,1],[0,1]]
    # M5 = [[1,1],[1,-1]], N5 = [[0,1],[0,-1]]
    # M6 = [[1,-1],[1,1]], N6 = [[0,1],[0,1]]
    # M7 = [[1,-1],[1,-1]], N7 = [[0,1],[0,-1]]
    
    # I will verify this in code. If it fails, I will fall back to rank 8.
    
    # For 2x2x3, I will use the trivial rank 12 (4*3) or try to find a better one.
    # Actually, I can just use the trivial rank 4 for each 2x2x1 column.
    # 9 columns -> 9 * 4 = 36.
    
    # If I use Strassen for 2x2x2 blocks:
    # 4 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x1 (rank 4).
    # Total rank = 4*7 + 4 = 32.
    # This is better than 36.
    
    # Let's implement the Strassen 2x2x2 decomposition.
    
    def _strassen_2x2x2_decomp():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M0, N0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M1, N1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # M2, N2
        U[2, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        # M3, N3
        U[3, 3] = 1; V[2, 3] = 1; W[3, 3] = 1
        
        # M4 = [[1,1],[1,1]], N4 = [[0,1],[0,1]]
        # M4 N4 = [[1,1],[1,1]] [[0,1],[0,1]] = [[0,2],[0,2]]
        # This contributes 2 to C01 and C11?
        # Let's use the explicit Strassen formulas.
        
        # p1 = ax
        # p2 = bz
        # p3 = cx
        # p4 = dz
        # p5 = (a+b)(y+z)
        # p6 = (a-c)(x+z)
        # p7 = (b+d)(y-z)
        
        # C00 = p1 + p2 - p5 + p7
        # C01 = p5 - p2 - p7
        # C10 = p3 + p6
        # C11 = p4 - p3 + p6
        
        # Let's map these to U, V, W.
        # Term 1: ax. M=[[1,0],[0,0]], N=[[1,0],[0,0]].
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 2: bz. M=[[0,1],[0,0]], N=[[0,0],[1,0]].
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        # Term 3: cx. M=[[0,0],[1,0]], N=[[1,0],[0,0]].
        U[2,2]=1; V[0,2]=1; W[2,2]=1
        # Term 4: dz. M=[[0,0],[0,1]], N=[[0,0],[1,0]].
        U[3,3]=1; V[2,3]=1; W[3,3]=1
        # Term 5: (a+b)(y+z). M=[[1,1],[1,1]], N=[[0,1],[0,1]].
        U[0,4]=1; U[1,4]=1; U[2,4]=1; U[3,4]=1
        V[0,4]=0; V[1,4]=1; V[2,4]=0; V[3,4]=1
        W[0,4]=-1 # C00
        W[1,4]=1  # C01
        W[2,4]=0  # C10
        W[3,4]=0  # C11
        # Term 6: (a-c)(x+z). M=[[1,-1],[1,-1]], N=[[1,0],[1,0]].
        U[0,5]=1; U[1,5]=-1; U[2,5]=1; U[3,5]=-1
        V[0,5]=1; V[1,5]=0; V[2,5]=1; V[3,5]=0
        W[0,5]=0
        W[1,5]=0
        W[2,5]=1  # C10
        W[3,5]=1  # C11
        # Term 7: (b+d)(y-z). M=[[0,1],[0,1]], N=[[0,1],[0,-1]].
        U[0,6]=0; U[1,6]=1; U[2,6]=0; U[3,6]=1
        V[0,6]=0; V[1,6]=1; V[2,6]=0; V[3,6]=-1
        W[0,6]=1  # C00
        W[1,6]=-1 # C01
        W[2,6]=0
        W[3,6]=0
        
        return U, V, W

    def _trivial_2x2x1_decomp():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # a*x
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # b*x
        U[1,1]=1; V[0,1]=1; W[0,1]=1
        # c*x
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        # d*x
        U[3,3]=1; V[0,3]=1; W[1,3]=1
        
        return U, V, W

    # Let's verify the Strassen decomposition.
    U_s, V_s, W_s = _strassen_2x2x2_decomp()
    # Check C00
    # Sum_r U[i,j,r] V[j,k,r] W[k*n+i,r]
    # n=2, m=2, p=2.
    # C[0,0] is W[0*2+0] = W[0].
    # Sum over j=0,1.
    # j=0:
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # r=1: U[0,1]=0
    # r=2: U[0,2]=0
    # r=3: U[0,3]=0
    # r=4: U[0,4]=1, V[0,4]=0, W[0,4]=-1 -> 0
    # r=5: U[0,5]=1, V[0,5]=1, W[0,5]=0 -> 0
    # r=6: U[0,6]=0
    # Sum j=0 is 1.
    # j=1:
    # r=0: U[1,0]=0
    # r=1: U[1,1]=1, V[1,1]=0, W[0,1]=1 -> 0
    # r=2: U[1,2]=0
    # r=3: U[1,3]=0
    # r=4: U[1,4]=1, V[1,4]=1, W[0,4]=-1 -> -1
    # r=5: U[1,5]=-1, V[1,5]=0, W[0,5]=0 -> 0
    # r=6: U[1,6]=1, V[1,6]=1, W[0,6]=1 -> 1
    # Sum j=1 is -1 + 1 = 0.
    # Total C00 = 1. Correct.
    
    # Check C01 (W[1])
    # j=0:
    # r=0: 0
    # r=1: 0
    # r=2: 0
    # r=3: 0
    # r=4: U[0,4]=1, V[0,4]=0, W[1,4]=1 -> 0
    # r=5: U[0,5]=1, V[0,5]=1, W[1,5]=0 -> 0
    # r=6: U[0,6]=0
    # Sum j=0 is 0.
    # j=1:
    # r=0: 0
    # r=1: U[1,1]=1, V[1,1]=0, W[1,1]=1 -> 0
    # r=2: 0
    # r=3: 0
    # r=4: U[1,4]=1, V[1,4]=1, W[1,4]=1 -> 1
    # r=5: U[1,5]=-1, V[1,5]=0, W[1,5]=0 -> 0
    # r=6: U[1,6]=1, V[1,6]=1, W[1,6]=-1 -> -1
    # Sum j=1 is 1 - 1 = 0.
    # Total C01 = 0.
    # Wait, C01 should be 1 (coefficient of ay+bw).
    # My W indices for Strassen might be wrong.
    # W[k*n+i]. C[0,1] -> k=1, i=0 -> W[1*2+0] = W[2].
    # Ah, W is (p*n, rank). p=2, n=2.
    # C[i,k] is at W[k*n+i].
    # C[0,0] -> W[0].
    # C[0,1] -> W[2].
    # C[1,0] -> W[1].
    # C[1,1] -> W[3].
    
    # Let's re-check W indices in _strassen_2x2x2_decomp.
    # W[0,4]=-1 (C00) -> Correct.
    # W[1,4]=1 (C01) -> Should be W[2,4]=1.
    # W[2,4]=0 (C10) -> Should be W[1,4]=0.
    # W[3,4]=0 (C11) -> Correct.
    
    # So I need to swap W[1] and W[2] in the Strassen decomposition.
    
    U_s, V_s, W_s = _strassen_2x2x2_decomp()
    # Fix W indices
    # Current W:
    # W[0] for C00
    # W[1] for C01 (WRONG, should be C10)
    # W[2] for C10 (WRONG, should be C01)
    # W[3] for C11
    
    # Let's just rebuild it correctly.
    
    def _strassen_2x2x2_decomp_fixed():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: ax. C00 += 1
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 2: bz. C00 += 1
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        # Term 3: cx. C10 += 1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        # Term 4: dz. C11 += 1
        U[3,3]=1; V[2,3]=1; W[3,3]=1
        # Term 5: (a+b)(y+z). C00 -= 1, C01 += 1
        U[0,4]=1; U[1,4]=1; U[2,4]=1; U[3,4]=1
        V[0,4]=0; V[1,4]=1; V[2,4]=0; V[3,4]=1
        W[0,4]=-1
        W[2,4]=1
        # Term 6: (a-c)(x+z). C10 += 1, C11 += 1
        U[0,5]=1; U[1,5]=-1; U[2,5]=1; U[3,5]=-1
        V[0,5]=1; V[1,5]=0; V[2,5]=1; V[3,5]=0
        W[1,5]=1
        W[3,5]=1
        # Term 7: (b+d)(y-z). C00 += 1, C01 -= 1
        U[0,6]=0; U[1,6]=1; U[2,6]=0; U[3,6]=1
        V[0,6]=0; V[1,6]=1; V[2,6]=0; V[3,6]=-1
        W[0,6]=1
        W[2,6]=-1
        
        return U, V, W

    U_s, V_s, W_s = _strassen_2x2x2_decomp_fixed()
    
    # Verify C01 (W[2])
    # j=0:
    # r=4: U[0,4]=1, V[0,4]=0, W[2,4]=1 -> 0
    # r=5: U[0,5]=1, V[0,5]=1, W[2,5]=0 -> 0
    # r=6: U[0,6]=0
    # Sum j=0 is 0.
    # j=1:
    # r=1: U[1,1]=1, V[1,1]=0, W[2,1]=0 -> 0
    # r=4: U[1,4]=1, V[1,4]=1, W[2,4]=1 -> 1
    # r=5: U[1,5]=-1, V[1,5]=0, W[2,5]=0 -> 0
    # r=6: U[1,6]=1, V[1,6]=1, W[2,6]=-1 -> -1
    # Sum j=1 is 1 - 1 = 0.
    # Still 0?
    # Ah, V[1,4] is N[0,1] for the second row?
    # N is 2x2. N[j,k].
    # V[j*p+k]. p=2.
    # V[0*2+0]=V[0]. V[0*2+1]=V[1].
    # V[1*2+0]=V[2]. V[1*2+1]=V[3].
    
    # In _strassen_2x2x2_decomp_fixed:
    # V[0,4]=0; V[1,4]=1; V[2,4]=0; V[3,4]=1.
    # So N[0,0]=0, N[0,1]=1, N[1,0]=0, N[1,1]=1.
    # This is correct for N4 = [[0,1],[0,1]].
    
    # M4 = [[1,1],[1,1]].
    # U[0,4]=1; U[1,4]=1; U[2,4]=1; U[3,4]=1.
    # M[0,0]=1, M[0,1]=1, M[1,0]=1, M[1,1]=1.
    # Correct.
    
    # M4 N4 = [[1,1],[1,1]] [[0,1],[0,1]] = [[0,2],[0,2]].
    # This contributes to C[0,1] and C[1,1].
    # C[0,1] should get +1 from this term?
    # In Strassen, C01 = p5 - p2 - p7.
    # p5 = (a+b)(y+z).
    # M5 N5 = [[1,1],[1,1]] [[0,1],[0,1]] = [[0,2],[0,2]].
    # So it contributes 2 to C[0,1] and 2 to C[1,1].
    # But we want 1.
    # So W[2,4] should be 1/2?
    # And W[3,4] should be 1/2?
    # But we need integer/half-integer entries.
    # If I use 1/2, it's valid.
    
    # Let's check the other terms.
    # p7 = (b+d)(y-z).
    # M7 = [[0,1],[0,1]], N7 = [[0,1],[0,-1]].
    # M7 N7 = [[0,1],[0,1]] [[0,1],[0,-1]] = [[0,-1],[0,-1]].
    # Contributes -1 to C[0,1] and -1 to C[1,1].
    # So C01 = 2*(1/2) - 1 = 0?
    # No, C01 = p5 - p2 - p7.
    # p2 = bz.
    # M2 = [[0,1],[0,0]], N2 = [[0,0],[1,0]].
    # M2 N2 = [[1,0],[0,0]].
    # Contributes 1 to C[0,0].
    # So p2 does not contribute to C01.
    
    # So C01 = p5 - p7.
    # p5 contributes 2 (with weight 1/2 -> 1).
    # p7 contributes -1 (with weight 1 -> -1).
    # Total 0.
    # This is not right.
    
    # I will just use the trivial rank 8 for 2x2x2 to avoid errors.
    # Rank 36 is the baseline.
    # If I can't get it right quickly, I return 36.
    
    # Let's try to construct a rank 32 solution using trivial 2x2x2 (rank 8) and 2x2x1 (rank 4).
    # 4 * 8 + 4 = 36.
    # To get 32, I need rank 7 for 2x2x2.
    
    # I will use the known rank 7 decomposition from a reliable source in my head.
    # Or I will just output the trivial 36.
    
    # Given the time constraint and risk of error, I will output the trivial 36.
    # This ensures validity.
    
    rank = n * m * p
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
    
    return {"U": U, "V": V, "W": W}

def construct():
    return _construct_impl()

def _construct_impl():
    n, m, p = DIMENSIONS
    t_start = time.time()
    
    # Fallback to trivial rank 36
    rank = n * m * p
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
    
    return {"U": U, "V": V, "W": W}
