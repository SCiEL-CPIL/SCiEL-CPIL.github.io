import numpy as np

def construct():
    n, m, p = 2, 4, 6
    # Use the known optimal decomposition for M_{2,4,6}.
    # The current best known rank is 42 (from Strassen-based blocking).
    # We construct the tensor decomposition explicitly using a systematic
    # approach based on tensor slicing and rank analysis.
    
    # The matrix multiplication tensor T has entries T[i,k,j,l] = 1 if k==j else 0
    # where i in [0,n), k in [0,m), j in [0,m), l in [0,p)
    # But with the trace convention: W[k*n+i, r] for output
    
    # For the specific case of 2x4 times 4x6, we use a decomposition
    # that combines Strassen's method on 2x2 blocks with efficient handling
    # of the remaining dimensions.
    
    # Strategy: Partition the 4x4 multiplication into four 2x2 blocks
    # and use Strassen for each, then handle cross-terms efficiently.
    
    # Actually, let's use the known construction:
    # The rank of M_{2,4,6} is at most 42, and this is believed to be optimal.
    
    # We'll construct using a block-based approach:
    # Split B (4x6) into three 4x2 blocks: B1, B2, B3
    # Split A (2x4) into two 2x2 blocks: A1, A2
    # Split C (2x6) into three 2x2 blocks: C1, C2, C3
    
    # C1 = A1*B1 + A2*B4 (where B4 is rows 2-3 of B, cols 0-1)
    # Wait, let me reconsider the block structure.
    
    # A is 2x4: columns 0,1 form A_L (2x2), columns 2,3 form A_R (2x2)
    # B is 4x6: 
    #   rows 0,1 and cols 0,1 form B_TL (2x2)
    #   rows 0,1 and cols 2,3 form B_TM (2x2)
    #   rows 0,1 and cols 4,5 form B_TR (2x2)
    #   rows 2,3 and cols 0,1 form B_BL (2x2)
    #   rows 2,3 and cols 2,3 form B_BM (2x2)
    #   rows 2,3 and cols 4,5 form B_BR (2x2)
    
    # C = A*B, so:
    # C[:,0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2]
    # C[:,2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4]
    # C[:,4:6] = A[:,0:2]*B[0:2,4:6] + A[:,2:4]*B[2:4,4:6]
    
    # Each 2x2 * 2x2 multiplication can be done with Strassen (rank 7)
    # But we have 3 outputs, each requiring 2 such multiplications = 6 multiplications
    # 6 * 7 = 42
    
    # Let's construct this:
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def strassen_2x2(A_u, B_v, C_w, base):
        """
        A_u: [a, b, c, d] indices into U for 2x2 matrix A
        B_v: [e, f, g, h] indices into V for 2x2 matrix B  
        C_w: [c11, c12, c21, c22] indices into W for 2x2 output C
        base: starting column index
        Returns next base index
        """
        a, b, c, d = A_u
        e, f, g, h = B_v
        c11, c12, c21, c22 = C_w
        
        # M1 = (a+d)(e+h)
        U[a][base] += 1; U[d][base] += 1
        V[e][base] += 1; V[h][base] += 1
        W[c11][base] += 1; W[c22][base] += 1
        base += 1
        
        # M2 = (c+d)e
        U[c][base] += 1; U[d][base] += 1
        V[e][base] += 1
        W[c21][base] += 1; W[c22][base] -= 1
        base += 1
        
        # M3 = a(f-h)
        U[a][base] += 1
        V[f][base] += 1; V[h][base] -= 1
        W[c12][base] += 1; W[c22][base] += 1
        base += 1
        
        # M4 = d(g-e)
        U[d][base] += 1
        V[g][base] += 1; V[e][base] -= 1
        W[c11][base] += 1; W[c21][base] += 1
        base += 1
        
        # M5 = (a+b)h
        U[a][base] += 1; U[b][base] += 1
        V[h][base] += 1
        W[c11][base] -= 1; W[c12][base] += 1
        base += 1
        
        # M6 = (c-a)(e+f)
        U[c][base] += 1; U[a][base] -= 1
        V[e][base] += 1; V[f][base] += 1
        W[c22][base] += 1
        base += 1
        
        # M7 = (b-d)(g+h)
        U[b][base] += 1; U[d][base] -= 1
        V[g][base] += 1; V[h][base] += 1
        W[c11][base] += 1
        base += 1
        
        return base
    
    col = 0
    
    # A_L: columns 0,1 -> U indices [0,1,4,5] (a=0,b=1,c=4,d=5)
    # A_R: columns 2,3 -> U indices [2,3,6,7] (a=2,b=3,c=6,d=7)
    A_L = [0, 1, 4, 5]
    A_R = [2, 3, 6, 7]
    
    # For each output column block k in {0,1,2}:
    for k in range(3):
        # Output C[:, 2k:2k+2]
        # C_w indices: c11 = C[0,2k], c12 = C[0,2k+1], c21 = C[1,2k], c22 = C[1,2k+1]
        # With trace convention W[k*n+i], so C[row, col] -> W[col*n + row]
        c11 = (2*k) * n + 0
        c12 = (2*k+1) * n + 0
        c21 = (2*k) * n + 1
        c22 = (2*k+1) * n + 1
        C_w = [c11, c12, c21, c22]
        
        # B top block (rows 0,1) cols 2k, 2k+1
        # B is 4x6, so B[row, col] -> V[row*6 + col]
        e = 0 * p + 2*k
        f = 0 * p + 2*k + 1
        g = 1 * p + 2*k
        h = 1 * p + 2*k + 1
        B_top = [e, f, g, h]
        
        # B bottom block (rows 2,3) cols 2k, 2k+1
        e = 2 * p + 2*k
        f = 2 * p + 2*k + 1
        g = 3 * p + 2*k
        h = 3 * p + 2*k + 1
        B_bot = [e, f, g, h]
        
        # C[:, 2k:2k+2] = A_L * B_top + A_R * B_bot
        col = strassen_2x2(A_L, B_top, C_w, col)
        col = strassen_2x2(A_R, B_bot, C_w, col)
    
    return {"U": U, "V": V, "W": W}
