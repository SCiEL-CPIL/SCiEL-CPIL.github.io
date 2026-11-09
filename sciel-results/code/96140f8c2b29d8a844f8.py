import numpy as np

def construct():
    n, m, p = 2, 4, 4
    
    # We'll build a decomposition using Strassen's algorithm on 2x2 blocks.
    # The matrix multiplication is A (2x4) * B (4x4) = C (2x4)
    # We split B into two 2x4 blocks: B1 = B[:, 0:2], B2 = B[:, 2:4]
    # Then C[:, 0:2] = A * B1 and C[:, 2:4] = A * B2
    # Each of these is a 2x4 * 4x2 = 2x2 multiplication? No, wait.
    # A is 2x4, B1 is 4x2. So A * B1 is 2x2.
    # So we have two independent 2x2 matrix multiplications:
    # C1 = A * B1 (2x2)
    # C2 = A * B2 (2x2)
    # But A is shared! So we can't just do two independent Strassen.
    # However, we can use the linearity. Let's think differently.
    
    # Actually, let's use the fact that 2x4 * 4x4 can be seen as:
    # For each column k of B, we compute A * B[:, k] which is a 2x1 vector.
    # So we have 4 independent 2x4 * 4x1 multiplications.
    # Each 2x4 * 4x1 is just 2 dot products of length 4.
    # The rank of 2x4 * 4x1 is at most 4 (schoolbook), but can it be less?
    # Actually, the tensor [2,4,1] has rank 4 (I believe).
    # So 4 of them would be rank 16. But wait, the B matrix is shared across k.
    
    # Let's try a different approach. Use the known rank for [2,4,4].
    # According to some sources, the rank of [2,4,4] is 20.
    # Let's try to construct a rank 20 decomposition.
    
    # Alternative: Use the fact that [2,4,4] can be decomposed using
    # Strassen-like methods on the j and k dimensions.
    
    # Let's try to use a structured decomposition.
    # We'll use the following idea:
    # Split j into {0,1} and {2,3}. Split k into {0,1} and {2,3}.
    # For each pair of j-groups and k-groups, we have a 2x2 * 2x2 sub-problem.
    # There are 2*2 = 4 such sub-problems.
    # Each 2x2 * 2x2 can be done in rank 7 (Strassen).
    # Total rank: 4 * 7 = 28.
    # But wait, the A matrix is shared across all k-groups, and B is shared across j-groups.
    # So we can't just do 4 independent Strassen.
    
    # Let's try rank 24.
    # We'll use a hybrid approach.
    
    # Actually, let's just implement a rank 28 decomposition using Strassen on 2x2 blocks.
    # This is simpler and guaranteed to work.
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    # Strassen's 7 products for 2x2 * 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 - M3 - M1  # Wait, let me verify this
    # C11 = M1 + M4 - M5 + M6
    
    # Let me verify C10:
    # M2 + M5 - M3 - M1
    # = (c+d)e + (a+b)h - a(f-h) - (a+d)(e+h)
    # = ce + de + ah + bh - af + ah - (ae + ah + de + dh)
    # = ce + de + ah + bh - af + ah - ae - ah - de - dh
    # = ce + bh - af - ae - dh + ah  # This doesn't look right.
    
    # Correct Strassen reconstruction:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 - M3 - M1  # Let me re-check
    # Actually, the standard Strassen is:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 - M3 - M1  # I think this is wrong. Let me look it up.
    # Correct:
    # C10 = M2 + M5 - M3 - M1  # No.
    # Let's use:
    # C10 = M2 + M5 - M3 - M1  # I'll just use the standard formulas.
    
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 - M3 - M1  # Let me verify with a=1,b=0,c=0,d=1,e=1,f=0,g=0,h=1
    # A=I, B=I. C should be I.
    # M1=(1+1)(1+1)=4, M2=(0+1)*1=1, M3=1*(0-1)=-1, M4=1*(0-1)=-1
    # M5=(1+0)*1=1, M6=(0-1)(1+0)=-1, M7=(0-1)(0+1)=-1
    # C00=4-(-1)+1-(-1)=7? That's wrong. C00 should be 1.
    
    # I think I have the Strassen formulas wrong. Let me use the correct ones.
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 - M3 - M1  # Still not sure.
    
    # Let me just use the schoolbook for 2x2 * 2x2 (rank 4) and see if I can get rank 16.
    # 4 blocks * 4 rank = 16. That's better than 32.
    
    # Let's do that.
    rank = 16
    U = [[0]*rank for _ in range(n*m)]  # 8 x 16
    V = [[0]*rank for _ in range(m*p)]  # 16 x 16
    W = [[0]*rank for _ in range(p*n)]  # 8 x 16
    
    col = 0
    # For each j-group (2 groups: {0,1} and {2,3}) and k-group (2 groups: {0,1} and {2,3})
    for jg in range(2):  # j-group index
        for kg in range(2):  # k-group index
            # For each (i, j_local, k_local) in 2x2x2
            for i in range(2):
                for jl in range(2):
                    for kl in range(2):
                        j = jg*2 + jl
                        k = kg*2 + kl
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
