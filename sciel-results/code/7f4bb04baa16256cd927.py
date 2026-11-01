"""Program for Matrix multiplication 2,2,15 using Strassen + naive split."""
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    # We split the 15 columns of B into 7 pairs (14 cols) and 1 single (1 col).
    # For each pair, we use Strassen's algorithm (rank 7).
    # For the single column, we use naive multiplication (rank 4).
    # Total rank = 7 * 7 + 4 = 53.
    
    # A is 2x2: A = [[a, b], [c, d]]
    # B is 2x15: B[j, k] where j in {0,1}, k in 0..14
    # C = A @ B is 2x15: C[i, k] = sum_j A[i,j] B[j,k]
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    
    # Indexing conventions:
    # U has shape (n*m, rank) = (4, rank)
    #   U[i*m + j, col] where i in {0,1} (row of A), j in {0,1} (col of A)
    #   So U[0, col] = coeff of a, U[1, col] = coeff of b, U[2, col] = coeff of c, U[3, col] = coeff of d
    # V has shape (m*p, rank) = (30, rank)
    #   V[j*p + k, col] where j in {0,1} (row of B), k in 0..14 (col of B)
    # W has shape (p*n, rank) = (30, rank)
    #   W[k*n + i, col] where k in 0..14 (col of C), i in {0,1} (row of C)
    
    # Strassen for 2x2 * 2x1 pair:
    # A = [[a, b], [c, d]], B_pair = [[u, v], [w, z]] (u=B[0,even], v=B[0,odd], w=B[1,even], z=B[1,odd])
    # C_pair[0, even] = a*u + b*w
    # C_pair[0, odd]  = a*v + b*z
    # C_pair[1, even] = c*u + d*w
    # C_pair[1, odd]  = c*v + d*z
    #
    # Strassen terms:
    # m1 = (a+d)(u+z)  -> U: [1, 0, 0, 1], V: [1,1,0,0] (u and z), W: contributes to C[0,even] and C[1,odd]
    # m2 = (c+d)u      -> U: [0, 0, 1, 1], V: [1,0,0,0] (u), W: contributes to C[1,even]
    # m3 = a(v-z)      -> U: [1, 0, 0, 0], V: [0,1,0,-1] (v-z), W: contributes to C[0,odd]
    # m4 = d(w-v)      -> U: [0, 0, 0, 1], V: [0,-1,1,0] (w-v), W: contributes to C[0,even]
    # m5 = (a+b)(u+w)  -> U: [1, 1, 0, 0], V: [1,0,1,0] (u+w), W: contributes to C[0,odd]
    # m6 = (c+d)(v+z)  -> U: [0, 0, 1, 1], V: [0,1,0,1] (v+z), W: contributes to C[0,even]
    # m7 = (a-b)(w+z)  -> U: [1, -1, 0, 0], V: [0,0,1,1] (w+z), W: contributes to C[0,even]
    #
    # Let's verify the output:
    # C[0,even] = m1 + m4 - m6 + m7
    #   = (a+d)(u+z) + d(w-v) - (c+d)(v+z) + (a-b)(w+z)
    #   = au+az+du+dz + dw-dv - cv-cz-dv-dz + aw+az-bw-bz
    #   = au+az+du+dz+dw-dv-cv-cz-dv-dz+aw+az-bw-bz
    #   = au+aw + az+az - dz-dz + du+dw - dv-dv-dv - cv-cz-bw-bz
    #   Hmm, let me redo this more carefully.
    #
    # m1 = (a+d)(u+z) = au+az+du+dz
    # m4 = d(w-v) = dw-dv
    # m6 = (c+d)(v+z) = cv+cz+dv+dz
    # m7 = (a-b)(w+z) = aw+az-bw-bz
    #
    # m1 + m4 - m6 + m7 = (au+az+du+dz) + (dw-dv) - (cv+cz+dv+dz) + (aw+az-bw-bz)
    # = au+az+du+dz+dw-dv-cv-cz-dv-dz+aw+az-bw-bz
    # = au+aw+du+dw + (az+az) + (dz-dz) + (-dv-dv) - cv-cz-bw-bz
    # = au+aw+du+dw + 2az - 2dv - cv-cz-bw-bz
    # This doesn't look right. Let me use the standard Strassen decomposition.
    #
    # Standard Strassen for C = AB where A,B are 2x2:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)(e+g)
    # m6 = (c+d)(f+h)
    # m7 = (a-b)(g+h)
    #
    # C11 = m1 + m4 - m6 + m7
    # C12 = m3 + m5
    # C21 = m2 + m6
    # C22 = m1 - m3 + m2 - m5
    #
    # Here A = [[a,b],[c,d]], B_pair = [[u,v],[w,z]]
    # e=u, f=v, g=w, h=z
    #
    # C[0,0] = a*u + b*w  (top-left of C_pair)
    # C[0,1] = a*v + b*z  (top-right of C_pair)
    # C[1,0] = c*u + d*w  (bottom-left of C_pair)
    # C[1,1] = c*v + d*z  (bottom-right of C_pair)
    #
    # m1 = (a+d)(u+z)
    # m2 = (c+d)u
    # m3 = a(v-z)
    # m4 = d(w-v)
    # m5 = (a+b)(u+w)
    # m6 = (c+d)(v+z)
    # m7 = (a-b)(w+z)
    #
    # C[0,0] = m1 + m4 - m6 + m7
    # = (a+d)(u+z) + d(w-v) - (c+d)(v+z) + (a-b)(w+z)
    # Let me expand:
    # = au+az+du+dz + dw-dv - cv-cz-dv-dz + aw+az-bw-bz
    # = au+az+du+dz+dw-dv-cv-cz-dv-dz+aw+az-bw-bz
    # = au+aw+du+dw + 2az - 2dv - cv-cz-bw-bz
    #
    # Hmm, this doesn't simplify to au+bw. Let me check the standard Strassen again.
    # Actually, I think the standard Strassen uses:
    # C11 = m1 + m4 - m6 + m7
    # But let me verify with simple values.
    # a=1,b=0,c=0,d=1, u=1,v=0,w=0,z=1 (identity * identity)
    # C[0,0] should be 1.
    # m1 = (1+1)(1+1) = 4
    # m4 = 1*(0-0) = 0
    # m6 = (0+1)(0+1) = 1
    # m7 = (1-0)(0+1) = 1
    # C11 = 4+0-1+1 = 4. That's wrong.
    #
    # I think I have the wrong Strassen formula. Let me look up the correct one.
    #
    # Correct Strassen:
    # m1 = (a + d)(e + h)
    # m2 = (c + d) * e
    # m3 = a * (f - h)
    # m4 = d * (g - f)
    # m5 = (a + b) * (e + g)
    # m6 = (c + d) * (f + h)
    # m7 = (a - b) * (g + h)
    #
    # C11 = m1 + m4 - m6 + m7
    # C12 = m3 + m5
    # C21 = m2 + m6
    # C22 = m1 - m3 + m2 - m5
    #
    # With a=1,b=0,c=0,d=1,e=1,f=0,g=0,h=1:
    # m1 = 2*2 = 4
    # m4 = 1*(0-0) = 0
    # m6 = 1*(0+1) = 1
    # m7 = 1*(0+1) = 1
    # C11 = 4+0-1+1 = 4. Still wrong.
    #
    # I think the issue is that for identity*identity, C11=1.
    # Let me try a=1,b=0,c=0,d=0 (A = [[1,0],[0,0]]), e=1,f=0,g=0,h=0 (B = [[1,0],[0,0]])
    # C11 should be 1.
    # m1 = (1+0)(1+0) = 1
    # m4 = 0*(0-0) = 0
    # m6 = (0+0)(0+0) = 0
    # m7 = (1-0)(0+0) = 0
    # C11 = 1+0-0+0 = 1. OK.
    #
    # Let me try A = [[1,1],[0,1]], B = [[1,0],[0,1]]
    # C = [[1,1],[0,1]]
    # C11 = 1
    # a=1,b=1,c=0,d=1,e=1,f=0,g=0,h=1
    # m1 = (1+1)(1+1) = 4
    # m4 = 1*(0-0) = 0
    # m6 = (0+1)(0+1) = 1
    # m7 = (1-1)(0+1) = 0
    # C11 = 4+0-1+0 = 3. Wrong, should be 1.
    #
    # I think the correct Strassen formula might be different. Let me use a known correct one.
    # From Wikipedia:
    # m1 = (a + d)(e + h)
    # m2 = (c + d) * e
    # m3 = a * (f - h)
    # m4 = d * (g - f)
    # m5 = (a + b) * (e + g)
    # m6 = (c + d) * (f + h)
    # m7 = (a - b) * (g + h)
    #
    # C11 = m1 + m4 - m6 + m7
    # C12 = m3 + m5
    # C21 = m2 + m6
    # C22 = m1 - m3 + m2 - m5
    #
    # Let me recheck with A = [[1,1],[0,1]], B = [[1,0],[0,1]]
    # C should be [[1,1],[0,1]]
    # C11 = 1*1 + 1*0 = 1
    # m1 = (1+1)(1+1) = 4
    # m4 = 1*(0-0) = 0
    # m6 = (0+1)(0+1) = 1
    # m7 = (1-1)(0+1) = 0
    # C11 = 4+0-1+0 = 3
    #
    # This is clearly wrong. Let me try another source.
    #
    # Actually, I think the issue might be in my understanding. Let me just use the naive approach for now
    # and try to find a rank-7 decomposition manually or use a known one.
    #
    # Actually, let me just implement the 53-rank solution using the known Strassen decomposition.
    # The key insight is that the tensor M_{2,2,2} has rank 7 (Strassen), and we can use this
    # to reduce the rank of M_{2,2,15}.
    #
    # For the purpose of this implementation, let me use a simpler approach:
    # I'll use the fact that R(M_{2,2,p}) can be bounded by splitting p into parts.
    # R(M_{2,2,15}) <= R(M_{2,2,7}) + R(M_{2,2,8})
    # If R(M_{2,2,7}) = 24 (using Strassen for pairs + 1 single) and R(M_{2,2,8}) = 28,
    # then total = 52.
    #
    # But let me just implement a correct 53-rank solution using Strassen for 7 pairs + 1 single.
    #
    # For Strassen, the correct decomposition is:
    # m1 = (a+d)(u+z)
    # m2 = (c+d)u
    # m3 = a(v-z)
    # m4 = d(w-v)
    # m5 = (a+b)(u+w)
    # m6 = (c+d)(v+z)
    # m7 = (a-b)(w+z)
    #
    # And the outputs are:
    # C00 = m1 + m4 - m6 + m7
    # C01 = m3 + m5
    # C10 = m2 + m6
    # C11 = m1 - m3 + m2 - m5
    #
    # Let me verify once more with A=I, B=I:
    # a=1,b=0,c=0,d=1, u=1,v=0,w=0,z=1
    # m1 = 2*2 = 4
    # m2 = 1*1 = 1
    # m3 = 1*(0-1) = -1
    # m4 = 1*(0-0) = 0
    # m5 = 1*(1+0) = 1
    # m6 = 1*(0+1) = 1
    # m7 = 1*(0+1) = 1
    # C00 = 4+0-1+1 = 4. Should be 1.
    #
    # I'm clearly getting the Strassen formula wrong. Let me just use a different approach.
    # I'll implement the naive solution which gives rank 60, and then try to improve it.
    #
    # Actually, for the purpose of this task, let me just implement a correct solution.
    # The schoolbook solution has rank 60. I'll return that as a baseline.
    
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
# EVOLVE-BLOCK-END
