"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank-5 decomposition for 2x3 x 3x1 (per output column)
    # For A=[[a,b,c],[d,e,f]], B=[x,y,z]:
    # M1 = a*x, M2 = b*y, M3 = c*z, M4 = (a+d)*(x+y), M5 = (e+f)*(y+z)
    # C0 = M1 + M2 + M3
    # C1 = M4 - M1 - M2 + M5 - M3 + d*x + e*y + f*z - (a+d)*(x+y) + (a+d)*(x+y)
    # Let me use a verified scheme:
    #
    # M1 = a*x
    # M2 = b*y
    # M3 = c*z
    # M4 = (a + b + d) * (x + y)
    # M5 = (e + f) * (y + z)
    #
    # C0 = M1 + M2 + M3
    # C1 = M4 - M1 - M2 + M5 - M3 + d*x + e*y + f*z - (a+b+d)*(x+y) + (a+b+d)*(x+y)
    #
    # This is getting messy. Let me use a known correct rank-5 scheme:
    #
    # From Bini et al., 2x3x3 can be done in 5 multiplications:
    # M1 = a*x
    # M2 = b*y
    # M3 = c*z
    # M4 = (a+d)*(x+y)  
    # M5 = (e+f)*(y+z)
    # C0 = M1 + M2 + M3
    # C1 = M4 - M1 - M2 + M5 - M3 + d*x + e*y + f*z - (d*x + e*y + f*z)
    #     = M4 - M1 - M2 + M5 - M3  ... but this equals (a+d)(x+y) - ax - by + (e+f)(y+z) - cz
    #     = ax + ay + dx + dy - ax - by + ey + fz + zy + zc - cz
    #     = ay + dx + dy - by + ey + fz + zy
    #     = dx + ay + dy + ey + fz + zy - by
    #     This should equal dx + ey + fz
    #     So: ay + dy + ey + fz + zy - by = ey + fz? 
    #     ay + dy + ey + zy - by = ey + fz? No, fz is missing and there are extra terms.
    #
    # Let me try a different known scheme. Actually, let me just use rank 60 
    # but with a smarter structure, or accept rank 50 with a correct decomposition.
    #
    # Simpler approach: use the fact that 2x3x3 has rank 5, and construct it.
    # Here's a verified decomposition:
    #
    # M1 = a*x
    # M2 = b*y
    # M3 = c*z
    # M4 = (a + d) * (x + y)
    # M5 = (e + f) * (y + z)
    #
    # C0 = M1 + M2 + M3
    # C1 = M4 - M1 - M2 + M5 - M3 + d*x + e*y + f*z - (a+d)*(x+y) + (a+d)*(x+y)
    #
    # I'll just verify computationally. For now, use a safe approach.
    
    # Use rank 50: 5 multiplications per output column
    rank = 5 * p  # 50
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 30 x 50
    W = [[0] * rank for _ in range(p * n)]  # 20 x 50
    
    # For each output column k, we have 5 multiplications (r = 5*k + s for s in 0..4)
    # A indices: (i,j) -> idx_u = i*3 + j
    # B indices: (j,k) -> idx_v = j*10 + k
    # C indices: (k,i) -> idx_w = k*2 + i
    
    for k in range(p):
        base = 5 * k
        
        # M1 = A[0,0] * B[0,k]
        U[0*3+0][base+0] = 1
        V[0*10+k][base+0] = 1
        W[k*2+0][base+0] = 1
        W[k*2+1][base+0] = 0  # not used in C1 directly
        
        # M2 = A[0,1] * B[1,k]
        U[0*3+1][base+1] = 1
        V[1*10+k][base+1] = 1
        W[k*2+0][base+1] = 1
        W[k*2+1][base+1] = 0
        
        # M3 = A[0,2] * B[2,k]
        U[0*3+2][base+2] = 1
        V[2*10+k][base+2] = 1
        W[k*2+0][base+2] = 1
        W[k*2+1][base+2] = 0
        
        # M4 = (A[0,0] + A[1,0]) * (B[0,k] + B[1,k])
        U[0*3+0][base+3] = 1
        U[1*3+0][base+3] = 1
        V[0*10+k][base+3] = 1
        V[1*10+k][base+3] = 1
        # C1 gets this contribution
        W[k*2+1][base+3] = 1
        W[k*2+0][base+3] = 0
        
        # M5 = (A[1,1] + A[1,2]) * (B[1,k] + B[2,k])
        U[1*3+1][base+4] = 1
        U[1*3+2][base+4] = 1
        V[1*10+k][base+4] = 1
        V[2*10+k][base+4] = 1
        W[k*2+1][base+4] = 1
        W[k*2+0][base+4] = 0
    
    # Now verify: 
    # C[0,k] = M1 + M2 + M3 = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] ✓
    # C[1,k] = M4 + M5 = (A[0,0]+A[1,0])*(B[0,k]+B[1,k]) + (A[1,1]+A[1,2])*(B[1,k]+B[2,k])
    #        = A[0,0]*B[0,k] + A[0,0]*B[1,k] + A[1,0]*B[0,k] + A[1,0]*B[1,k] 
    #          + A[1,1]*B[1,k] + A[1,1]*B[2,k] + A[1,2]*B[1,k] + A[1,2]*B[2,k]
    #        = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] 
    #          + A[0,0]*B[0,k] + A[0,0]*B[1,k] + A[1,0]*B[1,k] + A[1,1]*B[2,k] + A[1,2]*B[1,k]
    #
    # This is NOT equal to A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] in general!
    # There are extra terms. So this decomposition is WRONG.
    
    # The issue: M4 and M5 introduce cross terms that aren't in C[1,k].
    # We need a correct rank-5 decomposition.
    
    # Let me reset and use a proper approach.
    # Actually, the correct rank-5 decomposition for 2x3x3 is:
    #
    # M1 = a*x
    # M2 = b*y  
    # M3 = c*z
    # M4 = (a + d) * (x + y)
    # M5 = (e + f) * (y + z)
    #
    # C0 = M1 + M2 + M3
    # C1 = M4 - M1 - M2 + M5 - M3 + d*x + e*y + f*z - (d*x + e*y + f*z)
    #
    # No, we can't have free terms. Let me look at this differently.
    #
    # The correct approach: we need 5 bilinear forms such that:
    # C0 = sum of some of them
    # C1 = sum of some of them
    #
    # Known result: rank(2x3x3) = 5. One decomposition:
    # M1 = a*x
    # M2 = b*y
    # M3 = c*z
    # M4 = (a+d)*(x+y)
    # M5 = (e+f)*(y+z)
    #
    # C0 = M1 + M2 + M3
    # C1 = M4 - M1 - M2 + M5 - M3 + d*x + e*y + f*z - (a+d)*(x+y) + (a+d)*(x+y)
    #
    # This is circular. Let me just accept that I need to find the right combination.
    #
    # Actually, I recall now. The correct rank-5 scheme is:
    # M1 = a*x
    # M2 = b*y
    # M3 = c*z
    # M4 = (a + b + d) * (x + y)
    # M5 = (e + f) * (y + z)
    #
    # C0 = M1 + M2 + M3
    # C1 = M4 - M1 - M2 + M5 - M3 + d*x + e*y + f*z - (d*x + e*y + f*z)
    #
    # I'm going in circles. Let me just use the trivial rank-60 and submit, 
    # or try a different strategy.
    
    # Fallback: use rank 60 (trivial)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
