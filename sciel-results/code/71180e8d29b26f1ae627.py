"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(3):
        base = block * 7
        b0 = block * 2
        b1 = block * 2 + 1
        
        # Term 0: A[0,0]*B[0,b0] -> C[b0,0]
        U[0*3+0][base+0] = 1
        V[0*6+b0][base+0] = 1
        W[b0*3+0][base+0] = 1
        
        # Term 1: A[0,1]*B[1,b0] -> C[b0,0]
        U[0*3+1][base+1] = 1
        V[1*6+b0][base+1] = 1
        W[b0*3+0][base+1] = 1
        
        # Term 2: A[0,2]*B[2,b0] -> C[b0,0]
        U[0*3+2][base+2] = 1
        V[2*6+b0][base+2] = 1
        W[b0*3+0][base+2] = 1
        
        # Term 3: A[0,0]*B[0,b1] -> C[b1,0]
        U[0*3+0][base+3] = 1
        V[0*6+b1][base+3] = 1
        W[b1*3+0][base+3] = 1
        
        # Term 4: A[0,1]*B[1,b1] -> C[b1,0]
        U[0*3+1][base+4] = 1
        V[1*6+b1][base+4] = 1
        W[b1*3+0][base+4] = 1
        
        # Term 5: A[0,2]*B[2,b1] -> C[b1,0]
        U[0*3+2][base+5] = 1
        V[2*6+b1][base+5] = 1
        W[b1*3+0][base+5] = 1
        
        # Term 6: Combined term for C[1,b0], C[1,b1], C[2,b0], C[2,b1]
        # A part: A[1,0]+A[2,0], A[1,1]+A[2,1], A[1,2]+A[2,2]
        # B part: B[0,b0]+B[0,b1], B[1,b0]+B[1,b1], B[2,b0]+B[2,b1]
        # C part: C[1,b0]+C[1,b1]+C[2,b0]+C[2,b1]
        # This gives: (A[1,0]+A[2,0])(B[0,b0]+B[0,b1]) + (A[1,1]+A[2,1])(B[1,b0]+B[1,b1]) + (A[1,2]+A[2,2])(B[2,b0]+B[2,b1])
        # = A[1,0]B[0,b0]+A[1,0]B[0,b1]+A[2,0]B[0,b0]+A[2,0]B[0,b1]
        #   + A[1,1]B[1,b0]+A[1,1]B[1,b1]+A[2,1]B[1,b0]+A[2,1]B[1,b1]
        #   + A[1,2]B[2,b0]+A[1,2]B[2,b1]+A[2,2]B[2,b0]+A[2,2]B[2,b1]
        # We need: C[1,b0]=A[1,0]B[0,b0]+A[1,1]B[1,b0]+A[1,2]B[2,b0]
        #          C[1,b1]=A[1,0]B[0,b1]+A[1,1]B[1,b1]+A[1,2]B[2,b1]
        #          C[2,b0]=A[2,0]B[0,b0]+A[2,1]B[1,b0]+A[2,2]B[2,b0]
        #          C[2,b1]=A[2,0]B[0,b1]+A[2,1]B[1,b1]+A[2,2]B[2,b1]
        # The expansion gives all 12 terms, but we only need 4 specific combinations.
        # We can use this as-is if we accept the extra terms, but they don't cancel.
        # Actually this doesn't work directly. Let me use a different approach.
        
        # Better approach for term 6: Use the fact that we can have multi-entry vectors
        # A[6]: A[1,0] at idx 3, A[1,1] at idx 4, A[1,2] at idx 5, A[2,0] at idx 6, A[2,1] at idx 7, A[2,2] at idx 8
        # V[6]: B[0,b0] at idx 0*6+b0, B[1,b0] at idx 1*6+b0, B[2,b0] at idx 2*6+b0
        #        B[0,b1] at idx 0*6+b1, B[1,b1] at idx 1*6+b1, B[2,b1] at idx 2*6+b1
        # W[6]: C[1,b0] at idx b0*3+1, C[2,b0] at idx b0*3+2, C[1,b1] at idx b1*3+1, C[2,b1] at idx b1*3+2
        
        # Let's set:
        # A[6] = A[1,0]+A[1,1]+A[1,2] (row 1 of A)
        # V[6] = B[0,b0]+B[1,b0]+B[2,b0] (column b0 of B)
        # This gives sum_j A[1,j]*B[j,b0] = C[1,b0]
        
        # But we also need C[2,b0], C[1,b1], C[2,b1].
        # We only have 1 term left, so we need to pack 4 values into it.
        
        # Alternative: use rank 7 differently.
        # t0-t2: C[0,b0] (3 terms)
        # t3-t5: C[0,b1] (3 terms)  
        # t6: Must handle C[1,b0], C[1,b1], C[2,b0], C[2,b1] in one term
        
        # For t6 to work, we need:
        # (sum_a alpha_a * A[a]) * (sum_b beta_b * B[b]) = C[1,b0]+C[1,b1]+C[2,b0]+C[2,b1]
        # This is impossible with single coefficient vectors unless we use the structure.
        
        # Actually, let's reconsider. Maybe rank 7 isn't achievable for 3x2 in this way.
        # Let me just use rank 8 per block (total 24) to be safe, or find the correct rank-7.
        
        # For now, let's set term 6 to handle C[1,b0] only, and we'll need more terms.
        # Actually, let me just return schoolbook for correctness, then optimize.
        
        U[1*3+0][base+6] = 1
        U[1*3+1][base+6] = 1
        U[1*3+2][base+6] = 1
        V[0*6+b0][base+6] = 1
        V[1*6+b0][base+6] = 1
        V[2*6+b0][base+6] = 1
        W[b0*3+1][base+6] = 1
        
        # This only gives C[1,b0]. We're missing C[1,b1], C[2,b0], C[2,b1].
        # So rank 7 per block is not working with this simple construction.
        
    # This approach is flawed. Let me just return schoolbook rank 54 for correctness.
    # The task requires exact decomposition, so correctness is paramount.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
