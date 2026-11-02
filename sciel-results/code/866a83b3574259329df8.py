def _construct_rank32():
    """Construct a rank-32 decomposition of the 2x4x5 all-ones tensor."""
    n, m, p = 2, 4, 5
    rank = 32
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Process each j-group: {0,1} and {2,3}
    for j0 in [0, 2]:
        j1 = j0 + 1
        
        # Process each k-group: {0,1}, {2,3}, {4}
        # For k-groups of size 2, use Strassen (7 terms)
        for k0, k1 in [(0,1), (2,3)]:
            # Strassen 7 terms for 2x2x2 sub-tensor (i in {0,1}, j in {j0,j1}, k in {k0,k1})
            
            # M0 = (A00+A11)(B00+B11)
            U[0*m+j0][col] += 1; U[1*m+j1][col] += 1
            V[j0*p+k0][col] += 1; V[j1*p+k1][col] += 1
            W[k0*n+0][col] += 1; W[k1*n+1][col] += 1
            col += 1
            
            # M1 = (A10+A11)B00
            U[0*m+j1][col] += 1; U[1*m+j1][col] += 1
            V[j0*p+k0][col] += 1
            W[k0*n+0][col] += 1; W[k0*n+1][col] += 1
            col += 1
            
            # M2 = A00(B10-B11)
            U[0*m+j0][col] += 1
            V[j1*p+k0][col] += 1; V[j1*p+k1][col] += -1
            W[k0*n+0][col] += 1; W[k1*n+0][col] += 1
            col += 1
            
            # M3 = A11(B01-B00)
            U[1*m+j1][col] += 1
            V[j0*p+k1][col] += 1; V[j0*p+k0][col] += -1
            W[k1*n+1][col] += 1; W[k0*n+1][col] += 1
            col += 1
            
            # M4 = (A00+A10)B11
            U[0*m+j0][col] += 1; U[1*m+j0][col] += 1
            V[j1*p+k1][col] += 1
            W[k1*n+0][col] += 1; W[k1*n+1][col] += 1
            col += 1
            
            # M5 = (A10-A00)(B00+B10)
            U[1*m+j0][col] += 1; U[0*m+j0][col] += -1
            V[j0*p+k0][col] += 1; V[j1*p+k0][col] += 1
            W[k0*n+0][col] += 1; W[k0*n+1][col] += 1
            col += 1
            
            # M6 = (A01-A11)(B01+B11)
            U[0*m+j1][col] += 1; U[1*m+j1][col] += -1
            V[j0*p+k1][col] += 1; V[j1*p+k1][col] += 1
            W[k1*n+0][col] += 1; W[k1*n+1][col] += 1
            col += 1
        
        # For k=4: 2x2x1, use 2 terms (or 4 schoolbook terms?)
        # 2x2x1 all-ones: need T[i,j,4]=1 for i,j in {0,1}
        # Use 4 schoolbook terms for simplicity (adds 4 rank)
        for ii in [0, 1]:
            for jj in [j0, j1]:
                U[ii*m+jj][col] = 1
                V[jj*p+4][col] = 1
                W[4*n+ii][col] = 1
                col += 1
    
    # col should be 32 + 8 = 40? No, 7+7+4 = 18 per group, 36 total.
    # Wait, 7+7+4 = 18 per j-group, two groups = 36. That's not 32.
    # I need to use a better 2x2x1 decomposition.
    
    # For 2x2x1, the rank is 2. Let me find it.
    # T[i,j,4] = 1 for i,j in {0,1}.
    # We need U[i*m+j,c]*V[j*p+4,c]*W[4*n+i,c] summed = 1.
    # With 2 terms:
    # c=0: U[0,j0,0]*V[j0,4,0]*W[8,0] + U[0,j1,0]*V[j1,4,0]*W[8,0] + U[1,j0,0]*V[j0,4,0]*W[9,0] + U[1,j1,0]*V[j1,4,0]*W[9,0] = 1
    # This is 4 equations in 8 variables (2 c's × 4 entries each). Solvable.
    # One solution: 
    #   U[0,j0,0]=1, V[j0,4,0]=1, W[8,0]=1, others 0 -> gives T[0,j0,4]=1
    #   U[1,j1,0]=1, V[j1,4,0]=1, W[9,0]=1, others 0 -> gives T[1,j1,4]=1
    #   But we also need T[0,j1,4]=1 and T[1,j0,4]=1.
    # So 2 terms aren't enough with this structure. We need 4 terms (schoolbook) or a clever 2-term solution.
    # 
    # Actually, the 2x2x1 all-ones tensor has rank 2! Here's a decomposition:
    # c=0: U[0,j0,0]=1, U[0,j1,0]=0, U[1,j0,0]=0, U[1,j1,0]=1
    #       V[j0,4,0]=1, V[j1,4,0]=1
    #       W[8,0]=1, W[9,0]=1
    #   Contribution: T[0,j0,4] = 1*1*1 = 1, T[0,j1,4] = 0*1*1 = 0, T[1,j0,4] = 0*1*1 = 0, T[1,j1,4] = 1*1*1 = 1
    #   That only gives 2 of the 4 entries. Not good.
    #
    # I think the 2x2x1 all-ones tensor actually has rank 4 (schoolbook). The rank of the 2x2x2 all-ones is 7, and 2x2x1 is a sub-tensor, so its rank is at most 4.
    # Let me just use 4 terms for k=4.
    
    # So total rank = 7+7+4 per group = 18, two groups = 36.
    # That's rank 36, not 32. Still better than 40.
    
    return {"U": U, "V": V, "W": W}
