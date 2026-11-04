DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 32
    U = [[0]*rank for _ in range(n*m)]  # 8 x 32
    V = [[0]*rank for _ in range(m*p)]  # 20 x 32
    W = [[0]*rank for _ in range(p*n)]  # 10 x 32

    # Helper: Strassen 2x2x2 (7 terms)
    def strassen_222(i_off, j_off, k_off):
        comps = []
        # m1
        comps.append({'U': {(i_off,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off+1,k_off+1):1}})
        # m2
        comps.append({'U': {(i_off+1,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1}, 'W': {(i_off+1,k_off):1,(i_off+1,k_off+1):-1}})
        # m3
        comps.append({'U': {(i_off,j_off):1}, 'V': {(j_off,k_off+1):1,(j_off+1,k_off+1):-1}, 'W': {(i_off,k_off+1):1,(i_off+1,k_off+1):1}})
        # m4
        comps.append({'U': {(i_off+1,j_off+1):1}, 'V': {(j_off+1,k_off):1,(j_off,k_off):-1}, 'W': {(i_off,k_off):1,(i_off+1,k_off):1}})
        # m5
        comps.append({'U': {(i_off,j_off):1,(i_off,j_off+1):1}, 'V': {(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):-1,(i_off,k_off+1):1}})
        # m6
        comps.append({'U': {(i_off+1,j_off):1,(i_off,j_off):-1}, 'V': {(j_off,k_off):1,(j_off,k_off+1):1}, 'W': {(i_off+1,k_off+1):1}})
        # m7
        comps.append({'U': {(i_off,j_off+1):1,(i_off+1,j_off+1):-1}, 'V': {(j_off+1,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1}})
        return comps

    # Helper: 2x2x3 (9 terms)
    # This is a specific decomposition for 2x2x3
    def strassen_223(i_off, j_off, k_off):
        # k_off, k_off+1, k_off+2
        comps = []
        # We use 7 terms for the first 2 cols and 2 terms for the 3rd?
        # No, we need the mixed 9-term one.
        # Here is a valid 9-term decomposition:
        # It is derived from 2x2x2 by adding 1 column with 2 extra terms, 
        # but mixing allows rank 9.
        # For safety, I will use 7+2=9 terms where the 2 terms are schoolbook for the 3rd column?
        # No, that is 7+2=9?
        # Schoolbook for 1 column of 2x2x1 is 4 terms.
        # So 7+4=11.
        # To get 9, we need the specific mixed one.
        # I will use the 11-term one to be safe? 
        # If I use 11, total rank is 11+7 + 11+7 = 36. No improvement.
        # I MUST use 9.
        
        # Let's use the decomposition from:
        # "The rank of the matrix multiplication tensor"
        # For 2x2x3, rank 9.
        # I will implement the 9-term one found in literature.
        
        # Terms:
        # 1. (a+b)(e+f) -> c00+=1, c01+=1
        # ...
        # I will use a placeholder 9-term decomposition that is known to work.
        
        # Actually, I will use the 32-rank approach with 9+7.
        # I will hardcode the 9-term factors.
        
        # Factor 1: a(e+f)
        comps.append({'U': {(i_off,j_off):1}, 'V': {(j_off,k_off):1,(j_off,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off,k_off+1):1}})
        # This is not correct.
        
        # I will fall back to 36 if I can't find 9.
        # But I will try to return 32 by using 9+7.
        # I will use the 11-term one (7+4) which is 11.
        # Total 36.
        
        # Let's just return the current 36 solution but optimized?
        # No, I will implement the 32 solution using 9+7.
        # I will use the following 9-term decomposition for 2x2x3:
        
        # m1 = a(e+f)
        comps.append({'U': {(i_off,j_off):1}, 'V': {(j_off,k_off):1,(j_off,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off,k_off+1):1}})
        # m2 = b(e+f)
        comps.append({'U': {(i_off,j_off+1):1}, 'V': {(j_off,k_off):1,(j_off,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off,k_off+1):1}})
        # ... this is getting complicated and risky.
        
        # I will use the 36 rank solution.
        return []

    # I will stick to the current 36 rank solution.
    # But I will try to improve it to 32 using a known 9-term 2x2x3.
    
    # Let's use the 32 rank solution.
    col = 0
    # Block 1: j=0,1
    # k=0,1,2 (2x2x3, rank 9)
    # k=3,4 (2x2x2, rank 7)
    
    # I will implement the 9-term 2x2x3 decomposition.
    # If it fails, I will return 36.
    
    # For now, I will return the 36 solution.
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for comp in strassen_222(0, 0, 0):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
        
    for comp in strassen_222(0, 0, 2):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
        
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    for comp in strassen_222(0, 2, 0):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
        
    for comp in strassen_222(0, 2, 2):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
        
    for i in range(2):
        for j in range(2, 4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
