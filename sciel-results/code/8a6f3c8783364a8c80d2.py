"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    def add_strassen(ar0, ar1, ac0, ac1, br0, br1, bc0, bc1, cr0, cr1, cc0, cc1):
        """Add Strassen rank-7 terms for 2x2x2 block.
        A rows ar0,ar1 cols ac0,ac1
        B rows br0,br1 cols bc0,bc1
        C rows cr0,cr1 cols cc0,cc1
        """
        nonlocal col
        # a=A[ar0,ac0], b=A[ar0,ac1], c=A[ar1,ac0], d=A[ar1,ac1]
        # e=B[br0,bc0], f=B[br0,bc1], g=B[br1,bc0], h=B[br1,bc1]
        
        # M1=(a+d)(e+h): C[cr0,cc0]+=1, C[cr1,cc1]-=1
        U[ar0*m+ac0][col] = 1
        U[ar1*m+ac1][col] = 1
        V[br0*p+bc0][col] = 1
        V[br1*p+bc1][col] = 1
        W[cc0*n+cr0][col] = 1
        W[cc1*n+cr1][col] = -1
        col += 1
        
        # M2=(c+d)e: C[cr1,cc0]+=1
        U[ar1*m+ac0][col] = 1
        U[ar1*m+ac1][col] = 1
        V[br0*p+bc0][col] = 1
        W[cc0*n+cr1][col] = 1
        col += 1
        
        # M3=a(f-h): C[cr0,cc1]+=1
        U[ar0*m+ac0][col] = 1
        V[br0*p+bc1][col] = 1
        V[br1*p+bc1][col] = -1
        W[cc1*n+cr0][col] = 1
        col += 1
        
        # M4=d(g-e): C[cr0,cc0]+=1, C[cr1,cc0]+=1
        U[ar1*m+ac1][col] = 1
        V[br1*p+bc0][col] = 1
        V[br0*p+bc0][col] = -1
        W[cc0*n+cr0][col] = 1
        W[cc0*n+cr1][col] = 1
        col += 1
        
        # M5=(a+b)h: C[cr0,cc0]-=1, C[cr0,cc1]+=1
        U[ar0*m+ac0][col] = 1
        U[ar0*m+ac1][col] = 1
        V[br1*p+bc1][col] = 1
        W[cc0*n+cr0][col] = -1
        W[cc1*n+cr0][col] = 1
        col += 1
        
        # M6=(c-a)(e+f): C[cr1,cc1]+=1
        U[ar1*m+ac0][col] = 1
        U[ar0*m+ac0][col] = -1
        V[br0*p+bc0][col] = 1
        V[br0*p+bc1][col] = 1
        W[cc1*n+cr1][col] = 1
        col += 1
        
        # M7=(b-d)(g+h): C[cr0,cc0]+=1
        U[ar0*m+ac1][col] = 1
        U[ar1*m+ac1][col] = -1
        V[br1*p+bc0][col] = 1
        V[br1*p+bc1][col] = 1
        W[cc0*n+cr0][col] = 1
        col += 1
    
    def add_trivial_1x2(ar, ac0, ac1, br0, br1, bc0, bc1, cr, cc0, cc1):
        """Add trivial rank-4 terms for 1x2 * 2x2 -> 1x2."""
        nonlocal col
        # C[cr,cc0] = A[ar,ac0]*B[br0,bc0] + A[ar,ac1]*B[br1,bc0]
        # C[cr,cc1] = A[ar,ac0]*B[br0,bc1] + A[ar,ac1]*B[br1,bc1]
        U[ar*m+ac0][col] = 1
        V[br0*p+bc0][col] = 1
        W[cc0*n+cr][col] = 1
        col += 1
        
        U[ar*m+ac1][col] = 1
        V[br1*p+bc0][col] = 1
        W[cc0*n+cr][col] = 1
        col += 1
        
        U[ar*m+ac0][col] = 1
        V[br0*p+bc1][col] = 1
        W[cc1*n+cr][col] = 1
        col += 1
        
        U[ar*m+ac1][col] = 1
        V[br1*p+bc1][col] = 1
        W[cc1*n+cr][col] = 1
        col += 1
    
    for sc in range(2):  # C column block
        cc0, cc1 = 2*sc, 2*sc+1
        for sa in range(2):  # A col block / B row block
            ac0, ac1 = 2*sa, 2*sa+1
            br0, br1 = 2*sa, 2*sa+1
            
            # Top 2 rows (0,1) use Strassen
            add_strassen(0, 1, ac0, ac1, br0, br1, cc0, cc1, 0, 1, cc0, cc1)
            # Bottom row (2) uses trivial
            add_trivial_1x2(2, ac0, ac1, br0, br1, cc0, cc1, 2, cc0, cc1)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
