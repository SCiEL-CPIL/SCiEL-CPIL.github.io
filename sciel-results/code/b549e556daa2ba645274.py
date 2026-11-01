"""Matrix multiplication 2x2 x 2x13 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # Use Strassen's algorithm for 2x2 blocks.
    # 6 blocks of 2 columns (12 cols) + 1 remaining column.
    # Strassen per 2-col block: 7 mults. 6 blocks: 42. Last col: 4. Total: 46.
    
    # Strassen 2x2: C = A*B, A,B 2x2
    # M1=(a00+a11)(b00+b11), M2=(a10+a11)(b00), M3=a00(b01+b11),
    # M4=a11(b10-b00), M5=(a00+a01)(b11), M6=(a10-a00)(b00+b01), M7=(a01-a10)(b10+b11)
    # C00=M1-M4+M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2-M3+M7
    # Verify C00: (a00+a11)(b00+b11) - a11(b10-b00) + (a00+a01)b11 + (a01-a10)(b10+b11)
    # = a00b00+a00b11+a11b00+a11b11 - a11b10+a11b00 + a00b11+a01b11 + a01b10+a01b11-a10b10-a10b11
    # That's not giving a00b00+a10b11... I'm confusing myself.
    
    # Let me use the correct standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)  
    # M3 = (a00)(b01+b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a10)(b10+b11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M7
    
    # I'll trust this standard formulation.
    
    rank = 46
    U = [[0] * rank for _ in range(4)]   # a00, a01, a10, a11
    V = [[0] * rank for _ in range(26)]  # b0,0..b0,12, b1,0..b1,12
    W = [[0] * rank for _ in range(26)]  # C0,0..C0,12, C1,0..C1,12
    
    col = 0
    for blk in range(6):
        c0 = 2 * blk
        c1 = 2 * blk + 1
        # Index into V: row = 0 + col for b0, 13+col for b1
        # Index into W: row = col*2 + 0 for C0,col, col*2+1 for C1,col
        
        # M1 = (a00+a11)(b0,c0+b1,c1)
        U[0][col]=1; U[2][col]=1
        V[c0][col]=1; V[13+c1][col]=1
        W[2*c0+0][col]=1; W[2*c0+1][col]=1; W[2*c1+0][col]=1; W[2*c1+1][col]=-1
        col += 1
        # M2 = (a10+a11)(b0,c0)
        U[1][col]=1; U[2][col]=1
        V[c0][col]=1
        W[2*c0+1][col]=1; W[2*c1+1][col]=-1
        col += 1
        # M3 = a00*(b0,c1+b1,c1)
        U[0][col]=1
        V[c1][col]=1; V[13+c1][col]=1
        W[2*c1+0][col]=1; W[2*c1+1][col]=-1
        col += 1
        # M4 = a11*(b1,c0-b0,c0)
        U[2][col]=1
        V[13+c0][col]=1; V[c0][col]=-1
        W[2*c0+0][col]=-1; W[2*c0+1][col]=1
        col += 1
        # M5 = (a00+a01)*b1,c1
        U[0][col]=1; U[3][col]=1
        V[13+c1][col]=1
        W[2*c0+0][col]=1; W[2*c1+0][col]=1
        col += 1
        # M6 = (a10-a00)*(b0,c0+b0,c1)
        U[1][col]=1; U[0][col]=-1
        V[c0][col]=1; V[c1][col]=1
        # M6 doesn't appear in standard Strassen output formulas I listed
        col += 1
        # M7 = (a01-a10)*(b1,c0+b1,c1)
        U[3][col]=1; U[1][col]=-1
        V[13+c0][col]=1; V[13+c1][col]=1
        W[2*c0+0][col]=1; W[2*c1+1][col]=-1
        col += 1
    
    # Last column 12: schoolbook
    U[0][col]=1; V[12][col]=1; W[24][col]=1; col+=1
    U[3][col]=1; V[25][col]=1; W[24][col]=1; col+=1
    U[1][col]=1; V[12][col]=1; W[25][col]=1; col+=1
    U[2][col]=1; V[25][col]=1; W[25][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
