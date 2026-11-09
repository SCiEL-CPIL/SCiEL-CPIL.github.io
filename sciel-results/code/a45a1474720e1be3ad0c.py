import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Try to build a rank-59 decomposition using Strassen on a 2x2x2 block
    # The 2x2x2 block covers i in {0,1}, j in {0,1}, k in {0,1}
    # Standard rank for this block: 8, Strassen rank: 7, saving 1
    # Total rank: 60 - 8 + 7 = 59
    
    # First, let's verify the Strassen 2x2x2 decomposition
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_s = [[Fraction(0)]*7 for _ in range(4)]
    V_s = [[Fraction(0)]*7 for _ in range(4)]
    W_s = [[Fraction(0)]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11) -> contributes to c00 and c11
    U_s[0][0]=1; U_s[3][0]=1
    V_s[0][0]=1; V_s[3][0]=1
    W_s[0][0]=1; W_s[3][0]=1
    
    # M2 = (a10+a11)b00 -> contributes to c10
    U_s[2][1]=1; U_s[3][1]=1
    V_s[0][1]=1
    W_s[2][1]=1
    
    # M3 = a00(b10-b11) -> contributes to c01
    U_s[0][2]=1
    V_s[2][2]=1; V_s[3][2]=-1
    W_s[1][2]=1
    
    # M4 = a11(b01-b00) -> contributes to c00
    U_s[3][3]=1
    V_s[1][3]=1; V_s[0][3]=-1
    W_s[0][3]=1
    
    # M5 = (a00+a01)b11 -> contributes to c01
    U_s[0][4]=1; U_s[1][4]=1
    V_s[3][4]=1
    W_s[1][4]=1
    
    # M6 = (a10-a00)(b00+b01) -> contributes to c11
    U_s[2][5]=1; U_s[0][5]=-1
    V_s[0][5]=1; V_s[1][5]=1
    W_s[3][5]=1
    
    # M7 = (a01-a11)(b10+b11) -> contributes to c00
    U_s[1][6]=1; U_s[3][6]=-1
    V_s[2][6]=1; V_s[3][6]=1
    W_s[0][6]=1
    
    # Verify the Strassen decomposition
    valid_strassen = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = Fraction(0)
                for c in range(7):
                    s += U_s[i*2+j][c] * V_s[j*2+k][c] * W_s[k*2+i][c]
                if s != 1:
                    valid_strassen = False
                    break
            if not valid_strassen:
                break
        if not valid_strassen:
            break
    
    if valid_strassen:
        # Build the full rank-59 decomposition
        rank = 59
        U = [[Fraction(0)]*rank for _ in range(n*m)]
        V = [[Fraction(0)]*rank for _ in range(m*p)]
        W = [[Fraction(0)]*rank for _ in range(p*n)]
        
        # Strassen block: i in {0,1}, j in {0,1}, k in {0,1}
        # These 8 tensor positions are handled by the 7 Strassen terms
        col = 0
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+j][col] = U_s[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[j*p+k][col] = V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[k*n+i][col] = W_s[k*2+i][c]
            col += 1
        
        # Remaining tensor positions: all (i,j,k) NOT in the Strassen block
        # i in {0,1}, j in {0,1}, k in {0,1} are handled
        # Remaining: 
        # - i in {0,1}, j in {0,1}, k in {2,3,4} -> 2*2*3 = 12
        # - i in {0,1}, j in {2,3}, k in {0,1,2,3,4} -> 2*2*5 = 20
        # - i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4} -> 1*4*5 = 20
        # Total remaining: 12+20+20 = 52
        # Total: 7 + 52 = 59 ✓
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue  # Handled by Strassen
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        
        # Verify the full decomposition
        valid_full = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        valid_full = False
                        break
                if not valid_full:
                    break
            if not valid_full:
                break
        
        if valid_full:
            # Convert to plain integers/half-integers
            U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in U]
            V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in V]
            W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook rank 60
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
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
