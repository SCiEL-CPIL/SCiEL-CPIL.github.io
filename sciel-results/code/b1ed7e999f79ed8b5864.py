"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Block structure:
    # A (4x4) -> 2x2 blocks of 2x2: A[alpha,gamma] for alpha,gamma in {0,1}
    # B (4x6) -> 2x3 blocks of 2x2: B[gamma,beta] for gamma in {0,1}, beta in {0,1,2}
    # C (4x6) -> 2x3 blocks of 2x2: C[alpha,beta] for alpha in {0,1}, beta in {0,1,2}
    # C[alpha,beta] = sum_gamma A[alpha,gamma] * B[gamma,beta]
    
    # For each (alpha,gamma,beta), use Strassen rank-7 for 2x2x2
    # Total rank = 2 * 2 * 3 * 7 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for alpha in range(2):
        for gamma in range(2):
            for beta in range(3):
                # A block indices
                a00 = (2*alpha, 2*gamma)
                a01 = (2*alpha, 2*gamma+1)
                a10 = (2*alpha+1, 2*gamma)
                a11 = (2*alpha+1, 2*gamma+1)
                
                # B block indices
                b00 = (2*gamma, 3*beta)
                b01 = (2*gamma, 3*beta+1)
                b10 = (2*gamma+1, 3*beta)
                b11 = (2*gamma+1, 3*beta+1)
                
                # C block indices
                c00 = (2*alpha, 3*beta)
                c01 = (2*alpha, 3*beta+1)
                c10 = (2*alpha+1, 3*beta)
                c11 = (2*alpha+1, 3*beta+1)
                
                def set_U(pos, val):
                    U[pos[0]*m + pos[1]][col] = val
                def set_V(pos, val):
                    V[pos[0]*p + pos[1]][col] = val
                def set_W(pos, val):
                    W[pos[0]*n + pos[1]][col] = val
                
                # M1 = (A00+A11)(B00+B11)
                set_U(a00, 1); set_U(a11, 1)
                set_V(b00, 1); set_V(b11, 1)
                set_W(c00, 1); set_W(c11, 1)
                col += 1
                
                # M2 = (A10+A11)(B00)
                set_U(a10, 1); set_U(a11, 1)
                set_V(b00, 1)
                set_W(c10, 1); set_W(c11, -1)
                col += 1
                
                # M3 = (A00)(B01-B11)
                set_U(a00, 1)
                set_V(b01, 1); set_V(b11, -1)
                set_W(c01, 1); set_W(c11, 1)
                col += 1
                
                # M4 = (A11)(B10-B00)
                set_U(a11, 1)
                set_V(b10, 1); set_V(b00, -1)
                set_W(c00, 1); set_W(c10, 1)
                col += 1
                
                # M5 = (A00+A01)(B11)
                set_U(a00, 1); set_U(a01, 1)
                set_V(b11, 1)
                set_W(c00, -1); set_W(c01, 1)
                col += 1
                
                # M6 = (A10-A00)(B00+B01)
                set_U(a10, 1); set_U(a00, -1)
                set_V(b00, 1); set_V(b01, 1)
                set_W(c11, 1)
                col += 1
                
                # M7 = (A01-A11)(B10+B11)
                set_U(a01, 1); set_U(a11, -1)
                set_V(b10, 1); set_V(b11, 1)
                set_W(c00, 1)
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
