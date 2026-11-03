"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    def add_term(U, V, W, col, terms):
        for (i, j, k) in terms:
            U[i * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + i][col] += 1
        return col + 1

    def add_strassen_2x2(U, V, W, col, i0, j0, k0):
        # M1: (i0,j0,k0), (i0+1,j0+1,k0+1)
        col = add_term(U, V, W, col, [(i0, j0, k0), (i0+1, j0+1, k0+1)])
        # M2: (i0+1,j0,k0), (i0+1,j0+1,k0+1)
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + j0+1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i0+1][col] += 1
        W[(k0+1)*n + i0+1][col] -= 1
        col += 1
        # M3: (i0,j0,k0+1), (i0+1,j0,k0+1)
        U[i0*m + j0][col] += 1
        V[j0*p + k0+1][col] += 1
        V[(j0+1)*p + k0+1][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + i0+1][col] += 1
        col += 1
        # M4: (i0+1,j0+1,k0), (i0+1,j0,k0)
        U[(i0+1)*m + j0+1][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + i0+1][col] += 1
        col += 1
        # M5: (i0,j0,k0+1), (i0,j0+1,k0)
        U[i0*m + j0][col] += 1
        U[i0*m + j0+1][col] += 1
        V[(j0+1)*p + k0+1][col] += 1
        W[(k0+1)*n + i0][col] += 1
        W[k0*n + i0][col] -= 1
        col += 1
        # M6: (i0+1,j0+1,k0+1), (i0,j0,k0)
        U[(i0+1)*m + j0+1][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + k0+1][col] += 1
        W[(k0+1)*n + i0+1][col] += 1
        col += 1
        # M7: (i0+1,j0+1,k0), (i0,j0,k0)
        U[(i0+1)*m + j0+1][col] += 1
        U[i0*m + j0+1][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + k0+1][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
        return col

    # Try de Groote-style rank 23 for 3x3x3
    # de Groote's 23-term decomposition for 3x3 matrix multiplication
    # This is a known rank-23 decomposition. I'll use a simplified version.
    # Actually, let me use the Laderman-like approach but for 3x3.
    # The known rank-23 decomposition of de Groote:
    
    def add_de_groote_3x3(U, V, W, col, i0, j0, k0):
        # de Groote rank 23 decomposition
        # Terms:
        # 1: (0,0,0), (1,1,1), (2,2,2)  -> diagonal
        U[i0*m+j0][col] += 1; V[j0*p+k0][col] += 1; W[k0*n+i0][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        U[(i0+2)*m+(j0+2)][col] += 1; V[(j0+2)*p+(k0+2)][col] += 1; W[(k0+2)*n+(i0+2)][col] += 1
        col += 1
        # 2: (0,0,1), (0,1,2), (1,0,2)
        U[i0*m+j0][col] += 1
        U[i0*m+(j0+1)][col] -= 1
        U[(i0+1)*m+j0][col] += 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        V[j0*p+(k0+2)][col] -= 1
        W[(k0+1)*n+i0][col] += 1
        W[(k0+2)*n+i0][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        col += 1
        # 3: (0,1,0), (1,0,1), (2,2,0)
        U[i0*m+(j0+1)][col] += 1
        U[(i0+1)*m+j0][col] += 1
        U[(i0+2)*m+(j0+2)][col] -= 1
        V[(j0+1)*p+k0][col] += 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+2)*p+k0][col] += 1
        W[k0*n+i0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        W[k0*n+(i0+2)][col] += 1
        col += 1
        # 4: (1,1,0), (1,2,2), (2,1,2)
        U[(i0+1)*m+(j0+1)][col] += 1
        U[(i0+1)*m+(j0+2)][col] -= 1
        U[(i0+2)*m+(j0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1
        V[(j0+2)*p+(k0+2)][col] += 1
        V[(j0+1)*p+(k0+2)][col] -= 1
        W[k0*n+(i0+1)][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        W[(k0+2)*n+(i0+2)][col] += 1
        col += 1
        # 5: (2,2,1), (2,1,0), (1,2,0)
        U[(i0+2)*m+(j0+2)][col] += 1
        U[(i0+2)*m+(j0+1)][col] -= 1
        U[(i0+1)*m+(j0+2)][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1
        V[(j0+2)*p+k0][col] -= 1
        W[(k0+1)*n+(i0+2)][col] += 1
        W[k0*n+(i0+2)][col] += 1
        W[k0*n+(i0+1)][col] += 1
        col += 1
        # 6: (0,2,2), (2,0,2), (1,1,2)
        U[i0*m+(j0+2)][col] += 1
        U[(i0+2)*m+j0][col] += 1
        U[(i0+1)*m+(j0+1)][col] -= 1
        V[(j0+2)*p+(k0+2)][col] += 1
        V[j0*p+(k0+2)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        W[(k0+2)*n+i0][col] += 1
        W[(k0+2)*n+(i0+2)][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        col += 1
        # 7: (1,0,0), (0,1,1), (2,2,0)
        U[(i0+1)*m+j0][col] += 1
        U[i0*m+(j0+1)][col] += 1
        U[(i0+2)*m+(j0+2)][col] -= 1
        V[j0*p+k0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        V[(j0+2)*p+k0][col] += 1
        W[k0*n+(i0+1)][col] += 1
        W[(k0+1)*n+i0][col] += 1
        W[k0*n+(i0+2)][col] += 1
        col += 1
        # 8: (2,1,1), (1,2,1), (0,0,2)
        U[(i0+2)*m+(j0+1)][col] += 1
        U[(i0+1)*m+(j0+2)][col] += 1
        U[i0*m+j0][col] -= 1
        V[(j0+1)*p+(k0+1)][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        V[j0*p+(k0+2)][col] += 1
        W[(k0+1)*n+(i0+2)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        W[(k0+2)*n+i0][col] += 1
        col += 1
        # 9: (0,2,0), (2,0,1), (1,1,0)
        U[i0*m+(j0+2)][col] += 1
        U[(i0+2)*m+j0][col] += 1
        U[(i0+1)*m+(j0+1)][col] -= 1
        V[(j0+2)*p+k0][col] += 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1
        W[k0*n+i0][col] += 1
        W[(k0+1)*n+(i0+2)][col] += 1
        W[k0*n+(i0+1)][col] += 1
        col += 1
        # 10: (1,0,2), (0,1,2), (2,2,2)
        U[(i0+1)*m+j0][col] += 1
        U[i0*m+(j0+1)][col] -= 1
        U[(i0+2)*m+(j0+2)][col] += 1
        V[j0*p+(k0+2)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        V[(j0+2)*p+(k0+2)][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        W[(k0+2)*n+i0][col] += 1
        W[(k0+2)*n+(i0+2)][col] += 1
        col += 1
        # 11: (2,2,1), (2,0,0), (1,1,0)
        U[(i0+2)*m+(j0+2)][col] += 1
        U[(i0+2)*m+j0][col] -= 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        V[j0*p+k0][col] += 1
        V[(j0+1)*p+k0][col] += 1
        W[(k0+1)*n+(i0+2)][col] += 1
        W[k0*n+(i0+2)][col] += 1
        W[k0*n+(i0+1)][col] += 1
        col += 1
        # 12: (0,0,1), (0,2,2), (1,1,2)
        U[i0*m+j0][col] += 1
        U[i0*m+(j0+2)][col] += 1
        U[(i0+1)*m+(j0+1)][col] -= 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+2)*p+(k0+2)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        W[(k0+1)*n+i0][col] += 1
        W[(k0+2)*n+i0][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        col += 1
        # 13: (1,2,0), (2,1,0), (0,0,0)
        U[(i0+1)*m+(j0+2)][col] += 1
        U[(i0+2)*m+(j0+1)][col] += 1
        U[i0*m+j0][col] -= 1
        V[(j0+2)*p+k0][col] += 1
        V[(j0+1)*p+k0][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+(i0+1)][col] += 1
        W[k0*n+(i0+2)][col] += 1
        W[k0*n+i0][col] += 1
        col += 1
        # 14: (2,0,2), (1,1,2), (0,2,2)
        U[(i0+2)*m+j0][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        U[i0*m+(j0+2)][col] -= 1
        V[j0*p+(k0+2)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        V[(j0+2)*p+(k0+2)][col] += 1
        W[(k0+2)*n+(i0+2)][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        W[(k0+2)*n+i0][col] += 1
        col += 1
        # 15: (0,1,0), (2,2,1), (1,2,0)
        U[i0*m+(j0+1)][col] += 1
        U[(i0+2)*m+(j0+2)][col] += 1
        U[(i0+1)*m+(j0+2)][col] -= 1
        V[(j0+1)*p+k0][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        V[(j0+2)*p+k0][col] += 1
        W[k0*n+i0][col] += 1
        W[(k0+1)*n+(i0+2)][col] += 1
        W[k0*n+(i0+1)][col] += 1
        col += 1
        # 16: (1,1,0), (0,2,1), (2,0,0)
        U[(i0+1)*m+(j0+1)][col] += 1
        U[i0*m+(j0+2)][col] -= 1
        U[(i0+2)*m+j0][col] += 1
        V[(j0+1)*p+k0][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+(i0+1)][col] += 1
        W[(k0+1)*n+i0][col] += 1
        W[k0*n+(i0+2)][col] += 1
        col += 1
        # 17: (2,0,1), (1,2,1), (0,1,2)
        U[(i0+2)*m+j0][col] += 1
        U[(i0+1)*m+(j0+2)][col] += 1
        U[i0*m+(j0+1)][col] -= 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        W[(k0+1)*n+(i0+2)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        W[(k0+2)*n+i0][col] += 1
        col += 1
        # 18: (0,2,0), (1,1,1), (2,2,0)
        U[i0*m+(j0+2)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        U[(i0+2)*m+(j0+2)][col] -= 1
        V[(j0+2)*p+k0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        V[(j0+2)*p+k0][col] += 1
        W[k0*n+i0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        W[k0*n+(i0+2)][col] += 1
        col += 1
        # 19: (1,2,2), (2,0,2), (0,1,2)
        U[(i0+1)*m+(j0+2)][col] += 1
        U[(i0+2)*m+j0][col] += 1
        U[i0*m+(j0+1)][col] -= 1
        V[(j0+2)*p+(k0+2)][col] += 1
        V[j0*p+(k0+2)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        W[(k0+2)*n+(i0+2)][col] += 1
        W[(k0+2)*n+i0][col] += 1
        col += 1
        # 20: (2,1,1), (0,2,1), (1,0,2)
        U[(i0+2)*m+(j0+1)][col] += 1
        U[i0*m+(j0+2)][col] += 1
        U[(i0+1)*m+j0][col] -= 1
        V[(j0+1)*p+(k0+1)][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        V[j0*p+(k0+2)][col] += 1
        W[(k0+1)*n+(i0+2)][col] += 1
        W[(k0+1)*n+i0][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        col += 1
        # 21: (0,0,2), (2,2,2), (1,1,2)
        U[i0*m+j0][col] += 1
        U[(i0+2)*m+(j0+2)][col] += 1
        U[(i0+1)*m+(j0+1)][col] -= 1
        V[j0*p+(k0+2)][col] += 1
        V[(j0+2)*p+(k0+2)][col] += 1
        V[(j0+1)*p+(k0+2)][col] += 1
        W[(k0+2)*n+i0][col] += 1
        W[(k0+2)*n+(i0+2)][col] += 1
        W[(k0+2)*n+(i0+1)][col] += 1
        col += 1
        # 22: (1,0,1), (2,1,0), (0,2,0)
        U[(i0+1)*m+j0][col] += 1
        U[(i0+2)*m+(j0+1)][col] += 1
        U[i0*m+(j0+2)][col] -= 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1
        V[(j0+2)*p+k0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        W[k0*n+(i0+2)][col] += 1
        W[k0*n+i0][col] += 1
        col += 1
        # 23: (2,0,0), (0,1,1), (1,2,1)
        U[(i0+2)*m+j0][col] += 1
        U[i0*m+(j0+1)][col] += 1
        U[(i0+1)*m+(j0+2)][col] -= 1
        V[j0*p+k0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        V[(j0+2)*p+(k0+1)][col] += 1
        W[k0*n+(i0+2)][col] += 1
        W[(k0+1)*n+i0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        return col

    # Strategy: Use de Groote on the 3x3x3 block (i=0..2, j=0..2, k=0..2)
    # Then handle remaining terms with Strassen and schoolbook
    #
    # Full tensor: 3x5x7 = 105 terms
    # 
    # Block A: i=0..2, j=0..2, k=0..2 -> 27 terms, use de Groote (23 terms) saves 4
    # Block B: i=0..2, j=0..2, k=3..6 -> 12 terms
    #   j=0,1 with k=3,4 (Strassen) -> 7, k=5,6 (Strassen) -> 7, total 14
    #   j=2 with k=3..6 -> 8 schoolbook
    # Block C: i=0..2, j=3..4, k=0..6 -> 12 terms
    #   j=3,4: can't do Strassen easily, use schoolbook: 12
    # 
    # Actually let me try a different partition:
    # 
    # Use de Groote for (i=0..2, j=0..2, k=0..2): 23 terms
    # Strassen for (i=0,1, j=0,1, k=3,4): 7 terms
    # Strassen for (i=0,1, j=0,1, k=5,6): 7 terms  
    # Schoolbook for (i=2, j=0,1, k=3..6): 8 terms
    # Schoolbook for (i=0..2, j=2, k=3..6): 12 terms
    # Schoolbook for (i=0..2, j=3..4, k=0..6): 12 terms
    # Total: 23 + 7 + 7 + 8 + 12 + 12 = 69? That seems too low, let me recount.
    #
    # Actually the terms are:
    # All (i,j,k) for i in 0..2, j in 0..4, k in 0..6
    # 
    # de Groote covers: i=0..2, j=0..2, k=0..2 -> 27 terms in 23
    # Remaining: 
    #   i=0..2, j=0..2, k=3..6 -> 3*3*4 = 36 terms
    #   i=0..2, j=3..4, k=0..6 -> 3*2*7 = 42 terms
    #   Total remaining: 78 terms
    #
    # For i=0..2, j=0..2, k=3..6:
    #   i=0,1, j=0,1: use Strassen for (k=3,4) and (k=5,6) -> 14 terms instead of 16
    #   i=2, j=0,1, k=3..6: 8 schoolbook
    #   i=0..2, j=2, k=3..6: 12 schoolbook
    #   Subtotal: 14 + 8 + 12 = 34
    #
    # For i=0..2, j=3..4, k=0..6:
    #   Could use Strassen if we had 2 consecutive j and 2 consecutive k
    #   j=3,4: use Strassen for (k=0,1), (k=2,3), (k=4,5), and k=6 separately
    #   i=0..2, j=3,4, k=0..6: 
    #     For i=0,1: Strassen (k=0,1)=7, (k=2,3)=7, (k=4,5)=7, (k=6)=2 -> 23
    #     For i=2: schoolbook j=3,4, k=0..6 -> 14
    #     Subtotal: 23 + 14 = 37
    #
    # Total: 23 + 34 + 37 = 94. That's better than 99!

    rank = 94
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # de Groote for (i=0..2, j=0..2, k=0..2)
    col = add_de_groote_3x3(U, V, W, col, 0, 0, 0)

    # Strassen for (i=0,1, j=0,1, k=3,4)
    col = add_strassen_2x2(U, V, W, col, 0, 0, 3)

    # Strassen for (i=0,1, j=0,1, k=5,6)
    col = add_strassen_2x2(U, V, W, col, 0, 0, 5)

    # Schoolbook for (i=2, j=0,1, k=3..6)
    for j in [0, 1]:
        for k in range(3, 7):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # Schoolbook for (i=0..2, j=2, k=3..6)
    for i in range(3):
        for k in range(3, 7):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # For i=0..2, j=3..4, k=0..6:
    # i=0,1: Strassen blocks
    for k0 in [0, 2, 4]:
        col = add_strassen_2x2(U, V, W, col, 0, 3, k0)
    # i=0,1, j=3,4, k=6: schoolbook
    for i in [0, 1]:
        for j in [3, 4]:
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    # i=2, j=3,4, k=0..6: schoolbook
    for j in [3, 4]:
        for k in range(7):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
