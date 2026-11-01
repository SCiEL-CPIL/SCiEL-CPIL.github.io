import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    
    # Block 1: k=0,1
    # M1: (a+d)(e+h)
    add_term(0, 1, 0, 1, 0, 1, col) # a
    add_term(3, 1, 0, 1, 0, 1, col) # d
    add_term(0, 1, 6, 1, 0, 1, col) # e part? No, U and V are separate.
    # Wait, my add_term logic above was wrong. U, V, W entries are summed per column.
    # For M1:
    # U: A00 + A11 -> U[0]+=1, U[3]+=1
    # V: B00 + B11 -> V[0]+=1, V[6]+=1
    # W: C00 + C11 -> W[0]+=1, W[3]+=1
    U[0][col] += 1; U[3][col] += 1
    V[0][col] += 1; V[6][col] += 1
    W[0][col] += 1; W[3][col] += 1
    col += 1

    # M2: (c+d)e
    U[2][col] += 1; U[3][col] += 1
    V[0][col] += 1
    W[1][col] += 1; W[3][col] -= 1
    col += 1

    # M3: a(f-h)
    U[0][col] += 1
    V[1][col] += 1; V[6][col] -= 1
    W[2][col] += 1; W[3][col] += 1
    col += 1

    # M4: d(g-e)
    U[3][col] += 1
    V[5][col] += 1; V[0][col] -= 1
    W[0][col] += 1; W[1][col] += 1
    col += 1

    # M5: (a+b)h
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1
    W[0][col] -= 1; W[2][col] += 1
    col += 1

    # M6: (c-a)(e+f)
    U[2][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[3][col] += 1
    col += 1

    # M7: (b-d)(g+h)
    U[1][col] += 1; U[3][col] -= 1
    V[5][col] += 1; V[6][col] += 1
    W[0][col] += 1
    col += 1

    # Block 2: k=2,3
    # M1: (a+d)(e+h)
    # V indices: B02->2, B13->8. Wait, B13 is j=1, k=3. V[1*5+3]=V[8].
    # W indices: C02->4, C13->7.
    U[0][col] += 1; U[3][col] += 1
    V[2][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[7][col] += 1
    col += 1

    # M2: (c+d)e (e is B02)
    U[2][col] += 1; U[3][col] += 1
    V[2][col] += 1
    W[5][col] += 1; W[7][col] -= 1
    col += 1

    # M3: a(f-h) (f is B03, h is B13)
    U[0][col] += 1
    V[3][col] += 1; V[8][col] -= 1
    W[6][col] += 1; W[7][col] += 1
    col += 1

    # M4: d(g-e) (g is B12, e is B02)
    U[3][col] += 1
    V[7][col] += 1; V[2][col] -= 1
    W[4][col] += 1; W[5][col] += 1
    col += 1

    # M5: (a+b)h (h is B13)
    U[0][col] += 1; U[1][col] += 1
    V[8][col] += 1
    W[4][col] -= 1; W[6][col] += 1
    col += 1

    # M6: (c-a)(e+f) (e is B02, f is B03)
    U[2][col] += 1; U[0][col] -= 1
    V[2][col] += 1; V[3][col] += 1
    W[7][col] += 1
    col += 1

    # M7: (b-d)(g+h) (g is B12, h is B13)
    U[1][col] += 1; U[3][col] -= 1
    V[7][col] += 1; V[8][col] += 1
    W[4][col] += 1
    col += 1

    # Block 3: k=4 (Schoolbook)
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # Term 1: A00 B04
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1
    col += 1

    # Term 2: A01 B14
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1
    col += 1

    # Term 3: A10 B04
    U[2][col] += 1
    V[4][col] += 1
    W[9][col] += 1
    col += 1

    # Term 4: A11 B14
    U[3][col] += 1
    V[9][col] += 1
    W[9][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
