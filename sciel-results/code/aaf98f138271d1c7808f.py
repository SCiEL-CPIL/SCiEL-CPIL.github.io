"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    U_list = []
    V_list = []
    W_list = []

    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # Part 1: First 2 columns of B (k=0,1)
    # A is 2x3, B1 is 3x2.
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A*B1 is 2x2
    # C1[0,0] = a00*b00 + a01*b10 + a02*b20
    # C1[0,1] = a00*b01 + a01*b11 + a02*b21
    # C1[1,0] = a10*b00 + a11*b10 + a12*b20
    # C1[1,1] = a10*b01 + a11*b11 + a12*b21

    # Split into A1*B11 (2x2 * 2x2) + A2*B12 (2x1 * 1x2)
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    # A2 = [[a02], [a12]]
    # B12 = [[b20, b21]]

    # Strassen for A1*B11 (7 terms):
    # M1 = (a00+a11)*(b00+b11)
    u = [0]*6; v = [0]*30; w = [0]*20
    u[0] = 1; u[4] = 1  # a00 (i=0,j=0), a11 (i=1,j=1)
    v[0] = 1; v[11] = 1  # b00 (j=0,k=0), b11 (j=1,k=1)
    w[0] = 1; w[3] = 1  # C[0,0] (k=0,i=0), C[1,1] (k=1,i=1)
    add_term(u, v, w)

    # M2 = (a10+a11)*b00
    u = [0]*6; v = [0]*30; w = [0]*20
    u[3] = 1; u[4] = 1  # a10 (i=1,j=0), a11 (i=1,j=1)
    v[0] = 1  # b00 (j=0,k=0)
    w[2] = 1  # C[0,1] (k=0,i=1)
    add_term(u, v, w)

    # M3 = a00*(b01-b11)
    u = [0]*6; v = [0]*30; w = [0]*20
    u[0] = 1  # a00
    v[1] = 1; v[11] = -1  # b01 (j=0,k=1), b11 (j=1,k=1)
    w[1] = 1  # C[1,0] (k=1,i=0)
    add_term(u, v, w)

    # M4 = a11*(b10-b00)
    u = [0]*6; v = [0]*30; w = [0]*20
    u[4] = 1  # a11
    v[10] = 1; v[0] = -1  # b10 (j=1,k=0), b00 (j=0,k=0)
    w[2] = 1  # C[0,1] (k=0,i=1)
    add_term(u, v, w)

    # M5 = (a00+a01)*b11
    u = [0]*6; v = [0]*30; w = [0]*20
    u[0] = 1; u[1] = 1  # a00, a01
    v[11] = 1  # b11 (j=1,k=1)
    w[0] = -1; w[1] = 1  # -C[0,0], +C[1,0]
    add_term(u, v, w)

    # M6 = (a10-a00)*(b00+b01)
    u = [0]*6; v = [0]*30; w = [0]*20
    u[3] = 1; u[0] = -1  # a10, -a00
    v[0] = 1; v[1] = 1  # b00, b01
    w[3] = 1  # C[1,1] (k=1,i=1)
    add_term(u, v, w)

    # M7 = (a01-a11)*(b10+b11)
    u = [0]*6; v = [0]*30; w = [0]*20
    u[1] = 1; u[4] = -1  # a01, -a11
    v[10] = 1; v[11] = 1  # b10, b11
    w[0] = 1  # C[0,0] (k=0,i=0)
    add_term(u, v, w)

    # A2*B12 (4 schoolbook terms):
    # a02*b20 -> C[0,0]
    u = [0]*6; v = [0]*30; w = [0]*20
    u[2] = 1  # a02 (i=0,j=2)
    v[20] = 1  # b20 (j=2,k=0)
    w[0] = 1  # C[0,0]
    add_term(u, v, w)

    # a02*b21 -> C[1,0]
    u = [0]*6; v = [0]*30; w = [0]*20
    u[2] = 1  # a02
    v[21] = 1  # b21 (j=2,k=1)
    w[1] = 1  # C[1,0]
    add_term(u, v, w)

    # a12*b20 -> C[0,1]
    u = [0]*6; v = [0]*30; w = [0]*20
    u[5] = 1  # a12 (i=1,j=2)
    v[20] = 1  # b20
    w[2] = 1  # C[0,1]
    add_term(u, v, w)

    # a12*b21 -> C[1,1]
    u = [0]*6; v = [0]*30; w = [0]*20
    u[5] = 1  # a12
    v[21] = 1  # b21
    w[3] = 1  # C[1,1]
    add_term(u, v, w)

    # Part 2: Remaining 8 columns (k=2,...,9)
    # Schoolbook for A (2x3) * B2 (3x8)
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                u = [0]*6; v = [0]*30; w = [0]*20
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                add_term(u, v, w)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
