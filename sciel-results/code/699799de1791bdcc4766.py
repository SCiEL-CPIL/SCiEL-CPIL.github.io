"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 13
    U = []
    V = []
    W = []

    def add(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # --- 2x2 Strassen block for a pair of columns (2c, 2c+1) ---
    # A = [[a,b],[c,d]], Bcol = [[b0,b1],[c0,c1]]
    # product = [[a*b0 + b*c0, a*b1 + b*c1],[c*b0 + d*c0, c*b1 + d*c1]]
    for c in range(0, 12, 2):
        b0, b1 = 2 * c, 2 * c + 1
        add([1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1])          # M1=(a+b)(c0+b0)
        add([0, 0, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1])           # M2=(c+d)(b0)
        add([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1])           # M3=a(b1-b0)
        add([0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0])           # M4=d(c1-b1)
        add([1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0])           # M5=(a+c)(b0+c0)
        add([0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0])           # M6=(c-d)(b1+c1)
        add([1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0])           # M7=(a-b)(c0+d)

    # --- tail: single column 12 ---
    # A = [[a,b],[c,d]], B = [[b],[c]] => [[a*b + b*c],[c*b + d*c]]
    add([1, 0], [1, 0], [1, 0])                                  # a*b
    add([0, 1], [1, 1], [0, 1])                                  # b*c + d*c

    # W index is k*n + i, with n=2: column k -> [k*2+0, k*2+1]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
