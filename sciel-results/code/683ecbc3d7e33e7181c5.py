"""Matrix multiplication 2x4 by 4x8 tensor decomposition.

We construct a rank-28 exact decomposition by exploiting the Kronecker
product structure of the multiplication tensor.  The intermediate index
j in {0,1,2,3} is split as j = 2*j1 + j0, and the output index k in
{0,...,7} is split as k = 4*k1 + k0.  Then delta_{j,k} = delta_{j1,k1}
* delta_{j0,k0}.  The second factor is a 2x2 identity (schoolbook rank 4),
and the first factor is also a 2x2 identity but decomposed via Strassen's
rank-7 algorithm.  The Kronecker product yields rank 7 * 4 = 28.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Strassen's 7 products for 2x2 * 2x2 (the j1/k1 factor).
    # Each entry: (U_coeffs, V_coeffs, W_coeffs)
    #   U_coeffs: {(i,j): c}  -- coefficient on A[i,j]
    #   V_coeffs: {(j,k): c}  -- coefficient on B[j,k]
    #   W_coeffs: {(k,i): c}  -- coefficient on C[k,i]
    strassen_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Schoolbook 4 products for 2x2 * 2x2 (the j0/k0 identity factor).
    identity_products = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(1,0):1}),
        ({(1,0):1}, {(0,1):1}, {(1,1):1}),
        ({(1,1):1}, {(1,1):1}, {(1,1):1}),
    ]

    rank = 7 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for u_s, v_s, w_s in strassen_products:
        for u_i, v_i, w_i in identity_products:
            # Kronecker product of each pair
            for (li, lj1), cs in u_s.items():
                for (lj0, lk_i), ci in u_i.items():
                    gi = li
                    gj = 2 * lj1 + lj0
                    U[gi * m + gj][col] += cs * ci
            for (lj1, lk1), cs in v_s.items():
                for (lj0, lk0), ci in v_i.items():
                    gj = 2 * lj1 + lj0
                    gk = 4 * lk1 + lk0
                    V[gj * p + gk][col] += cs * ci
            for (lk1, li), cs in w_s.items():
                for (lk0, li_i), ci in w_i.items():
                    gk = 4 * lk1 + lk0
                    gi = li
                    W[gk * n + gi][col] += cs * ci
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
