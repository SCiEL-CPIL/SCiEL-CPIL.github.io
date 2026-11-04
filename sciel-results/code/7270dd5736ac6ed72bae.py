<<<<<<< SEARCH
    # M3 = A00*B00
    U[0][2] = 1
    V[0][2] = 1
    W[0][2] = 1; W[3][2] = -1  # c00 gets +M3, c11 gets -M3
=======
    # M3 = A00*B00
    U[0][2] = 1
    V[0][2] = 1
    W[0][2] = 1; W[3][2] = -1; W[2][2] = 1  # c00 gets +M3, c11 gets -M3, c01 gets +M3
>>>>>>> REPLACE

Now let me also fix the rank to 19:

