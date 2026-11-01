import numpy as np
import time

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    def build_rank32():
        rank = 32
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0

        # Build decomposition using Strassen for 2x2 blocks and Schoolbook for the remainder
        # We have 4 blocks of 2 columns (0-1, 2-3, 4-5, 6-7) and 1 single column (8)
        
        # Strassen formulas for 2x2 block
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A01)(B01+B11)
        # M4 = (A10+A11)B10
        # M5 = (A00+A01)B00
        # M6 = (A10-A00)(B01+B11)  -- Wait, standard Strassen M6 is (A10-A00)(B01+B11)? 
        # Let's check the seed's M6: (c-a)(x+z) where c=A10, a=A00, x=B00, z=B10.
        # Seed M6: (A10-A00)(B00+B10). 
        # Standard Strassen:
        # m1 = (a+d)(x+w)
        # m2 = (c+d)x
        # m3 = (a-b)(y+w)
        # m4 = (c+d)z
        # m5 = (a+b)x
        # m6 = (c-a)(x+z)  -> (A10-A00)(B00+B10)
        # m7 = (b-d)(y+w)  -> (A01-A11)(B01+B11)
        
        # Outputs:
        # C00 = m1 - m4 + m6 + m7
        # C01 = m3 + m4
        # C10 = m2 + m5
        # C11 = m1 - m3 + m5 - m2

        for pair_idx in range(4):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1

            # M1: U:{a:1,d:1}, V:{B[0,k1]:1, B[1,k2]:1}
            U[0, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1; V[1*p+k2, col] = 1
            # W: C00 (+1), C11 (+1)
            W[k1*n+0, col] = 1
            W[k2*n+1, col] = 1
            col += 1

            # M2: U:{c:1,d:1}, V:{B[0,k1]:1}
            U[2, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1
            # W: C10 (+1), C11 (-1)
            W[k1*n+1, col] = 1
            W[k2*n+1, col] = -1
            col += 1

            # M3: U:{a:1,b:-1}, V:{B[0,k2]:1, B[1,k2]:1}
            U[0, col] = 1; U[1, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            # W: C01 (+1), C11 (-1)
            W[k2*n+0, col] = 1
            W[k2*n+1, col] = -1
            col += 1

            # M4: U:{c:1,d:1}, V:{B[1,k1]:1}
            U[2, col] = 1; U[3, col] = 1
            V[1*p+k1, col] = 1
            # W: C00 (-1), C01 (+1)
            W[k1*n+0, col] = -1
            W[k2*n+0, col] = 1
            col += 1

            # M5: U:{a:1,b:1}, V:{B[0,k1]:1}
            U[0, col] = 1; U[1, col] = 1
            V[0*p+k1, col] = 1
            # W: C10 (+1), C11 (+1)
            W[k1*n+1, col] = 1
            W[k2*n+1, col] = 1
            col += 1

            # M6: U:{c:1,a:-1}, V:{B[0,k1]:1, B[1,k1]:1}
            U[2, col] = 1; U[0, col] = -1
            V[0*p+k1, col] = 1; V[1*p+k1, col] = 1
            # W: C00 (+1)
            W[k1*n+0, col] = 1
            col += 1

            # M7: U:{b:1,d:-1}, V:{B[0,k2]:1, B[1,k2]:1}
            U[1, col] = 1; U[3, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            # W: C00 (+1)
            W[k1*n+0, col] = 1
            col += 1

        # Last single column k=8: rank 4
        k = 8
        # A00 * B08 -> C00
        U[0, col] = 1; V[0*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        # A01 * B18 -> C00
        U[1, col] = 1; V[1*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        # A10 * B08 -> C10
        U[2, col] = 1; V[0*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
        # A11 * B18 -> C10
        U[3, col] = 1; V[1*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

        assert col == rank, f"Expected {rank}, got {col}"

        return U.tolist(), V.tolist(), W.tolist()

    result = build_rank32()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
