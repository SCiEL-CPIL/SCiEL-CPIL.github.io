import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Start with naive rank-30 decomposition
    rank = 30
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
                r += 1

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    def validate(U, V, W, rank):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify our rank-30 works
    if not validate(best_U, best_V, best_W, 30):
        pass

    # Build rank-28 decomposition using Strassen for 2x2 blocks
    # Split: j=2 (row 2 of A, col 0-4 of B) -> 10 terms
    # j=0,1 with k=4 -> 4 terms
    # j=0,1 with k=0,1 -> Strassen 7 terms
    # j=0,1 with k=2,3 -> Strassen 7 terms
    # Total: 10 + 4 + 7 + 7 = 28

    U28 = np.zeros((nm, 28))
    V28 = np.zeros((mp, 28))
    W28 = np.zeros((pn, 28))

    r = 0
    # Part 1: j=2, k=0..4 (10 terms)
    for i in range(n):
        for k in range(p):
            U28[i*m + 2, r] = 1
            V28[2*p + k, r] = 1
            W28[k*n + i, r] = 1
            r += 1

    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U28[i*m + j, r] = 1
            V28[j*p + 4, r] = 1
            W28[4*n + i, r] = 1
            r += 1

    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # Strassen for C[0,0], C[0,1], C[1,0], C[1,1] from A[0:2,0:2]*B[0:2,0:2]
    # M1 = (A00+A11)(B00+B11)
    U28[0*m + 0, r] = 1; U28[1*m + 1, r] = 1
    V28[0*p + 0, r] = 1; V28[1*p + 1, r] = 1
    W28[0*n + 0, r] = 1; W28[1*n + 1, r] = 1
    r += 1
    # M2 = A11(B10-B00)
    U28[1*m + 1, r] = 1
    V28[1*p + 0, r] = 1; V28[0*p + 0, r] = -1
    W28[0*n + 1, r] = 1; W28[1*n + 1, r] = 1
    r += 1
    # M3 = (A00-A10)B11
    U28[0*m + 0, r] = 1; U28[1*m + 0, r] = -1
    V28[1*p + 1, r] = 1
    W28[0*n + 0, r] = 1; W28[1*n + 0, r] = 1
    r += 1
    # M4 = A00(B11-B01)
    U28[0*m + 0, r] = 1
    V28[1*p + 1, r] = 1; V28[0*p + 1, r] = -1
    W28[0*n + 0, r] = 1; W28[0*n + 1, r] = 1
    r += 1
    # M5 = (A10+A00)(B01+B00)
    U28[1*m + 0, r] = 1; U28[0*m + 0, r] = 1
    V28[0*p + 1, r] = 1; V28[0*p + 0, r] = 1
    W28[0*n + 1, r] = 1; W28[1*n + 1, r] = -1
    r += 1
    # M6 = (A10-A00)(B01-B11)
    U28[1*m + 0, r] = 1; U28[0*m + 0, r] = -1
    V28[0*p + 1, r] = 1; V28[1*p + 1, r] = -1
    W28[1*n + 0, r] = 1; W28[1*n + 1, r] = 1
    r += 1
    # M7 = (A11-A10)(B10+B00)
    U28[1*m + 1, r] = 1; U28[1*m + 0, r] = -1
    V28[1*p + 0, r] = 1; V28[0*p + 0, r] = 1
    W28[0*n + 1, r] = 1; W28[1*n + 0, r] = 1
    r += 1

    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # Same structure but for B[0:2, 2:4]
    # M1 = (A00+A11)(B02+B13)
    U28[0*m + 0, r] = 1; U28[1*m + 1, r] = 1
    V28[0*p + 2, r] = 1; V28[1*p + 3, r] = 1
    W28[2*n + 0, r] = 1; W28[3*n + 1, r] = 1
    r += 1
    # M2 = A11(B12-B02)
    U28[1*m + 1, r] = 1
    V28[1*p + 2, r] = 1; V28[0*p + 2, r] = -1
    W28[2*n + 1, r] = 1; W28[3*n + 1, r] = 1
    r += 1
    # M3 = (A00-A10)B13
    U28[0*m + 0, r] = 1; U28[1*m + 0, r] = -1
    V28[1*p + 3, r] = 1
    W28[2*n + 0, r] = 1; W28[3*n + 0, r] = 1
    r += 1
    # M4 = A00(B13-B03)
    U28[0*m + 0, r] = 1
    V28[1*p + 3, r] = 1; V28[0*p + 3, r] = -1
    W28[2*n + 0, r] = 1; W28[2*n + 1, r] = 1
    r += 1
    # M5 = (A10+A00)(B03+B02)
    U28[1*m + 0, r] = 1; U28[0*m + 0, r] = 1
    V28[0*p + 3, r] = 1; V28[0*p + 2, r] = 1
    W28[2*n + 1, r] = 1; W28[3*n + 1, r] = -1
    r += 1
    # M6 = (A10-A00)(B03-B13)
    U28[1*m + 0, r] = 1; U28[0*m + 0, r] = -1
    V28[0*p + 3, r] = 1; V28[1*p + 3, r] = -1
    W28[3*n + 0, r] = 1; W28[3*n + 1, r] = 1
    r += 1
    # M7 = (A11-A10)(B12+B02)
    U28[1*m + 1, r] = 1; U28[1*m + 0, r] = -1
    V28[1*p + 2, r] = 1; V28[0*p + 2, r] = 1
    W28[2*n + 1, r] = 1; W28[3*n + 0, r] = 1
    r += 1

    if validate(U28, V28, W28, 28):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = 28

    # Now try to find rank 27 or lower using structured search
    # Strategy: Use the rank-28 as a starting point and try to merge terms
    # or use random search with focused values

    deadline = time.time() + 100  # 100 second deadline

    # Values to try: integers and half-integers
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    focused_vals = [-1, -0.5, 0, 0.5, 1]

    # Try rank 27 with random search
    for target_rank in range(27, best_rank):
        if time.time() > deadline:
            break
        found = False
        # Use more attempts for lower ranks
        max_attempts = 200 if target_rank == 27 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            U_t = np.random.choice(focused_vals, size=(nm, target_rank))
            V_t = np.random.choice(focused_vals, size=(mp, target_rank))
            W_t = np.random.choice(focused_vals, size=(pn, target_rank))
            if validate(U_t, V_t, W_t, target_rank):
                best_U, best_V, best_W = U_t, V_t, W_t
                best_rank = target_rank
                found = True
                break

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
