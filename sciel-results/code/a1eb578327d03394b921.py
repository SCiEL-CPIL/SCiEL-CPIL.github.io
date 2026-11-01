import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    The known minimal rank for this tensor is 11 (Strassen 7 + 4 schoolbook terms).
    """
    n, m, p = DIMENSIONS
    rank = 11

    # Build matrices as flat lists of lists (integer entries)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # --- Strassen-style 7 terms for the 2x2 x 2x2 part (columns 0,1 of B) ---
    # Term 0: (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 1: (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 2: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 3: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 4: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 5: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # --- 4 schoolbook terms for column 2 of B ---
    # Term 7: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition computes matrix multiplication exactly."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    # Tensor T[a,b,c,d] = sum_r W[k*n+i][r] * U[a*m+b][r] * V[c*p+d][r]
    # where k is output column, i is output row
    # Expected: T[a,b,c,d] = 1 iff b==c and a==i and d==k
    # i.e., T[a,b,c,d] = 1 iff b==c and a==i and d==k
    # More precisely: output[i,k] = sum_{b,c} A[a,b] * B[c,d] with i=a, k=d
    # So the tensor is T[a,b,c,d] = delta_{b,c} * delta_{a,i} * delta_{d,k}
    # But i and k are not free indices here; they are determined by the output position.
    # The multiplication tensor has indices (i, a, b, c, d, k) but really:
    # output[i,k] = sum_{a,b,c,d} A[a,b] B[c,d] * T[i,a,b,c,d,k]
    # where T[i,a,b,c,d,k] = 1 iff i==a and k==d and b==c, else 0.
    # So T[i,a,b,c,d,k] = delta_{i,a} * delta_{k,d} * delta_{b,c}

    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (b == c and a == i and d == k) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += W[k * n + i][r] * U[a * m + b][r] * V[c * p + d][r]
                            if abs(actual - expected) > 1e-10:
                                return False
    return True
# EVOLVE-BLOCK-END
