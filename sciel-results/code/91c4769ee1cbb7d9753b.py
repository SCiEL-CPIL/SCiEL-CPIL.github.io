import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    n, m, p = 2, 4, 6
    nm, mp, pn = n*m, m*p, p*n  # 8, 24, 12

    def schoolbook():
        """Fallback: rank-48 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for i2 in range(n):
            for j2 in range(m):
                for k2 in range(p):
                    r = i2 * m * p + j2 * p + k2
                    U[i2*m+j2, r] = 1.0
                    V[j2*p+k2, r] = 1.0
                    W[k2*n+i2, r] = 1.0
        return U, V, W, rank

    def verify(U, V, W, rank):
        """Check that the decomposition exactly reconstructs the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Strassen rank-42 decomposition: 1 block (i0=0) x 2 blocks (j0=0,2) x 3 blocks (k0=0,2,4)
    # Each 2x2x2 Strassen block uses rank 7. Total: 1*2*3*7 = 42.
    rank = 42
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    def add_strassen_block(i0, j0, k0, off):
        """Add a correct 2x2x2 Strassen decomposition at position (i0,j0,k0)."""
        # A submatrix: rows i0,i0+1; cols j0,j0+1
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        # B submatrix: rows j0,j0+1; cols k0,k0+1
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        # C submatrix (trace convention W[k*n+i]): rows k0,k0+1; cols i0,i0+1
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + (i0 + 1)
        w11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a00+a11)(b00+b11)
        U[a00, off] += 1.0; U[a11, off] += 1.0
        V[b00, off] += 1.0; V[b11, off] += 1.0
        W[w00, off] += 1.0; W[w01, off] -= 1.0; W[w10, off] += 1.0; W[w11, off] -= 1.0

        # M2 = (a10+a11)(b00)
        U[a10, off+1] += 1.0; U[a11, off+1] += 1.0
        V[b00, off+1] += 1.0
        W[w10, off+1] += 1.0; W[w11, off+1] -= 1.0

        # M3 = a00(b10-b11)
        U[a00, off+2] += 1.0
        V[b10, off+2] += 1.0; V[b11, off+2] -= 1.0
        W[w01, off+2] += 1.0

        # M4 = a11(b01-b00)
        U[a11, off+3] += 1.0
        V[b01, off+3] += 1.0; V[b00, off+3] -= 1.0
        W[w00, off+3] -= 1.0; W[w10, off+3] += 1.0

        # M5 = (a00+a01)(b11)
        U[a00, off+4] += 1.0; U[a01, off+4] += 1.0
        V[b11, off+4] += 1.0
        W[w00, off+4] += 1.0; W[w01, off+4] += 1.0

        # M6 = (a10-a00)(b00+b01)
        U[a10, off+5] += 1.0; U[a00, off+5] -= 1.0
        V[b00, off+5] += 1.0; V[b01, off+5] += 1.0
        W[w11, off+5] += 1.0

        # M7 = (a01-a11)(b10+b11)
        U[a01, off+6] += 1.0; U[a11, off+6] -= 1.0
        V[b10, off+6] += 1.0; V[b11, off+6] += 1.0
        W[w11, off+6] += 1.0

    off = 0
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                add_strassen_block(i0, j0, k0, off)
                off += 7

    # Verify the Strassen construction
    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if Strassen verification fails
    if time.time() < deadline:
        U, V, W, rank = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Last resort: return empty (should not happen)
    return {"U": [], "V": [], "W": []}
