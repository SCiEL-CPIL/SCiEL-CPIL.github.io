"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _try_rank_1(n, m, p):
    """Try rank-1 decomposition: all entries = 1."""
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W
    return None

def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists with integer/half-integer entries."""
    def convert(mat):
        rows = []
        for row in mat:
            r = []
            for val in row:
                val_f = float(val)
                nearest_int = round(val_f)
                nearest_half = round(val_f * 2) / 2.0
                if abs(val_f - nearest_int) < 1e-10:
                    r.append(int(nearest_int))
                elif abs(val_f - nearest_half) < 1e-10:
                    r.append(nearest_half)
                else:
                    r.append(val_f)
            rows.append(r)
        return rows
    return {"U": convert(U), "V": convert(V), "W": convert(W)}

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try rank 1 first
    result = _try_rank_1(n, m, p)
    if result is not None:
        U, V, W = result
        return _to_lists(U, V, W)
    
    # Fall back to schoolbook
    U, V, W = _schoolbook()
    return _to_lists(U, V, W)
# EVOLVE-BLOCK-END
