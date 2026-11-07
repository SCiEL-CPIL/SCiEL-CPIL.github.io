"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_basis(z: np.ndarray) -> np.ndarray:
    """Return T_0..T_5 evaluated at z (each shape matches z)."""
    T = np.empty((6,) + z.shape, dtype=z.dtype)
    T[0] = 1.0
    T[1] = z
    T[2] = 2.0 * z * z - 1.0
    T[3] = 4.0 * z * z * z - 3.0 * z
    z2 = z * z
    T[4] = 8.0 * z2 * z2 - 8.0 * z2 + 1.0
    T[5] = 16.0 * z * z2 * z2 - 20.0 * z2 * z + 5.0 * z
    return T


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    2D Chebyshev-like basis with 10 parameters.
    Basis (10 terms):
      T0, Tx, Ty, Tx*Ty, T2x, T2y, T3x, T3y, T4x*T0 + T0*T4y (combined as separate terms below),
      Actually we pick 10 specific tensor products:
        p0 * T0x*T0y
        p1 * T1x*T0y
        p2 * T0x*T1y
        p3 * T1x*T1y
        p4 * T2x*T0y
        p5 * T0x*T2y
        p6 * T3x*T0y
        p7 * T0x*T3y
        p8 * T2x*T2y
        p9 * T1x*T2y + T2x*T1y  -> but we only have one param; instead use T1x*T3y? 
    Let's be explicit and pick 10 distinct tensor products to span a useful subspace:
      (0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (3,0), (0,3), (2,1), (1,2)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 100.0  # safety margin before 120s limit

    Tx = _chebyshev_basis(x)
    Ty = _chebyshev_basis(y)

    # Build the 10 basis functions as arrays
    b0 = Tx[0] * Ty[0]
    b1 = Tx[1] * Ty[0]
    b2 = Tx[0] * Ty[1]
    b3 = Tx[1] * Ty[1]
    b4 = Tx[2] * Ty[0]
    b5 = Tx[0] * Ty[2]
    b6 = Tx[3] * Ty[0]
    b7 = Tx[0] * Ty[3]
    b8 = Tx[2] * Ty[1]
    b9 = Tx[1] * Ty[2]

    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)

    if time.time() > deadline:
        # This should not happen normally since the computation is O(n),
        # but if it does, we still return what we have.
        pass

    return pred
# EVOLVE-BLOCK-END
