<<<<<<< SEARCH
def _snap_to_rational(U, V, W, max_denom=100):
    """Try to snap floating point values to rationals with small denominators."""
    U_r = np.zeros_like(U, dtype=object)
    V_r = np.zeros_like(V, dtype=object)
    W_r = np.zeros_like(W, dtype=object)
    
    def snap(val):
        for denom in range(1, max_denom+1):
            num = round(val * denom)
            if abs(val - num/denom) < 1e-10:
                return Fraction(num, denom)
        return Fraction(val)
    
    for i in range(len(U)):
        for j in range(len(U[0])):
            U_r[i, j] = snap(U[i, j])
    for i in range(len(V)):
        for j in range(len(V[0])):
            V_r[i, j] = snap(V[i, j])
    for i in range(len(W)):
        for j in range(len(W[0])):
            W_r[i, j] = snap(W[i, j])
    
    return U_r, V_r, W_r
=======
def _snap_to_half_integer(U, V, W):
    """Snap floating point values to half-integers (multiples of 0.5)."""
    def snap(val):
        # Try to snap to nearest half-integer
        nearest = round(val * 2) / 2.0
        if abs(val - nearest) < 1e-8:
            return nearest
        # If not close to a half-integer, try integer
        nearest_int = round(val)
        if abs(val - nearest_int) < 1e-8:
            return float(nearest_int)
        return val
    
    U_snapped = np.array([[snap(val) for val in row] for row in U], dtype=float)
    V_snapped = np.array([[snap(val) for val in row] for row in V], dtype=float)
    W_snapped = np.array([[snap(val) for val in row] for row in W], dtype=float)
    
    return U_snapped, V_snapped, W_snapped
>>>>>>> REPLACE

