import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    # Returns U, V, W for 2x2x2
    # Shapes (4, 7)
    pass

def _build_rank42():
    # Builds U, V, W for 2x4x6
    # Shapes (8, 42), (24, 42), (12, 42)
    pass

def _verify(U, V, W, n, m, p):
    pass

def _to_output(mat):
    pass

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank42()
    if _verify(U, V, W, n, m, p):
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}
    # Fallback to schoolbook just in case (rank 48)
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": _to_output(U_fb), "V": _to_output(V_fb), "W": _to_output(W_fb)}
