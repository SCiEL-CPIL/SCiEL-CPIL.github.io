<<<<<<< SEARCH
def _try_polish(U, V, W, n, m, p, max_denom=20):
    """Try to snap to small rationals and verify."""
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
=======
def _try_polish(U, V, W, n, m, p, max_denom=20):
    """Try to snap to small rationals and verify."""
    if isinstance(U, np.ndarray):
        U_list = U.tolist()
    else:
        U_list = U
    if isinstance(V, np.ndarray):
        V_list = V.tolist()
    else:
        V_list = V
    if isinstance(W, np.ndarray):
        W_list = W.tolist()
    else:
        W_list = W
>>>>>>> REPLACE
