def _round_to_half(x):
    return np.round(x * 2) / 2

# In _search_rank, after finding a candidate:
U_rounded = _round_to_half(U)
V_rounded = _round_to_half(V)
W_rounded = _round_to_half(W)
if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
    return U_rounded, V_rounded, W_rounded
# Also check the unrounded version if error is tiny
if np.allclose(np.einsum('ar,br,cr->abc', U, V, W), target, atol=1e-9):
    return U, V, W
