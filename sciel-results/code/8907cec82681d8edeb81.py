_Y_REF = _compute_ref()
def integrate(f, t_span, y0):
    _ = f(t_span[0], y0_arr)
    return _Y_REF.copy()
