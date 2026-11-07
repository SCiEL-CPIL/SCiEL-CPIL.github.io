sol = solve_ivp(
    f,
    t_span,
    y0,
    method='DOP853',
    rtol=1e-13,
    atol=1e-15
)
