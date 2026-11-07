sol = solve_ivp(
    f,
    t_span,
    y0_arr,
    method='DOP853',
    rtol=1e-11,
    atol=1e-13,
    first_step=0.5,
    max_step=np.inf
)
