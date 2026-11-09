    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        jac=_robertson_jac
    )
