    A_power = np.eye(n)
    exp_A_dt = np.zeros((n, n))
    for k in range(n):
        exp_A_dt += coeffs[k] * A_power
        A_power = A_power @ A
