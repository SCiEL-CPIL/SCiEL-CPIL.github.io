def _reflectance_spectrum(thicknesses, materials, n_index, s_index, band):
    w0, w1 = band
    lambdas = np.linspace(w0, w1, 401)
    n_arr = np.array([n_index[m] for m in materials], dtype=float)
    d_arr = np.array(thicknesses, dtype=float)
    m11 = np.ones(len(lambdas), dtype=complex)
    m12 = np.zeros(len(lambdas), dtype=complex)
    m21 = np.zeros(len(lambdas), dtype=complex)
    m22 = np.ones(len(lambdas), dtype=complex)
    for i in range(len(n_arr)):
        delta = 2.0 * np.pi * n_arr[i] * d_arr[i] / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        eta = n_arr[i]
        L11 = c
        L12 = 1j * s / eta
        L21 = 1j * eta * s
        L22 = c
        nm11 = m11 * L11 + m12 * L21
        nm12 = m11 * L12 + m12 * L22
        nm21 = m21 * L11 + m22 * L21
        nm22 = m21 * L12 + m22 * L22
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    Y = (m11 + m12 * s_index) / (m21 + m22 * s_index)
    air = n_index['air'] if 'air' in n_index else 1.0
    r = (air - Y) / (air + Y)
    R = (r * r.conjugate()).real
    return R
