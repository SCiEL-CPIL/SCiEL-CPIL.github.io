<<<<<<< SEARCH
def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
    # Precompute up to n=12 (need x^{24})
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        # (2n)! / (4^n * n!) * sqrt(pi)
        # Recurrence: I[n] = I[n-1] * (2n-1) / 2
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)
=======
def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    Uses convolution for faster moment calculation.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx
    # |P(x)|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # Coefficients of |P|^2 in terms of x^{2m} are conv(p, p)
    p_sq_coeffs = np.convolve(p, p)
    
    m = np.zeros(3)
    # m[k] = sum_{m=0}^{12} p_sq_coeffs[m] * I[k+m]
    # We only need m[0], m[1], m[2]
    # m[0] = sum_{m=0}^{12} p_sq_coeffs[m] * I[m]
    # m[1] = sum_{m=0}^{12} p_sq_coeffs[m] * I[m+1]
    # m[2] = sum_{m=0}^{12} p_sq_coeffs[m] * I[m+2]
    # I has length 13, so I[m+2] is valid for m up to 10.
    # p_sq_coeffs has length 13 (indices 0..12).
    for m in range(13):
        coeff = p_sq_coeffs[m]
        if m + 0 < 13: m[0] += coeff * I[m]
        if m + 1 < 13: m[1] += coeff * I[m+1]
        if m + 2 < 13: m[2] += coeff * I[m+2]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    # P(-i*xi) = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q_sq_coeffs = np.convolve(q, q)
    
    n = np.zeros(3)
    for m in range(13):
        coeff = q_sq_coeffs[m]
        if m + 0 < 13: n[0] += coeff * I[m]
        if m + 1 < 13: n[1] += coeff * I[m+1]
        if m + 2 < 13: n[2] += coeff * I[m+2]
    
    n *= 2.0 * np.pi

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)
>>>>>>> REPLACE

