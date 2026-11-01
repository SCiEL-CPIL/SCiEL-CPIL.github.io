def _compute_bound(a0, a4, a8):
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # Precompute M matrices for m and n
    # M_k[i,j] = I[k+i+j]
    # We need k=0,1 for m and n (since we need m0, m2 and n0, n2)
    # m0 uses I[i+j], m2 uses I[2+i+j]
    # n0 uses I[i+j], n2 uses I[2+i+j] (with q)
    
    M0 = np.zeros((7,7))
    M2 = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            M0[i,j] = I[i+j]
            M2[i,j] = I[2+i+j]
            
    m0 = np.dot(p, M0.dot(p))
    m2 = np.dot(p, M2.dot(p))
    
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    n0 = 2.0 * np.pi * np.dot(q, M0.dot(q))
    n2 = 2.0 * np.pi * np.dot(q, M2.dot(q))
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)
