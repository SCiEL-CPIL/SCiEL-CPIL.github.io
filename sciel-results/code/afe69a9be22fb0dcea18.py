def _optimize_coordinate(h, i):
    h_zero = h.copy()
    old_val = h_zero[i]
    h_zero[i] = 0.0
    
    c_base = np.convolve(h_zero, h_zero)
    S_fixed = np.sum(h_zero)
    
    # Precompute M1 coefficients
    # k in [i, i+N-1]
    ks = np.arange(i, i+N)
    # Filter k != 2i
    mask = ks != 2*i
    valid_ks = ks[mask]
    
    if len(valid_ks) > 0:
        m_coeffs = 2 * h_zero[valid_ks - i]
        b_coeffs = c_base[valid_ks]
    else:
        m_coeffs = np.array([])
        b_coeffs = np.array([])
        
    # Precompute M2
    if 0 <= 2*i < 2*N - 1:
        m2_base = c_base[2*i]
    else:
        m2_base = -1e18
        
    # Precompute M3
    m3_val = -1e18
    if i > 0:
        m3_val = max(m3_val, np.max(c_base[:i]))
    if i + N < 2*N - 1:
        m3_val = max(m3_val, np.max(c_base[i+N:]))
        
    def eval_v_fast(x):
        s = S_fixed + x
        if s == 0:
            return 1e12
            
        # M1
        if len(m_coeffs) > 0:
            m1 = np.max(m_coeffs * x + b_coeffs)
        else:
            m1 = -1e18
            
        # M2
        m2 = m2_base + x*x
        
        m_total = max(m1, m2, m3_val)
        
        val = abs(2 * N * m_total / (s * s))
        return val

    # Search logic
    best_x = old_val
    best_v = eval_v_fast(old_val)
    
    center = old_val
    radius = max(0.1, 0.1 * abs(center))
    
    for _ in range(10):
        x_min = center - radius
        x_max = center + radius
        
        v_min = eval_v_fast(x_min)
        v_max = eval_v_fast(x_max)
        
        if v_min < best_v or v_max < best_v:
             radius *= 2
             continue
             
        xs = np.linspace(x_min, x_max, 50)
        vs = np.array([eval_v_fast(x) for x in xs])
        idx_min = np.argmin(vs)
        
        if vs[idx_min] < best_v:
            best_v = vs[idx_min]
            best_x = xs[idx_min]
            
        center = best_x
        radius = (x_max - x_min) / 10.0
        
        if radius < 1e-6:
            break

    return best_x
