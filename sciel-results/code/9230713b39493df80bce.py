def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Strategy: Return the exact zeros of the generalized Laguerre polynomial L_{12}^{(1)}(x).
    """
    m = 12
    n = 1
    
    # Get initial roots: zeros of L_{12}^{(1)}(x)
    degree = m  # 12
    alpha = 2 * n - 1  # = 1
    
    try:
        L = genlaguerre(degree, alpha)
        roots = L.roots()
        roots = roots[np.real(roots) > 0]
        roots = np.real(roots)
        roots = np.sort(roots)
        
        if len(roots) != m:
            if len(roots) > m:
                roots = roots[:m]
            else:
                roots = _get_fallback_roots(m, n)
    except Exception:
        roots = _get_fallback_roots(m, n)
    
    # Verify roots are positive
    if not np.all(roots > 0):
        roots = np.maximum(roots, 1e-10)
    
    return [float(x) for x in roots]
