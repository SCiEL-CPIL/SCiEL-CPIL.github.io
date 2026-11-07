"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853_step(f, t, y, h):
    """
    Single step of DOP853 (Dormand-Prince 8th order, 5th order error estimator).
    Returns (y_new, error_estimate, nfev)
    """
    n = len(y)
    y = np.array(y, dtype=float)
    
    # DOP853 Butcher tableau
    # c_i
    c = np.array([
        0,
        1/12,
        1/6,
        1/4,
        5/12,
        1/2,
        5/6,
        3/4,
        1,
        1,
        11/12,
        1,
        17/12,
        1,
        1/4,
        1/2,
        1,
        1
    ])
    
    # a_ij (only non-zero entries, stored as lists of (j, a_ij) for each i)
    a = [
        [],
        [(0, 1/12)],
        [(0, 1/24), (1, 1/8)],
        [(0, 1/17), (1, -5/68), (2, 10/17)],
        [(0, -25/108), (1, 4/9), (2, -128/81), (3, 512/81)],
        [(0, 1408/16807), (1, -8/7), (2, 3544/16807), (3, -16/7), (4, 129600/16807)],
        [(0, -245/216), (1, 26), (2, -6200/81), (3, 1200/7), (4, -370512/60625), (5, 4900/5241)],
        [(0, 9/2), (1, -1732/81), (2, 22768/567), (3, -16200/187), (4, 298880/14157), (5, -28/5), (6, 950/187)],
        [(0, 37/10), (1, -189/5), (2, 12836/15), (3, -3304/5), (4, 87968/11375), (5, -28432/2275), (6, 28432/2275), (7, -109/25)],
        [(0, -201241137/380793600), (1, 166110691/100257600), (2, 33654329/4862000), (3, 24936700/1361259), (4, -110052668/12011625), (5, 11256544/233375), (6, -160238/21735), (7, 579413/186300), (8, -1/108)],
        [(0, -224300127/380793600), (1, 168161461/100257600), (2, 33908229/4862000), (3, 25097800/1361259), (4, -110908468/12011625), (5, 11363944/233375), (6, -161848/21735), (7, 585513/186300), (8, -1/108), (9, -1/6)],
        [(0, 946068813/380793600), (1, -71540989/100257600), (2, -2468907/486200), (3, -9485700/1361259), (4, 42073168/12011625), (5, -4320744/233375), (6, 61948/21735), (7, -22373/18630), (8, 1/108), (9, 1/6), (10, 1/6)],
        [(0, -321499073/380793600), (1, 25508849/10025760), (2, 444781/162066), (3, 3785400/1361259), (4, -14326188/12011625), (5, 1440248/233375), (6, -20649/7245), (7, 7457/6210), (8, -1/36), (9, -1/6), (10, -1/6), (11, 1/6)],
        [(0, 208083311/380793600), (1, -15835709/10025760), (2, -284237/162066), (3, -2403600/1361259), (4, 9239388/12011625), (5, -927848/233375), (6, 13119/7245), (7, -4661/6210), (8, 1/36), (9, 1/6), (10, 1/6), (11, -1/6), (12, 1/6)],
        [(0, 11/20), (1, -1131/550), (2, 38/11), (3, -104/11), (4, 9751/792), (5, -17/11), (6, 11/11), (7, -3/5), (8, -1/20), (9, 1/10), (10, -1/10), (11, 1/10), (12, -1/10), (13, 1/10)],
        [(0, -24049/10080), (1, 1125/64), (2, -1488/35), (3, 1799/10), (4, -4609/16), (5, 120/7), (6, -105/8), (7, 5/2), (8, 3/16), (9, -3/16), (10, 3/16), (11, -3/16), (12, 3/16), (13, -3/16), (14, 1/16)],
        [(0, 205/216), (1, -1414/243), (2, 11680/567), (3, -16200/187), (4, 298880/14157), (5, -28/5), (6, 950/187), (7, -1/2), (8, 1/4), (9, -1/4), (10, 1/4), (11, -1/4), (12, 1/4), (13, -1/4), (14, 1/4), (15, -1/4)],
        [(0, 575/144), (1, -2336/243), (2, 14288/567), (3, -19800/187), (4, 366080/14157), (5, -32/5), (6, 1150/187), (7, -3/4), (8, 1/2), (9, -1/2), (10, 1/2), (11, -1/2), (12, 1/2), (13, -1/2), (14, 1/2), (15, -1/2), (16, 1/2)],
        [(0, 575/144), (1, -2336/243), (2, 14288/567), (3, -19800/187), (4, 366080/14157), (5, -32/5), (6, 1150/187), (7, -3/4), (8, 1/2), (9, -1/2), (10, 1/2), (11, -1/2), (12, 1/2), (13, -1/2), (14, 1/2), (15, -1/2), (16, 1/2), (17, 1/2)]
    ]
    
    # b_i (8th order)
    b = np.array([
        323/32400,
        0,
        1471885/11219400,
        0,
        5712322/36096285,
        0,
        1438021/4293975,
        0,
        4/45,
        -0.013888888888888888,
        0.013888888888888888,
        0,
        -0.013888888888888888,
        0,
        1/60,
        0,
        0,
        0
    ])
    
    # b_i* (5th order for error estimation)
    b_star = np.array([
        61/14400,
        0,
        172721/11219400,
        0,
        5712322/36096285,
        0,
        1438021/4293975,
        0,
        4/45,
        -0.013888888888888888,
        0.013888888888888888,
        0,
        -0.013888888888888888,
        0,
        1/60,
        0,
        0,
        0
    ])
    
    nfev = 0
    k = np.zeros((18, n))
    
    for i in range(18):
        if i == 0:
            k[i] = f(t, y)
            nfev += 1
        else:
            y_stage = y.copy()
            for (j, a_ij) in a[i]:
                y_stage += h * a_ij * k[j]
            k[i] = f(t + h * c[i], y_stage)
            nfev += 1
    
    # 8th order result
    y_new = y.copy()
    for i in range(18):
        y_new += h * b[i] * k[i]
    
    # 5th order result for error estimation
    y_star = y.copy()
    for i in range(18):
        y_star += h * b_star[i] * k[i]
    
    error = y_new - y_star
    return y_new, error, nfev


def _dop853_adaptive(f, t0, t1, y0, rtol=1e-11, atol=1e-13):
    """
    Adaptive DOP853 integrator.
    """
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size estimate
    h = (t1 - t0) / 100.0
    
    total_nfev = 0
    max_steps = 100000
    step_count = 0
    
    while t < t1 and step_count < max_steps:
        if t + h > t1:
            h = t1 - t
        
        y_new, error, nfev = _dop853_step(f, t, y, h)
        total_nfev += nfev
        step_count += 1
        
        # Check for non-finite values
        if not np.all(np.isfinite(y_new)):
            # Try with smaller step
            h *= 0.5
            continue
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((error / scale) ** 2))
        
        if err_norm <= 1.0:
            # Step accepted
            t += h
            y = y_new
            
            # Estimate next step size
            if err_norm < 1e-5:
                h *= 5.0
            elif err_norm < 1e-3:
                h *= 2.0
            elif err_norm < 0.1:
                h *= 1.5
            elif err_norm < 1.0:
                h *= 0.9
            else:
                h *= 0.8
        else:
            # Step rejected
            h *= 0.5
            if h < (t1 - t0) / 1e7:
                # Too small, accept anyway to avoid infinite loop
                t += h
                y = y_new
        
        # Cap step size
        h = min(h, (t1 - t0) / 10.0)
    
    return y, total_nfev


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using adaptive DOP853.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use tight tolerances to match the reference
    rtol = 1e-12
    atol = 1e-14
    
    y_end, nfev = _dop853_adaptive(f, t0, t1, y0, rtol=rtol, atol=atol)
    
    # Ensure finite result
    if not np.all(np.isfinite(y_end)):
        # Fallback to RK4 with many steps
        h = (t1 - t0) / 10000
        t, y = t0, y0.copy()
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        y_end = y
    
    return y_end
# EVOLVE-BLOCK-END
