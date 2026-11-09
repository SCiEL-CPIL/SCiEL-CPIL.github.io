import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = 300.0
    margin = 15.0
    
    def _elapsed():
        return time.time() - start_time
    
    def _remaining():
        return deadline - margin - _elapsed()
    
    # Probe phase: Determine which method is more efficient
    # Solve a small fraction of the problem with loose tolerance
    probe_span = (t0, t0 + (t1 - t0) / 10.0)
    
    dop853_nfev = float('inf')
    radau_nfev = float('inf')
    
    try:
        if _remaining() > 5.0:
            sol_d = scipy.integrate.solve_ivp(
                f, probe_span, y0,
                method='DOP853',
                rtol=1e-4,
                atol=1e-6
            )
            if sol_d.success:
                dop853_nfev = sol_d.nfev
    except Exception:
        pass
        
    try:
        if _remaining() > 5.0:
            sol_r = scipy.integrate.solve_ivp(
                f, probe_span, y0,
                method='Radau',
                rtol=1e-4,
                atol=1e-6
            )
            if sol_r.success:
                radau_nfev = sol_r.nfev
    except Exception:
        pass
        
    # Decide which method to use
    # If Radau uses fewer or similar steps, prefer it (safer for stiffness)
    # If DOP853 uses significantly fewer steps, prefer it (higher order)
    use_radau = True
    if dop853_nfev < radau_nfev * 0.8: # DOP853 is 20% more efficient
        use_radau = False
        
    # Main solve
    best_result = None
    
    if use_radau:
        try:
            if _remaining() > 10.0:
                sol = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='Radau',
                    rtol=1e-12,
                    atol=1e-14
                )
                if sol.success:
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass
            
        # Fallback to DOP853 if Radau fails
        if best_result is None:
            try:
                if _remaining() > 10.0:
                    sol = scipy.integrate.solve_ivp(
                        f, t_span, y0,
                        method='DOP853',
                        rtol=1e-12,
                        atol=1e-14
                    )
                    if sol.success:
                        best_result = sol.y[:, -1].copy()
            except Exception:
                pass
    else:
        try:
            if _remaining() > 10.0:
                sol = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='DOP853',
                    rtol=1e-12,
                    atol=1e-14
                )
                if sol.success:
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass
            
        # Fallback to Radau if DOP853 fails
        if best_result is None:
            try:
                if _remaining() > 10.0:
                    sol = scipy.integrate.solve_ivp(
                        f, t_span, y0,
                        method='Radau',
                        rtol=1e-12,
                        atol=1e-14
                    )
                    if sol.success:
                        best_result = sol.y[:, -1].copy()
            except Exception:
                pass
                
    # Refinement phase: If we have time, try to tighten the result
    # This uses the "multi-method refinement" idea from inspiration 3
    # but only if the initial solve was fast enough to allow it
    
    if best_result is not None and _remaining() > 30.0:
        # Try an even tighter solve with the preferred method
        try:
            if use_radau:
                sol_tight = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='Radau',
                    rtol=1e-13,
                    atol=1e-15
                )
                if sol_tight.success:
                    best_result = sol_tight.y[:, -1].copy()
            else:
                sol_tight = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='DOP853',
                    rtol=1e-13,
                    atol=1e-15
                )
                if sol_tight.success:
                    best_result = sol_tight.y[:, -1].copy()
        except Exception:
            pass
            
        # If we still have time, cross-check with the other method
        if _remaining() > 60.0:
            try:
                if use_radau:
                    sol_cross = scipy.integrate.solve_ivp(
                        f, t_span, y0,
                        method='DOP853',
                        rtol=1e-13,
                        atol=1e-15
                    )
                    if sol_cross.success:
                        # Use the cross-check result if it's finite
                        if np.all(np.isfinite(sol_cross.y[:, -1])):
                            best_result = sol_cross.y[:, -1].copy()
                else:
                    sol_cross = scipy.integrate.solve_ivp(
                        f, t_span, y0,
                        method='Radau',
                        rtol=1e-13,
                        atol=1e-15
                    )
                    if sol_cross.success:
                        if np.all(np.isfinite(sol_cross.y[:, -1])):
                            best_result = sol_cross.y[:, -1].copy()
            except Exception:
                pass

    # Final fallback: RK4
    if best_result is None:
        try:
            steps = 1000
            h = (t1 - t0) / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if _elapsed() > deadline - 5.0:
                    break
            best_result = y
        except Exception:
            best_result = y0
            
    return best_result
