"""Root finder for 3 fixed 2-D systems: Multi-start Broyden's method."""
import numpy as np
import time


def _broyden_solve(F, x_start, max_calls=100, deadline=0.0):
    """
    Solve F(x)=0 using Broyden's method.
    Returns (x, Fx, success, calls_made).
    """
    x = np.array(x_start, dtype=float)
    call_count = 0
    
    def safe_F(x_in):
        nonlocal call_count
        if time.time() > deadline:
            raise TimeoutError()
        call_count += 1
        result = F(x_in)
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            result = result.flatten()[:2]
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result

    try:
        Fx = safe_F(x)
    except TimeoutError:
        return x, Fx, False, call_count

    if np.linalg.norm(Fx) <= 1e-8:
        return x, Fx, True, call_count

    # Initial Jacobian via finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_e = safe_F(x + e)
        except TimeoutError:
            return x, Fx, False, call_count
        J[:, j] = (Fx_e - Fx) / h

    best_x = x.copy()
    best_Fx = Fx.copy()
    best_norm = np.linalg.norm(Fx)

    for _ in range(max_calls - call_count):
        if time.time() > deadline:
            break
        if call_count >= max_calls:
            break

        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm < best_norm:
            best_norm = Fx_norm
            best_x = x.copy()
            best_Fx = Fx.copy()

        if Fx_norm <= 1e-8:
            return x, Fx, True, call_count

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Damping / Line Search
        # Try full step
        x_new = x + dx
        try:
            Fx_new = safe_F(x_new)
        except TimeoutError:
            break
        
        Fx_new_norm = np.linalg.norm(Fx_new)
        
        if np.all(np.isfinite(Fx_new)) and Fx_new_norm < Fx_norm:
            # Accept step
            s = x_new - x
            y = Fx_new - Fx
            if s @ s > 1e-14:
                J = J + np.outer(y - J @ s, s) / (s @ s)
            x = x_new
            Fx = Fx_new
        else:
            # Reject step, try smaller steps
            accepted = False
            for factor in [0.5, 0.25, 0.1]:
                x_try = x + factor * dx
                try:
                    Fx_try = safe_F(x_try)
                except TimeoutError:
                    break
                Fx_try_norm = np.linalg.norm(Fx_try)
                if np.all(np.isfinite(Fx_try)) and Fx_try_norm < Fx_norm:
                    s = x_try - x
                    y = Fx_try - Fx
                    if s @ s > 1e-14:
                        J = J + np.outer(y - J @ s, s) / (s @ s)
                    x = x_try
                    Fx = Fx_try
                    accepted = True
                    break
            if not accepted:
                # Stagnation, break
                break

    return best_x, best_Fx, best_norm <= 1e-8, call_count


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    x0 = np.array(x0, dtype=float)
    
    # 1. Check if origin is a root (homotopy hint)
    # This costs 1 call.
    # If F(0) is small, return 0.
    # But we need to be careful with call budget.
    # Let's just include origin in the multi-start list.
    
    candidates = [
        x0,
        x0 * 0.5,
        x0 * 2.0,
        np.zeros(2),
        x0 + np.array([1.0, 1.0]),
        x0 + np.array([-1.0, -1.0])
    ]

    best_x = x0
    best_norm = np.inf
    total_calls = 0

    # We have a budget of ~450 calls per system to be safe (500 limit).
    # Let's allocate 75 calls per candidate.
    calls_per_candidate = 70

    for start in candidates:
        if time.time() > deadline:
            break
        
        x, Fx, success, calls_made = _broyden_solve(F, start, max_calls=calls_per_candidate, deadline=deadline)
        total_calls += calls_made
        
        if success:
            return x
        
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_norm = norm
            best_x = x

        # If we are getting close, stop trying new starts?
        # Or if we used too many calls.
        if total_calls > 400:
            break

    return best_x
