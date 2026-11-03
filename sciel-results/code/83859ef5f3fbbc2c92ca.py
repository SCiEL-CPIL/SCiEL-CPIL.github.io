<<<<<<< SEARCH
def _final_local_search(c, r, max_iter=500, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in multiple directions, keeping the best improvement.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Generate directions: 24 angular directions
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in dirs:
                c_try = c.copy()
                c_try[i] += step_scale * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-13:
                    c = c_try
                    r_cur = r_try
                    improved = True
        if not improved:
            break
    return c, r_cur
=======
def _final_local_search(c, r, max_iter=500, step_scale=1e-5):
    """
    Deterministic local search: for each circle, try small perturbations
    in multiple directions, keeping the best improvement.
    Uses adaptive step size that shrinks as improvements get smaller.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Generate directions: 32 angular directions + 4 axis-aligned
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    current_step = step_scale
    no_improve_count = 0
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for d in dirs:
                c_try = c.copy()
                c_try[i] += current_step * d
                c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                r_try = _radius_fast(c_try)
                if r_try > r_cur + 1e-14:
                    c = c_try
                    r_cur = r_try
                    improved = True
                    no_improve_count = 0
        if not improved:
            # Try negative directions too
            for i in range(N):
                for d in dirs:
                    c_try = c.copy()
                    c_try[i] -= current_step * d
                    c_try = np.clip(c_try, -0.5 + 1e-12, 0.5 - 1e-12)
                    r_try = _radius_fast(c_try)
                    if r_try > r_cur + 1e-14:
                        c = c_try
                        r_cur = r_try
                        improved = True
                        no_improve_count = 0
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 2:
                    break
                # Shrink step size
                current_step *= 0.5
                if current_step < 1e-10:
                    break
    
    return c, r_cur
>>>>>>> REPLACE

