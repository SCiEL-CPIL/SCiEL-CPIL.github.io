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
    Uses adaptive step scaling.
    """
    c = c.copy()
    r_cur = _radius_fast(c)
    
    # Generate directions: 32 angular directions for finer resolution
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Also add axis-aligned and diagonal directions
    extra_dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                           [1, 1], [-1, 1], [1, -1], [-1, -1]])
    extra_dirs /= np.sqrt(2)
    dirs = np.vstack([dirs, extra_dirs])
    
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
                    break
        if not improved:
            # Reduce step size
            current_step *= 0.5
            no_improve_count += 1
            if no_improve_count > 5 or current_step < 1e-9:
                break
    return c, r_cur
>>>>>>> REPLACE

