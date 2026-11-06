<<<<<<< SEARCH
def _local_optimize(a, deadline):
    """Local optimization: try reducing each element."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    best_valid = _is_valid(a)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
            for new_val in range(1, old):
                if new_val in a:
                    continue
                trial = list(a)
                trial[i] = new_val
                trial.sort()
                if _is_valid(trial):
                    if trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = trial[-1]
                        best_valid = True
                        improved = True
                        break
    
    return best, best_diam, best_valid
=======
def _local_optimize(a, deadline):
    """Local optimization: try reducing each element, especially the max."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    best_valid = _is_valid(a)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to reduce the max element first (most important for diameter)
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
            # Try values from old-1 down to 1
            for new_val in range(old - 1, 0, -1):
                if new_val in a:
                    continue
                trial = list(a)
                trial[i] = new_val
                trial.sort()
                if _is_valid(trial):
                    if trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = trial[-1]
                        best_valid = True
                        improved = True
                        break
            if improved:
                break
    
    return best, best_diam, best_valid

def _try_reduce_diameter(a, deadline):
    """Try to reduce the maximum element by 1, 2, 3... using local search."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    
    for reduction in range(1, 5):
        if time.time() > deadline:
            break
        target_diam = best_diam - reduction
        if target_diam < 30:
            break
        
        # Try to find a valid set with max element <= target_diam
        # Use a quick local search
        trial = list(best)
        # Try to move the max element down
        max_idx = len(trial) - 1
        for new_max in range(target_diam, best_diam - 1, -1):
            if new_max in trial[:-1]:
                continue
            trial[max_idx] = new_max
            trial.sort()
            if _is_valid(trial):
                best = list(trial)
                best_diam = trial[-1]
                break
        else:
            # If we couldn't reduce by this amount, stop
            break
    
    return best, best_diam
>>>>>>> REPLACE

