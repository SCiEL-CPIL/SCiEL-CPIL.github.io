<<<<<<< SEARCH
def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g with positivity penalty."""
    if grid is None:
        grid = np.linspace(0.0, 25.0, 50001)
    g = _g_of_lambda(a, grid)
    
    # Penalty for g <= 0: if g is small or negative, the ratio is huge.
    # We use a smooth approximation of the indicator that g > 0.
    # If g <= 0, we want the objective to be very large.
    # Let's use a large constant penalty if min(g) is close to 0 or negative.
    min_g = float(np.min(g))
    if min_g < 1e-6:
        # Strong penalty for violating g > 0
        return 1e10 + (1e-6 - min_g) * 1e8

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    
    # Adaptive temperature: smaller as we get closer to optimum? 
    # For now, keep fixed but allow lower values.
    m = float(np.max(ratio))
    # Use a slightly smaller temperature for sharper peak approximation
    temp = temperature * 0.5 if temperature > 0.01 else temperature
    s = np.exp((ratio - m) / temp)
    return m + temp * math.log(float(np.mean(s)))
=======
def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g with positivity penalty."""
    if grid is None:
        grid = np.linspace(0.0, 25.0, 30001)
    g = _g_of_lambda(a, grid)
    
    # Penalty for g <= 0: if g is small or negative, the ratio is huge.
    # We use a smooth approximation of the indicator that g > 0.
    # If g <= 0, we want the objective to be very large.
    min_g = float(np.min(g))
    if min_g < 1e-6:
        # Strong penalty for violating g > 0
        return 1e10 + (1e-6 - min_g) * 1e8

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    
    m = float(np.max(ratio))
    temp = temperature
    s = np.exp((ratio - m) / temp)
    return m + temp * math.log(float(np.mean(s)))
>>>>>>> REPLACE

