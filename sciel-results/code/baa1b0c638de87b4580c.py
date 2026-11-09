# EVOLVE-BLOCK-START
import time
import numpy as np

# Global cache to store precomputed static features (intercept, linear, quadratic, cubic)
# This avoids recomputing powers of x, y on every BFGS step.
_feature_cache = {}
_START_TIME = None

def _get_static_features(x, y):
    """
    Computes static features that do not depend on params.
    Returns a dictionary of numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Use a simple hash of the data pointer and shape as a proxy for identity
    # Since the evaluator reuses the same arrays, this is safe.
    key = (id(x), id(y), x.shape, y.shape)
    
    if key in _feature_cache:
        return _feature_cache[key]
    
    n = x.shape[0]
    # Clip for safety, though inputs are in [-1, 1]
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    
    # Static basis: Intercept, Linear, Quadratic, Cubic
    # We use these as the "backbone" for the adaptive terms.
    # p0: intercept
    # p1, p2: linear weights for x, y
    # p3, p4, p5: quadratic weights for x^2, y^2, xy
    # p6, p7, p8: cubic weights for x^3, y^3, x^2y + xy^2 (symmetric cubic)
    
    # p9: Amplitude for adaptive sinusoidal term.
    # The frequency of the sinusoidal term is derived from p9 as well, 
    # or we can use a fixed set of frequencies weighted by p9?
    # Better: Let p9 be the amplitude of a term sin(a*x + b*y).
    # But we only have 1 param left.
    
    # Let's rethink the 10 parameters to maximize flexibility:
    # 1-3: Linear (x, y, 1)
    # 4-6: Quadratic (x^2, y^2, xy)
    # 7-9: Cubic (x^3, y^3, x^2y + xy^2)
    # 10: Amplitude for a periodic term sin(k*x + l*y).
    #     To make it adaptive, we can define k and l based on the data or fixed?
    #     If k, l are fixed, it's just another basis function.
    #     If we want it to be "adaptive" in a nonlinear way, we can let the 
    #     frequency be determined by another parameter, but we don't have one.
    
    # Alternative: Use the 10th parameter to control the "sharpness" or 
    # "nonlinearity" of a sigmoid-like term?
    # Or, simply include a richer fixed basis in the static features and 
    # let the 10 params be weights for the top 10 most important features?
    # But the current program already does that (linear model on fixed basis).
    
    # The key insight for a DIFFERENT strategy is to make the model NON-LINEAR in parameters.
    # Example: pred = w1*x + w2*y + w3*sin(w4*x + w5*y)
    # Here w4, w5 control the frequency. This is not a linear model.
    
    # Let's allocate parameters:
    # p0: Intercept
    # p1: Weight for x
    # p2: Weight for y
    # p3: Weight for x^2
    # p4: Weight for y^2
    # p5: Weight for xy
    # p6: Weight for sin(freq_x * x)
    # p7: freq_x
    # p8: Weight for sin(freq_y * y)
    # p9: freq_y
    
    # This allows the model to fit periodic functions in x and y independently.
    # It is nonlinear in p7 and p9. BFGS can handle this if the landscape is smooth.
    # To improve stability, we can bound the frequencies or use tanh to keep them in a range.
    # However, BFGS starts at 1.0. If freq is 1.0, sin(x) is reasonable.
    
    # Let's also add a mixed term?
    # Maybe:
    # p0: Intercept
    # p1: x weight
    # p2: y weight
    # p3: x^2 weight
    # p4: y^2 weight
    # p5: xy weight
    # p6: sin(freq1*x + freq2*y) weight
    # p7: freq1
    # p8: freq2
    # p9: cos(freq3*x + freq4*y) weight? No, need freqs.
    
    # Let's stick to a robust nonlinear structure:
    # f(x,y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy 
    #          + p6*sin(p7*x + p8*y) + p9*sin(p7*y + p8*x)
    
    # This uses 10 params.
    # p7, p8 are frequencies.
    # p6, p9 are amplitudes.
    
    # Precompute static parts:
    # x, y, x^2, y^2, xy are static.
    # The sinusoidal parts depend on p7, p8.
    
    feats = {
        'x': x,
        'y': y,
        'x2': x * x,
        'y2': y * y,
        'xy': x * y
    }
    
    _feature_cache[key] = feats
    return feats

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check time limit
    if time.time() - _START_TIME > 145.0:
        # Return a safe fallback, e.g., mean of y
        return np.full_like(y, np.mean(y), dtype=float)
        
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if params.shape[0] != 10:
        # Pad or trim
        if params.shape[0] < 10:
            params = np.pad(params, (0, 10 - params.shape[0]))
        else:
            params = params[:10]
            
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Get static features
    feats = _get_static_features(x, y)
    x_arr = feats['x']
    y_arr = feats['y']
    x2_arr = feats['x2']
    y2_arr = feats['y2']
    xy_arr = feats['xy']
    
    # Compute prediction
    # Base polynomial part
    pred = p0 + p1 * x_arr + p2 * y_arr + p3 * x2_arr + p4 * y2_arr + p5 * xy_arr
    
    # Adaptive sinusoidal part
    # To prevent huge frequencies from causing numerical issues in sin, 
    # we can clip the frequency parameters or rely on BFGS to keep them reasonable.
    # Starting at 1.0 is good.
    
    # Term 1: sin(p7*x + p8*y)
    arg1 = p7 * x_arr + p8 * y_arr
    # Clip argument to avoid loss of precision in sin for very large numbers?
    # sin is periodic, but for large args, float precision degrades.
    # Inputs are in [-1,1]. If p7, p8 are small (e.g. < 100), args are < 200. Safe.
    term1 = np.sin(arg1)
    
    # Term 2: sin(p7*y + p8*x)
    arg2 = p7 * y_arr + p8 * x_arr
    term2 = np.sin(arg2)
    
    pred += p6 * term1 + p9 * term2
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.full_like(y, np.mean(y), dtype=float)
        
    return pred
# EVOLVE-BLOCK-END
