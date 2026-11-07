import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a universal sine-based feature map.
    
    Strategy:
    Instead of a fixed small set of non-linear units (like tanh or polynomials), 
    we use a 'Fourier Features' or 'Random Fourier Features' style approach with 
    fixed, deterministic frequencies. This creates a high-dimensional, smooth, 
    periodic feature space that can approximate a very wide class of smooth 
    functions on the interval [-2, 2].
    
    The model is:
    y(x) = c0 + sum_{i=1}^{N} a_i * sin(f_i * x + p_i)
    
    where f_i and p_i are fixed constants chosen to provide good coverage of 
    frequencies, and a_i are the learnable parameters (part of the 10-param array).
    
    Since the relationship between y and params is linear, the optimization landscape 
    for BFGS is convex (quadratic MSE), ensuring reliable convergence from the 
    initial 'all ones' start, unlike the non-convex landscape of the previous 
    sigmoid/polynomial mix.
    
    Parameters (length 10):
    params[0]: Constant term (bias)
    params[1:10]: Weights for 9 sine basis functions
    """
    c0 = params[0]
    weights = params[1:10]
    
    # Fixed frequencies and phases for the basis functions.
    # We choose frequencies that cover a range from low to medium-high.
    # Since x is in [-2, 2], f*x ranges roughly [-2f, 2f].
    # We want to capture features up to maybe 20-30 radians across the domain.
    # Let's pick 9 distinct frequencies.
    # A geometric progression or specific set of harmonics.
    # Using integers 1, 2, 3, 4, 5, 7, 10, 13, 17 to avoid simple aliasing and cover bands.
    # Or simply 1..9. Let's use a mix to be safe.
    # Actually, for a "universal" feel, let's use a set of frequencies that are 
    # well-spaced in log-scale or linear scale.
    # Given the domain is small [-2,2], high frequencies might be overkill if the 
    # target is smooth, but having them available allows fitting sharper features 
    # if present.
    
    # Let's use frequencies: 1, 2, 3, 4, 5, 6, 8, 10, 12
    # And phases: 0, pi/2, pi, 3pi/2, etc., or just 0 for simplicity?
    # With both sin and cos we'd need more params. With only sin, we can shift phase.
    # Let's use sin(f*x) and cos(f*x) would be better, but we only have 9 weights.
    # So we must mix phases or stick to sin.
    # A pure sine basis with varying frequencies is sufficient for a universal 
    # approximator on a bounded domain if we have enough terms.
    
    # Frequencies:
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0])
    
    # Compute the feature matrix: shape (len(x), 9)
    # X_freqs = x[:, None] * freqs[None, :]
    # features = np.sin(X_freqs)
    
    # To be more robust, let's add some phase shifts or use a combination.
    # Actually, just sin(f*x) is fine. If the target has a cosine component, 
    # sin(f*x + pi/2) can cover it, but we don't have separate params for phases.
    # However, a linear combination of sines with different frequencies can 
    # approximate cosines as well (via sum-to-product identities or just 
    # approximation theory).
    
    # Let's stick to sin(f_i * x).
    X = x[:, np.newaxis]
    F = freqs[np.newaxis, :]
    
    # features shape: (N, 9)
    features = np.sin(X * F)
    
    # Linear combination
    # y = c0 + weights @ features.T
    y = c0 + features @ weights
    
    return y
