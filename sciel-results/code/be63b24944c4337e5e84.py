import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 140.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Process (RBF) kernel basis.
    y(x) = sum_{i=1}^{N} w_i * exp(-gamma * (x - c_i)^2) + b
    where c_i are fixed centers on a grid, gamma is the inverse length-scale squared,
    and w_i, b are linear parameters.
    We perform a coarse grid search over gamma (width) inside the equation to handle
    the non-linear parameter, while BFGS optimizes the linear weights w_i.
    """
    if x.size == 0:
        return np.array([])
    
    if time.time() - _START_TIME > TIME_LIMIT:
        return np.zeros_like(x)

    # We have 10 parameters.
    # Let's use 8 weights for 8 RBF centers, 1 for bias, and 1 for the width parameter (gamma).
    # Actually, to be safe and allow BFGS to optimize the width as well (though we grid search it),
    # we can just use the first 8 as weights, 9th as bias, and 10th as a log-gamma or similar.
    # However, the prompt says "The evaluator fits params... minimizing training MSE".
    # If we grid search gamma inside equation(), the MSE landscape for BFGS will be non-smooth
    # with respect to the gamma parameter if we use it directly.
    # Better: Fix gamma to a few candidate values? No, that's not adaptive.
    
    # Alternative strategy:
    # Use all 10 params as weights for 10 fixed RBF centers.
    # Fix the width (gamma) to a value that allows capturing the function.
    # But we don't know the function.
    
    # Let's look at the inspiration: "Static, orthogonal Chebyshev feature basis".
    # The current program is a rational function.
    # The observed child p53 used RBF with a grid search and scored 155.5, which is a HUGE improvement 
    # over the parent p46 (1.7e13) and even the inspiration (1e11).
    # Wait, the score is "normalized MSE... lower is better". 
    # 155 is bad (MSE > variance). 1e11 is catastrophic.
    # The seed program score was 17309960182929.9785.
    # The inspiration score was 101239161194.6519.
    # The child p53 score was 155.56.
    # So p53 is MUCH better.
    
    # Strategy: Replicate the successful structure of p53 but refine it.
    # p53 used: y(x) = sum w_i * exp(-gamma * (x - c_i)^2) + b
    # It performed a grid search over gamma.
    
    # Let's use 9 centers and 1 bias? Or 10 centers?
    # With 10 params, if we use 1 for gamma, we have 9 for weights.
    # If we grid search gamma, BFGS only sees the weights as variables (for a fixed gamma step).
    # But BFGS optimizes ALL 10 params simultaneously.
    # If the function depends on params[9] (gamma) in a discontinuous way (grid search), 
    # the gradient for params[9] is effectively zero or noisy.
    # BFGS might get stuck or oscillate.
    
    # Better approach: 
    # Use a smooth mapping for gamma. 
    # Let gamma = exp(params[9]) * base_gamma.
    # Then use 9 weights for 9 centers.
    # This makes the problem smooth for BFGS.
    
    # Centers: 9 points uniformly distributed in [-2, 2].
    # c_i = -2 + (2 - (-2)) * i / (N - 1) for i in 0..N-1
    # N = 9.
    # c_i = -2 + 4 * i / 8 = -2 + 0.5 * i.
    # Centers: -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0.
    
    N = 9
    centers = np.linspace(-2.0, 2.0, N)
    
    # Extract params
    weights = params[:N]
    log_gamma = params[N] # 9th param (index 9)
    
    # Map log_gamma to a reasonable range for gamma.
    # gamma is the inverse squared length scale.
    # Small gamma -> wide bumps. Large gamma -> narrow bumps.
    # For x in [-2, 2], a typical width might be 0.5 to 2.0.
    # gamma = 1 / (2 * sigma^2).
    # If sigma = 0.5, gamma = 2.
    # If sigma = 2.0, gamma = 0.125.
    # Let's map log_gamma to gamma in [0.1, 10.0].
    # gamma = exp(log_gamma) might be too sensitive.
    # Let's use a soft mapping.
    # gamma = 1.0 + np.tanh(log_gamma) * 5.0? Range [-4, 6].
    # Or just gamma = np.exp(log_gamma).
    # Starting from all ones: log_gamma = 1.0 -> gamma = e ~= 2.718.
    # This is a reasonable starting point.
    
    gamma = np.exp(log_gamma)
    
    # Compute RBF values
    # x shape: (M,)
    # centers shape: (N,)
    # diff shape: (M, N)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-gamma * diff**2)
    
    # Prediction
    y = rbf_values @ weights
    
    return y

# EVOLVE-BLOCK-START
# The code above is the complete strategy.
# We define it here to match the block structure if required, 
# but the function equation is the entry point.
# EVOLVE-BLOCK-END
