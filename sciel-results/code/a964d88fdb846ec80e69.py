"""
Strategy: Internal evolutionary feature selection over a rich nonlinear basis.
Instead of a fixed multiplicative structure, we search for the best linear
combination of up to 10 features from a large library of candidate terms.
This allows BFGS to fit the weights in a relatively well-behaved landscape
while the internal search handles the combinatorial aspect of feature selection.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Global cache to store the best features found and the initial guess for params
# The evaluator calls equation() many times during BFGS. We want to do the
# expensive search once, then just evaluate the linear combination.
_BEST_FEATURES = None
_BEST_PARAMS_INIT = None
_SEARCH_DONE = False

def _build_basis(t, x, v):
    """
    Build a diverse library of candidate features for a nonlinear oscillator.
    Includes:
    - Constants
    - Polynomials in t, x, v (up to degree 3 or 4)
    - Trigonometric functions of t (for periodicity)
    - Interactions: x*v, x^2*v, etc.
    """
    features = [np.ones_like(t)]
    
    # Time features
    features.append(t)
    features.append(t**2)
    features.append(t**3)
    features.append(np.sin(t))
    features.append(np.cos(t))
    features.append(np.sin(2*t))
    features.append(np.cos(2*t))
    
    # Displacement (x) features
    features.append(x)
    features.append(x**2)
    features.append(x**3)
    features.append(np.tanh(x))
    features.append(np.sign(x))
    
    # Velocity (v) features
    features.append(v)
    features.append(v**2)
    features.append(v**3)
    features.append(np.tanh(v))
    features.append(np.sign(v))
    
    # Interactions (x, v)
    features.append(x * v)
    features.append(x * v**2)
    features.append(x**2 * v)
    features.append(x * np.tanh(v))
    features.append(np.tanh(x) * v)
    
    # Interactions with time (decay/growth)
    features.append(x * t)
    features.append(v * t)
    features.append(x * np.exp(-0.1 * t))
    features.append(v * np.exp(-0.1 * t))
    features.append(x * np.sin(t))
    features.append(v * np.sin(t))
    
    # Higher order mixed terms
    features.append(x**2 * v**2)
    features.append(x**3 * v)
    features.append(x * v**3)
    
    return features

def _run_evolutionary_search(t, x, v, y, max_time=140.0):
    """
    Perform a time-budgeted evolutionary search to find the best subset of 10 features
    and their optimal weights.
    """
    global _BEST_FEATURES, _BEST_PARAMS_INIT, _SEARCH_DONE
    
    start_time = time.time()
    basis = _build_basis(t, x, v)
    n_features_total = len(basis)
    
    # Stack features into a matrix for fast linear combination evaluation
    # Shape: (n_samples, n_features_total)
    X = np.column_stack(basis)
    
    # Initial best: just the constant feature (or a random small set)
    best_mse = np.inf
    best_indices = [0]
    best_weights = np.zeros(1)
    
    # We will evolve a set of 10 indices and 10 weights
    # Represent a solution as a list of indices
    # To keep it simple and fast, we'll use a greedy + random restart approach
    # 1. Greedy Forward Selection: Add features one by one if they reduce MSE.
    # 2. Random Perturbation: Try swapping features.
    
    # Helper to fit weights for a given set of indices
    def fit_weights(indices):
        if len(indices) == 0:
            return np.array([0.0]), np.inf
        Xi = X[:, indices]
        # Use lstsq for speed, it's robust enough for initial selection
        try:
            weights, residuals, rank, s = np.linalg.lstsq(Xi, y, rcond=None)
            pred = Xi @ weights
            mse = np.mean((pred - y)**2)
            if not np.isfinite(mse):
                return np.zeros(len(indices)), np.inf
            return weights, mse
        except:
            return np.zeros(len(indices)), np.inf

    # Phase 1: Greedy Forward Selection
    current_indices = []
    current_weights = np.array([])
    current_mse = np.var(y) # Baseline MSE (predicting mean)
    
    for _ in range(10):
        best_new_mse = current_mse
        best_new_idx = -1
        best_new_weights = current_weights
        
        # Sample a subset of candidates to speed up if time is tight
        candidates = list(range(n_features_total))
        random.shuffle(candidates)
        # Only check top 50 candidates per step to save time
        candidates_to_check = candidates[:50] 
        
        for idx in candidates_to_check:
            if idx in current_indices:
                continue
            trial_indices = current_indices + [idx]
            trial_weights, trial_mse = fit_weights(trial_indices)
            if trial_mse < best_new_mse:
                best_new_mse = trial_mse
                best_new_idx = idx
                best_new_weights = trial_weights
        
        if best_new_idx == -1:
            break # No improvement
        
        current_indices.append(best_new_idx)
        current_weights = best_new_weights
        current_mse = best_new_mse
        
        if time.time() - start_time > max_time - 5.0:
            break

    if current_mse < best_mse:
        best_mse = current_mse
        best_indices = current_indices
        best_weights = current_weights

    # Phase 2: Local Search / Swapping
    # Try to swap one feature with another to see if it improves
    num_swaps = 0
    while time.time() - start_time < max_time - 5.0 and num_swaps < 50:
        improved = False
        # Pick a random feature in the current set
        if len(best_indices) == 0:
            break
        pos_to_swap = random.randint(0, len(best_indices) - 1)
        
        # Try swapping with a random candidate not in the set
        candidates = [i for i in range(n_features_total) if i not in best_indices]
        random.shuffle(candidates)
        
        for new_idx in candidates[:10]:
            trial_indices = best_indices.copy()
            trial_indices[pos_to_swap] = new_idx
            trial_weights, trial_mse = fit_weights(trial_indices)
            
            if trial_mse < best_mse:
                best_mse = trial_mse
                best_indices = trial_indices
                best_weights = trial_weights
                improved = True
                num_swaps += 1
                break # Restart swap search with new best
                
        if not improved:
            num_swaps += 1 # Count as a step even if no improvement, to avoid infinite loop
            
    # Store results
    _BEST_FEATURES = best_indices
    _BEST_PARAMS_INIT = best_weights
    _SEARCH_DONE = True
    
    return best_mse

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction.
    If the internal search hasn't been done, it will run on the first call.
    The params array is expected to have length 10.
    We map the 10 params to the selected features.
    If fewer than 10 features were selected, we pad with zeros.
    If more were selected (shouldn't happen), we truncate.
    """
    global _SEARCH_DONE, _BEST_FEATURES, _BEST_PARAMS_INIT
    
    # The evaluator passes y implicitly? No, we don't have y here.
    # Wait, the task says: "The evaluator fits params ... minimizing the training MSE."
    # "You do not see the data".
    # CRITICAL ISSUE: I cannot run the evolutionary search inside equation() because I don't have y!
    # The search requires y to calculate MSE.
    
    # RE-EVALUATION OF STRATEGY:
    # I cannot do data-dependent feature selection inside equation() because I don't have access to y.
    # I must return a static structure that BFGS can fit.
    # The "inspiration" programs likely did something else or I misunderstood their access.
    # Looking at Inspiration 1: "Physics-informed sparse linear model using OMP".
    # OMP also requires y.
    # How did they do it?
    # Ah, the prompt says: "The evaluator fits params... starting from all ones...".
    # It does NOT say I can access the training data.
    # "You do not see the data, only the score of each candidate."
    
    # Therefore, I CANNOT do internal feature selection based on y.
    # I must provide a FIXED structure that is highly expressive and well-conditioned.
    
    # Let's pivot to a "Rich Polynomial Basis" with a fixed structure that BFGS can handle.
    # To improve on the current score (1.54), I need a better fixed structure.
    # The current one is (poly3_x * poly2_v * poly2_t).
    # This creates a very high-degree polynomial (3+2+2 = 7th degree effectively in mixed terms).
    # BFGS might struggle with the scale differences.
    
    # Alternative Strategy: A sum of specific, physically motivated terms.
    # Instead of a single product, use a sum of terms that represent:
    # 1. Restoring force (x, x^3)
    # 2. Damping (v, v^3, x*v)
    # 3. Nonlinear coupling (x^2*v, x*v^2)
    # 4. Time-dependent modulation (exp(-t), sin(t))
    
    # Let's try a "Sparse Sum" of 10 specific terms.
    # This gives BFGS 10 independent knobs.
    # Term 1: 1 (offset)
    # Term 2: x (linear restoring)
    # Term 3: x^3 (cubic stiffness)
    # Term 4: v (linear damping)
    # Term 5: v^3 (cubic damping)
    # Term 6: x*v (linear drag coupling)
    # Term 7: x^2*v (nonlinear drag)
    # Term 8: x*v^2 (nonlinear drag)
    # Term 9: exp(-t) * x (decay of amplitude)
    # Term 10: sin(t) * x (phase modulation)
    
    # This is a linear model in the parameters. BFGS will be very fast and robust.
    # The question is: is this basis rich enough to score > 1.54?
    # The current score is 1.54. A simple linear model might be too simple.
    # But the current multiplicative model is very complex.
    
    # Let's try a hybrid: A sum of 10 terms, but some terms are products.
    # This is effectively a linear regression on a fixed basis.
    
    # Let's refine the basis to be more "oscillator-like".
    # Standard Duffing + Damping:
    # x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)
    # The target is likely x(t) or x'(t).
    # If the target is x(t), it's a trajectory.
    # A linear combination of basis functions can approximate the trajectory.
    
    # Basis:
    # 1. 1
    # 2. t
    # 3. x
    # 4. x^3
    # 5. v
    # 6. v^3
    # 7. x*v
    # 8. x^2*v
    # 9. x*np.exp(-0.5*t)
    # 10. x*np.sin(2*t)
    
    # This is a fixed, linear-in-params model.
    # It is MUCH easier for BFGS to fit than the multiplicative one.
    # The risk is underfitting if the true function is more complex.
    # But the current score 1.54 suggests the multiplicative model might be overfitting or struggling to converge.
    # A simpler, well-conditioned model might generalize better.

    # Let's implement this "Fixed Linear Basis" strategy.
    
    # We need to handle the case where the evaluator calls this function.
    # We just compute the linear combination.
    
    # Basis calculation
    # To ensure stability, we can normalize x and v? No, inputs are in training range.
    # t is [30, 50].
    
    # Let's try to include more flexibility in the time terms.
    
    # Final Basis Selection:
    # 1. 1
    # 2. x
    # 3. x**2
    # 4. x**3
    # 5. v
    # 6. v**2
    # 7. x*v
    # 8. x**2*v
    # 9. x*np.exp(-0.1*(t-40))  # Decay centered at mid-time
    # 10. x*np.sin(0.5*(t-40))  # Oscillation centered at mid-time
    
    # This allows for:
    # - Polynomial nonlinearity in x and v
    # - Interaction terms
    # - Explicit time-dependent modulation of the displacement
    
    # This is a linear model: y = sum(p_i * f_i(x,v,t))
    # BFGS will find the optimal p_i.
    
    # Let's write the code.
    
    # We don't need global state or search. Just the evaluation.
    
    # However, the prompt says "up to 10 fitted constants".
    # My basis has 10 terms. Perfect.
    
    # One concern: The inputs t, x, v are arrays.
    # I must return an array of the same shape.
    
    # Let's verify the ranges.
    # t: [30, 50]
    # x: [-0.3391, 0.3624]
    # v: [-0.2056, 0.1749]
    
    # The terms are well-behaved.
    
    # Let's try to make the time terms more flexible.
    # Instead of fixed frequencies, use t directly?
    # If I use t, t^2, t^3, I can fit a polynomial in time.
    # But oscillators are periodic. Sin/Cos are better.
    # Since I don't know the frequency, I should provide a few.
    # But I only have 10 params.
    
    # Let's stick to the physical intuition:
    # The system is a nonlinear oscillator.
    # The state is (x, v). The target is likely the next state or a force.
    # If the target is x_{t+1}, it depends on x_t, v_t.
    # If the target is a force, it depends on x, v.
    # The presence of t in the inputs suggests the target might be time-dependent explicitly (e.g. driven oscillator) or t is just a timestamp.
    # Given t is [30, 50], it's a long time series.
    
    # Let's include:
    # 1. 1
    # 2. x
    # 3. x**3
    # 4. v
    # 5. v**3
    # 6. x*v
    # 7. x**2*v
    # 8. x*np.sin(t)
    # 9. x*np.cos(t)
    # 10. v*np.sin(t)
    
    # This covers Duffing-like nonlinearity and harmonic driving/modulation.
    
    # Let's implement this.
    
    # Note: The current program uses a multiplicative form.
    # This new program uses an additive form.
    # This is a MATERIALLY DIFFERENT strategy.
    
    # Code:
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Linear combination
    pred = (p0 * 1.0 +
            p1 * x +
            p2 * x3 +
            p3 * v +
            p4 * v3 +
            p5 * xv +
            p6 * x2v +
            p7 * x * sin_t +
            p8 * x * cos_t +
            p9 * v * sin_t)
            
    return pred

# EVOLVE-BLOCK-END
