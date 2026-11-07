import numpy as np
from scipy.stats import norm
import time

# Global state to persist across calls within a single seed/evaluation loop
# We use a dictionary to allow for potential parallelism or resetting if needed,
# but the task implies a sequential loop per seed. 
# To be safe and simple, we use a module-level list.
_evaluated_points = []
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Density-Weighted Expected Improvement.
    
    Parameters:
    mu: numpy array of shape (2000,) - GP mean
    sigma: numpy array of shape (2000,) - GP std dev
    f_best: float - current best observed value
    
    Returns:
    score: numpy array of shape (2000,) - acquisition scores
    """
    global _evaluated_points, _start_time
    
    # Deadline guard: if we are close to the 330s limit, return a safe fallback
    # to avoid being killed. The fallback is pure UCB to ensure finite output.
    if time.time() - _start_time > 320:
        return -mu + 1.0 * sigma

    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We handle sigma near 0 to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Kernel Density Estimate (KDE) of evaluated points
    # The candidates are in [-5, 5]. 
    # We assume the candidate array passed to the GP is uniform in [-5, 5].
    # We need to map these indices back to coordinates for distance calculation.
    # However, we don't receive the coordinates! 
    # Wait, the prompt says "No coordinates are passed."
    # This is a critical constraint. We cannot compute distance to previous points 
    # if we don't know the coordinates of the current 2000 candidates.
    
    # RE-READING CONTRACT: "No coordinates are passed."
    # This means I CANNOT implement a standard KDE penalty based on Euclidean distance 
    # between current candidates and past evaluations, because I don't know where 
    # the current 2000 candidates are located in the [-5, 5] space.
    
    # However, the prompt *also* says: "Strategy to try in this change: Use a Kernel 
    # Density Estimation (KDE) of the evaluated points to construct a density-aware 
    # acquisition score..."
    
    # If coordinates are not passed, how can I implement KDE?
    # Hypothesis: The "coordinates" might be implicitly recoverable or the strategy 
    # must be adapted. 
    # BUT, looking at the "INSPIRATIONS", they use `time` and `numpy`. 
    # If I truly cannot access coordinates, I cannot compute distance.
    
    # Let's look at the "CURRENT PROGRAM". It uses `mu` and `sigma`.
    # Is it possible the task implies that the *index* or some other feature is passed? No.
    
    # Alternative Interpretation: 
    # Perhaps the "KDE" strategy is a red herring in the directive if coordinates 
    # are unavailable, OR I am expected to infer that the candidates are sorted or 
    # have a known distribution? No, they are "fresh uniform candidates".
    
    # Let's re-read carefully: "No coordinates are passed."
    # This makes a spatial KDE impossible without extra state tracking that maps 
    # index -> coordinate, but the indices change every step (2000 fresh uniform 
    # candidates). The mapping from index to coordinate is random and unknown to the 
    # function.
    
    # Therefore, a *spatial* KDE is strictly impossible with the given interface.
    
    # However, the directive says: "Strategy to try in this change: Use a Kernel 
    # Density Estimation (KDE) of the evaluated points..."
    # Maybe the "density" refers to the density of the *posterior uncertainty* or 
    # *mean* values? 
    # Or perhaps the "KDE" is applied to the *history of f_best* or *history of 
    # selected points* in a feature space defined by mu/sigma?
    
    # Let's consider a "Feature-Space KDE".
    # We can store the (mu, sigma) pair of the *selected* point in each previous step.
    # Then, for the current 2000 candidates, we compute the distance in the 
    # (mu, sigma) feature space to these past selected points.
    # This penalizes candidates that look similar (in terms of GP prediction) 
    # to previously selected points, encouraging diversity in the *prediction 
    # space*.
    
    # This is a valid interpretation of "density-aware" when spatial coordinates 
    # are unavailable.
    
    # Implementation Plan:
    # 1. Maintain a list of past (mu_selected, sigma_selected).
    # 2. For current candidates, compute distance to each past point in 2D feature space.
    # 3. Compute a KDE score: sum of Gaussian kernels.
    # 4. Penalty = KDE score (high density = high penalty).
    # 5. Score = EI - alpha * Penalty.
    
    # We need to store the mu and sigma of the *best* candidate (highest score) 
    # from the previous step.
    # But we don't know which candidate was selected by the evaluator! 
    # The evaluator selects the candidate with the *highest score* returned by this function.
    # So, we can track the index of the max score in the previous call.
    
    # Let's refine the state:
    # _history: list of (mu_val, sigma_val) for the selected points.
    # _last_max_idx: index of the max score in the previous call.
    
    # Wait, the function is called once per step.
    # Step 1: Call acquisition(mu, sigma, f_best). Returns scores.
    # Evaluator picks argmax(scores).
    # Step 2: Evaluator evaluates that point, updates GP, gets new mu, sigma.
    # Step 3: Call acquisition(new_mu, new_sigma, new_f_best).
    
    # We don't receive the coordinates, but we DO receive the mu and sigma arrays 
    # for the *new* 2000 candidates.
    # We can store the mu and sigma values of the *selected* point from the 
    # *previous* step.
    
    # How do we know which point was selected?
    # We return `score`. The evaluator picks `argmax(score)`.
    # So, we can store `mu[argmax(score)]` and `sigma[argmax(score)]` in our state.
    
    # Let's implement this "Feature-Space KDE".
    
    global _feature_history
    if not hasattr(acquisition, '_feature_history'):
        acquisition._feature_history = []
    
    # Compute EI as before
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute Density Penalty in Feature Space
    if len(acquisition._feature_history) > 0:
        # Stack history into a matrix of shape (N_history, 2)
        hist_mu = np.array([h[0] for h in acquisition._feature_history])
        hist_sigma = np.array([h[1] for h in acquisition._feature_history])
        
        # Current candidates shape: (2000, 1) -> (2000, 2)
        curr_mu = mu.reshape(-1, 1)
        curr_sigma = sigma.reshape(-1, 1)
        curr_features = np.hstack([curr_mu, curr_sigma])
        
        # Compute squared Euclidean distance between each current candidate and each history point
        # dist^2 = curr^2 + hist^2 - 2 * curr * hist^T
        # Shape: (2000, 1) + (1, N_hist) - 2 * (2000, N_hist)
        
        # To avoid memory blowup if history is large, we can cap the history size
        # or process in chunks. 2000 * 50 is small.
        
        # Standard deviation of the kernel. 
        # The scale of mu and sigma varies. We should normalize the features.
        # Let's normalize by the range of current mu/sigma or use a fixed scale.
        # Better: Normalize each dimension by its std in the current batch + history.
        
        all_mu = np.concatenate([mu, hist_mu])
        all_sigma = np.concatenate([sigma, hist_sigma])
        
        mu_mean = np.mean(all_mu)
        mu_std = np.std(all_mu) + 1e-9
        sigma_mean = np.mean(all_sigma)
        sigma_std = np.std(all_sigma) + 1e-9
        
        curr_mu_norm = (mu - mu_mean) / mu_std
        curr_sigma_norm = (sigma - sigma_mean) / sigma_std
        
        hist_mu_norm = (hist_mu - mu_mean) / mu_std
        hist_sigma_norm = (hist_sigma - sigma_mean) / sigma_std
        
        curr_features_norm = np.vstack([curr_mu_norm, curr_sigma_norm]).T # (2000, 2)
        hist_features_norm = np.vstack([hist_mu_norm, hist_sigma_norm]).T # (N_hist, 2)
        
        # Compute pairwise distances
        # Using broadcasting
        # curr: (2000, 1, 2), hist: (1, N_hist, 2)
        diff = curr_features_norm[:, np.newaxis, :] - hist_features_norm[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2) # (2000, N_hist)
        
        # Gaussian KDE: sum of exp(-0.5 * sq_dist / bandwidth^2)
        # Bandwidth: 1.0 in normalized space
        bandwidth = 1.0
        kernel_vals = np.exp(-0.5 * sq_dist / (bandwidth**2))
        
        # Density estimate
        density = np.sum(kernel_vals, axis=1) # (2000,)
        
        # Normalize density to [0, 1] for weighting
        max_density = np.max(density)
        if max_density > 0:
            density_norm = density / max_density
        else:
            density_norm = np.zeros_like(density)
            
        # Penalty: Higher density means more penalty
        # We want to *subtract* this from EI
        # Weight alpha: How strongly to penalize?
        # If alpha is too high, we ignore EI. If too low, we ignore diversity.
        # Let's start with alpha = 0.5 * max_ei
        max_ei = np.max(ei)
        alpha = 0.5 * max_ei if max_ei > 0 else 0.0
        penalty = alpha * density_norm
    else:
        penalty = 0.0
    
    # Final Score
    score = ei - penalty
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Update State: Store the feature of the selected point (argmax score)
    max_idx = np.argmax(score)
    acquisition._feature_history.append((mu[max_idx], sigma[max_idx]))
    
    # Cap history size to prevent memory growth and computation time
    if len(acquisition._feature_history) > 50:
        acquisition._feature_history = acquisition._feature_history[-50:]
        
    return score
