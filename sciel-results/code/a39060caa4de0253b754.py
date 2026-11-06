"""Rewritten program: nonlinear particle-filter-style update with KDE posterior approximation."""
import numpy as np

def _lorenz96_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(x)
    def f(state):
        # Compute derivatives
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # Indices are periodic mod n
        x_im2 = np.roll(state, 2)
        x_im1 = np.roll(state, 1)
        x_ip1 = np.roll(state, -1)
        return (x_ip1 - x_im2) * x_im1 - state + F
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear particle filter update with KDE posterior approximation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations at even indices
    - observed_indices: (10,) array of observed variable indices
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Ensure inputs are float arrays
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    
    # Step 1: Compute observation-space residuals for each member
    # For each member, compute the difference between observed values and member's values at observed indices
    # member_obs = ensemble[:, obs_idx]  shape (n_members, n_obs)
    
    # Step 2: Importance weights based on observation likelihood
    # Likelihood: N(observations | member_obs, observation_variance)
    # Assuming independent Gaussian observation noise
    resid = observations[None, :] - ensemble[:, obs_idx]  # (n_members, n_obs)
    # Log-likelihood: -0.5 * sum(resid^2 / obs_var) - 0.5 * n_obs * log(2*pi*obs_var)
    # We can ignore the constant term for weighting
    obs_var = max(observation_variance, 1e-6)
    log_weights = -0.5 * np.sum(resid ** 2, axis=1) / obs_var
    
    # Normalize weights
    log_weights -= np.max(log_weights)  # for numerical stability
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Step 3: Systematic resampling to get new particle set
    # This gives us a set of particles proportional to the posterior
    # We'll use systematic resampling for low variance
    
    # Compute cumulative weights
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1.0  # ensure it ends at 1
    
    # Systematic resampling
    r = np.random.uniform(0, 1.0 / n_members)
    u = (np.arange(n_members) + r) / n_members
    
    # Find indices
    indices = np.searchsorted(cum_weights, u)
    indices = np.clip(indices, 0, n_members - 1)
    
    resampled = ensemble[indices, :]
    
    # Step 4: Kernel Density Estimate to approximate the posterior
    # We want to represent the posterior distribution using the resampled particles
    # with a KDE. However, since we need to output an ensemble of the same size,
    # we'll use the resampled particles as our new ensemble.
    
    # But to better capture the non-Gaussian structure, we can apply a small
    # deterministic perturbation based on the KDE bandwidth to spread particles
    # around the high-density regions.
    
    # Compute a simple bandwidth estimate (Silverman's rule of thumb)
    # For each dimension, estimate the bandwidth
    # We'll use a global bandwidth based on the spread of the resampled ensemble
    
    # Compute mean and std of resampled ensemble
    mean_state = np.mean(resampled, axis=0)
    std_state = np.std(resampled, axis=0)
    
    # Silverman's rule: h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
    # For simplicity, use h = 1.06 * std * n^(-1/5)
    n_pow = n_members ** (-1.0 / 5.0)
    bandwidth = 1.06 * std_state * n_pow
    
    # Ensure bandwidth is positive and reasonable
    bandwidth = np.maximum(bandwidth, 1e-3)
    
    # Add small deterministic perturbations based on the KDE
    # We'll use a fixed pattern of perturbations to ensure reproducibility
    # and to spread particles around the KDE modes
    
    # Generate deterministic perturbations using a fixed random seed
    rng = np.random.RandomState(42)
    perturbations = rng.normal(0, 1, size=(n_members, n_vars))
    
    # Scale perturbations by bandwidth
    updated_ensemble = resampled + perturbations * bandwidth
    
    # Step 5: Ensure the updated ensemble is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Replace any non-finite values with the mean
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.where(np.isfinite(updated_ensemble), updated_ensemble, mean_state)
    
    return updated_ensemble
