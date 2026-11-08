"""Program for minimizing the normalized maximum autoconvolution of a step function."""
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Represent the step function as h = |z|^2 where z is a complex vector.
    Use Wirtinger Flow: optimize the phase of z while keeping |z| fixed,
    or use full complex gradient descent with analytical gradients.
    """
    
    def compute_autoconvolution_max(h):
        """Compute the maximum of the autoconvolution of step function h."""
        conv = np.convolve(h, h, mode='full')
        return np.max(conv) / N, np.argmax(conv)
    
    def wirtinger_gradient(z, h, max_idx):
        """
        Compute the Wirtinger gradient for the objective.
        For h = |z|^2, the Wirtinger gradient is:
        ∂f/∂conj(z[i]) = 2 * z[i] * (∂f/∂h[i])
        
        For autoconvolution at max_idx:
        conv[max_idx] = sum_i h[i] * h[max_idx - i] (with bounds)
        ∂conv[max_idx]/∂h[i] = h[max_idx - i] + h[i] (when both in range)
        """
        n = len(h)
        grad_h = np.zeros(n)
        
        for i in range(n):
            j = max_idx - i
            if 0 <= j < n:
                grad_h[i] = h[j] + h[i]
            else:
                # Only one term contributes
                grad_h[i] = h[i]
        
        # Wirtinger gradient: ∂f/∂conj(z) = 2 * z * grad_h
        grad_z = 2 * z * grad_h
        
        return grad_z
    
    # Initialize z: use random phases with uniform magnitudes
    np.random.seed(42)
    # Start with a flat-ish profile to avoid bad local minima
    magnitudes = np.ones(N) * np.sqrt(N)  # so that sum(|z|^2) = N
    phases = 2 * np.pi * np.random.randn(N)
    z = magnitudes * np.exp(1j * phases)
    
    best_obj = 1e10
    best_z = z.copy()
    
    # Wirtinger Flow iterations
    for iteration in range(500):
        h = np.abs(z)**2
        h_sum = np.sum(h)
        if h_sum < 1e-10:
            break
        h_norm = h * (N / h_sum)
        
        obj, max_idx = compute_autoconvolution_max(h_norm)
        
        if obj < best_obj:
            best_obj = obj
            best_z = z.copy()
        
        # Compute gradient
        grad_z = wirtinger_gradient(z, h_norm, max_idx)
        
        # Adaptive learning rate
        lr = 0.5 / (1 + iteration * 0.01)
        
        # Gradient step
        z = z - lr * grad_z
        
        # Keep magnitudes approximately constant (Wirtinger Flow style)
        # Rescale to maintain sum(|z|^2) = N
        z_mag = np.abs(z)
        if np.sum(z_mag**2) > 1e-10:
            z = z * np.sqrt(N / np.sum(z_mag**2))
    
    # Extract h from best_z
    h = np.abs(best_z)**2
    h = h * (N / (np.sum(h) + 1e-10))
    
    return list(h)
# EVOLVE-BLOCK-END
