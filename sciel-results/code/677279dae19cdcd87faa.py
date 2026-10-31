import numpy as np
from scipy.optimize import differential_evolution

N = 600
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

def _gaussian_mixture_heights(params, n_basis=6):
    n = n_basis
    heights = np.zeros(N)
    for i in range(n):
        c = params[3 * i]
        w = params[3 * i + 1]
        a = params[3 * i + 2]
        heights += a * np.exp(-(( _centers - c) ** 2) / (2.0 * w ** 2))
    return np.maximum(heights, 0.0)

def _compute_score_analytic(params, n_basis=6):
    """
    Compute score using analytic convolution of Gaussian components.
    params: [c1, w1, a1, ..., cN, wN, aN]
    """
    # Extract params
    cs = params[0::3]
    ws = params[1::3]
    as_ = params[2::3]
    
    # Ensure widths are positive
    ws = np.maximum(ws, 1e-6)
    
    # Calculate L2 norm squared
    # Integral of product of two Gaussians
    # S_ij = sqrt(2*pi*w_i^2*w_j^2 / (w_i^2+w_j^2)) * exp(-(c_i-c_j)^2 / (2*(w_i^2+w_j^2)))
    w_sq = ws ** 2
    # Matrix of w_i^2 + w_j^2
    w_sum_sq = w_sq[:, None] + w_sq[None, :]
    # Pre-factor for amplitude
    prefactor = np.sqrt(2 * np.pi * w_sq[:, None] * w_sq[None, :] / w_sum_sq)
    # Exponent part
    c_diff = cs[:, None] - cs[None, :]
    exponent = - (c_diff ** 2) / (2 * w_sum_sq)
    S = prefactor * np.exp(exponent)
    
    l2_sq = np.dot(as_, np.dot(S, as_))
    
    if l2_sq <= 1e-15:
        return 1e10
        
    # Calculate Autoconvolution
    # (f*f)(t) = sum_{i,j} a_i a_j C_ij(t)
    # C_ij(t) is Gaussian centered at c_i + c_j, width w_ij = sqrt(w_i^2 + w_j^2)
    # Amplitude A_ij = a_i a_j * sqrt(2*pi*w_i^2*w_j^2 / (w_i^2+w_j^2))
    # Note: The integral of exp(-x^2/2w^2) is w*sqrt(2*pi).
    # The convolution of two such functions has peak height:
    # Integral of product at shift 0 is S_ij.
    # The function C_ij(t) is S_ij * exp(-(t - (c_i+c_j))^2 / (2*w_ij^2)) / (w_ij * sqrt(2*pi)) ?
    # Let's re-derive carefully.
    # f(x) = a * exp(-(x-c)^2 / 2w^2).
    # (f*f)(t) = a^2 * Integral exp(-(x-c1)^2/2w1^2) exp(-(x-t-c2)^2/2w2^2) dx.
    # Let u = x - c1. x = u + c1. x - t - c2 = u - (t - c1 + c2).
    # Integral = exp(-(t - c1 + c2)^2 / 2(w1^2+w2^2)) * Integral exp(-u^2/2w1^2 - (u-d)^2/2w2^2) du where d = t - c1 + c2? No.
    # Standard result: Product of two Gaussians is a Gaussian.
    # The integral is sqrt(2*pi*w1^2*w2^2/(w1^2+w2^2)) * exp(-(t - c1 + c2)^2 / 2(w1^2+w2^2)).
    # So C_ij(t) = a_i a_j * S_ij * exp(-(t - (c_i - c_j))^2 / 2(w_i^2+w_j^2)).
    # Wait, the center is c_i - c_j?
    # If f1 is at c1 and f2 is at c2.
    # (f1 * f2)(t) peaks at t = c1 + c2.
    # Let's check the formula again.
    # exp(-(x-c1)^2/2w1^2) * exp(-(x-t-c2)^2/2w2^2)
    # = exp(-1/2 [ (x-c1)^2/w1^2 + (x-t-c2)^2/w2^2 ])
    # Let y = x - c1. Then x - t - c2 = y + c1 - t - c2 = y - (t - c1 + c2).
    # Integral over y.
    # The exponent is -1/2 [ y^2/w1^2 + (y - D)^2/w2^2 ] where D = t - c1 + c2.
    # Minimum is at y = D * w1^2 / (w1^2+w2^2).
    # The value of the integral is exp(-D^2 / 2(w1^2+w2^2)) * sqrt(2*pi*w1^2*w2^2/(w1^2+w2^2)).
    # So the term is proportional to exp(-(t - c1 + c2)^2 / 2(w1^2+w2^2)).
    # The center is t = c1 - c2.
    # This is cross-correlation.
    # But autoconvolution is (f*f)(t) = integral f(x) f(t-x) dx.
    # If we use f(t-x), then the second center is t-x.
    # If f2 is centered at c2, then f2(t-x) is centered at t-x = c2 => x = t - c2.
    # So the overlap is best when x is close to c1 AND x is close to t - c2.
    # So c1 = t - c2 => t = c1 + c2.
    # So the center is c1 + c2.
    # The formula derived above used f(x-t), which is cross-correlation.
    # The task says "autoconvolution".
    # Usually in signal processing, autoconvolution means f*f.
    # Let's assume f*f.
    # Center = c_i + c_j.
    
    c_sum = cs[:, None] + cs[None, :]
    w_sum = np.sqrt(w_sq[:, None] + w_sq[None, :])
    A = as_[:, None] * as_[None, :] * S # S is the integral value at peak?
    # S calculated above is the integral of the product at shift 0?
    # S_ij = sqrt(...) * exp(-(c_i-c_j)^2 / ...).
    # This S_ij is the value of the integral \int f_i(x) f_j(x) dx.
    # This is the peak of the cross-correlation.
    # For convolution f_i * f_j, the peak is at c_i+c_j and has the same value \int f_i(x) f_j(-x) dx?
    # If f_j is symmetric, \int f_i(x) f_j(x) dx = \int f_i(x) f_j(-x) dx.
    # If we use symmetric basis, f_j is symmetric? No, the basis functions are Gaussians, which are symmetric.
    # So f_i(x) is symmetric around c_i. f_j(x) is symmetric around c_j.
    # \int f_i(x) f_j(c_i+c_j - x) dx?
    # Let's just use the property that the peak of f_i * f_j is \int f_i(x) f_j(-x) dx shifted?
    # Actually, for Gaussians, the peak value of the convolution is the same as the peak of the cross-correlation if the functions are symmetric?
    # No.
    # Let's compute the peak value directly.
    # Peak of f_i * f_j is at t = c_i + c_j.
    # Value = \int f_i(x) f_i(c_i+c_j - x) dx? No, f_j(c_i+c_j - x).
    # Since f_j is Gaussian centered at c_j, f_j(c_i+c_j - x) = exp(-(c_i+c_j-x - c_j)^2/2w_j^2) = exp(-(c_i-x)^2/2w_j^2) = exp(-(x-c_i)^2/2w_j^2).
    # So the integral is \int exp(-(x-c_i)^2/2w_i^2) exp(-(x-c_i)^2/2w_j^2) dx.
    # This is exactly the S_ij calculated above (with c_i-c_j = 0? No, the centers in the exponent are both c_i).
    # In my S_ij formula, I had exp(-(c_i-c_j)^2 / ...).
    # Here the centers are effectively the same (c_i).
    # So the peak value is S_ij with c_diff = 0?
    # No, S_ij depends on c_i and c_j.
    # The integral \int exp(-(x-c_i)^2/2w_i^2) exp(-(x-c_i)^2/2w_j^2) dx
    # = sqrt(2*pi*w_i^2*w_j^2/(w_i^2+w_j^2)).
    # This does NOT depend on c_i or c_j!
    # So the peak of the convolution of two Gaussians is independent of their centers, only their widths and amplitudes.
    # Peak_ij = a_i a_j * sqrt(2*pi*w_i^2*w_j^2/(w_i^2+w_j^2)).
    
    A_peak = as_[:, None] * as_[None, :] * np.sqrt(2 * np.pi * w_sq[:, None] * w_sq[None, :] / w_sum_sq)
    
    # The convolution is a sum of Gaussians:
    # (f*f)(t) = sum_{i,j} A_peak[i,j] * exp(-(t - c_sum[i,j])^2 / (2 * w_sum[i,j]^2))
    
    # We need the max of this sum over t in [-0.5, 0.5].
    # Since it's a sum of positive Gaussians, the max is likely near one of the c_sum.
    # We can evaluate on a grid.
    t_grid = np.linspace(-0.5, 0.5, 200)
    conv_val = np.zeros_like(t_grid)
    for i in range(n_basis):
        for j in range(n_basis):
            conv_val += A_peak[i, j] * np.exp(-(t_grid - c_sum[i, j]) ** 2 / (2 * w_sum[i, j] ** 2))
            
    max_conv = np.max(conv_val)
    
    return max_conv / l2_sq

def construct():
    n_basis = 6
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))
        bounds.append((0.01, 0.5))
        bounds.append((0.0, 5.0))
        
    def objective(params):
        return _compute_score_analytic(params, n_basis)
        
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,
        tol=1e-6,
        popsize=15,
        init='sobol',
        polish=True,
        updating='immediate',
        workers=1
    )
    
    best_params = result.x
    best_heights = _gaussian_mixture_heights(best_params, n_basis)
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
