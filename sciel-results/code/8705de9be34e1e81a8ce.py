"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Radial Basis Function (RBF) network.
    This approach uses learnable centers, widths, and amplitudes for Gaussian kernels,
    allowing the model to adapt its shape locally to the data, unlike a fixed linear basis.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Parameter mapping:
    # p[0]: Offset
    # p[1], p[2], p[3]: Amplitude 1, Center 1, Log-Width 1
    # p[4], p[5], p[6]: Amplitude 2, Center 2, Log-Width 2
    # p[7], p[8], p[9]: Amplitude 3, Center 3, Log-Width 3
    
    offset = p[0]
    
    # Unpack and process parameters
    # Using exp for width ensures positive sigma, which keeps the Gaussian well-defined.
    
    w1, c1, log_s1 = p[1], p[2], p[3]
    w2, c2, log_s2 = p[4], p[5], p[6]
    w3, c3, log_s3 = p[7], p[8], p[9]
    
    s1 = np.exp(log_s1)
    s2 = np.exp(log_s2)
    s3 = np.exp(log_s3)
    
    # Calculate Gaussian terms
    # The exponent is always <= 0, so np.exp is safe (range (0, 1]).
    # We compute (x - c)^2 / (2 * s^2)
    
    # To prevent potential division by zero if s is extremely small (though exp(log_s) > 0 always),
    # and to handle large values, we can add a small epsilon to the denominator or just rely on float64.
    # float64 exp(log_s) will underflow to 0 if log_s is very negative.
    # If s=0, we have division by zero.
    # np.exp(-1000) is 0.0.
    # If s1 is 0.0, s1**2 is 0.0. Division by 0.
    # We should guard against this.
    
    # A safe way:
    # term = w * exp(-0.5 * ((x-c)/s)**2)
    # If s is very small, (x-c)/s is huge, squared is huge, exp(-huge) is 0.
    # But if x==c, (x-c)/s is 0/0? No, 0/huge is 0.
    # If s underflows to 0, and x!=c, we get inf. exp(-inf) is 0.
    # If s underflows to 0, and x==c, we get 0/0 -> nan.
    
    # To be absolutely safe, we can clip the log_sigma to a minimum value, e.g., -100.
    # exp(-100) is ~ 3.7e-44.
    # (x-c)^2 is at most ~ 36.
    # 36 / (2 * 3.7e-44) ~ 5e44.
    # exp(-5e44) is 0.
    # So even with very small sigmas, it's safe unless x==c exactly.
    # Given x is float64 from uniform distribution, exact equality is rare but possible?
    # Uniform random floats are distinct with high probability.
    
    # Let's add a small epsilon to the denominator to be safe.
    eps = 1e-12
    
    s1_sq = s1 * s1 + eps
    s2_sq = s2 * s2 + eps
    s3_sq = s3 * s3 + eps
    
    term1 = w1 * np.exp(-0.5 * ((x - c1) ** 2) / s1_sq)
    term2 = w2 * np.exp(-0.5 * ((x - c2) ** 2) / s2_sq)
    term3 = w3 * np.exp(-0.5 * ((x - c3) ** 2) / s3_sq)
    
    return offset + term1 + term2 + term3
# EVOLVE-BLOCK-END
