"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] to match the domain of Legendre polynomials
    # x is in [-3, 3], so z = x / 3 is in [-1, 1]
    z = x / 3.0
    
    # Compute the first 10 Legendre polynomials P_0(z) ... P_9(z)
    # P_0 = 1
    # P_1 = z
    # P_{n+1} = ((2n+1)z P_n - n P_{n-1}) / (n+1)
    
    # We need to accumulate the sum: sum(params[i] * P_i(z))
    # Let's compute P_0 and P_1 first
    P_prev = np.ones_like(z)  # P_0
    P_curr = z                # P_1
    
    pred = params[0] * P_prev
    if len(params) > 1:
        pred += params[1] * P_curr
        
    for i in range(2, 10):
        # Recurrence: P_i(z) = ((2*(i-1)+1)*z*P_{i-1}(z) - (i-1)*P_{i-2}(z)) / i
        # Note: i is the index, so we are computing P_i from P_{i-1} and P_{i-2}
        # Let k = i-1. Then P_{k+1} = ((2k+1)z P_k - k P_{k-1}) / (k+1)
        # Here i corresponds to k+1. So k = i-1.
        # P_i = ((2(i-1)+1)z P_{i-1} - (i-1)P_{i-2}) / i
        P_next = ((2 * (i - 1) + 1) * z * P_curr - (i - 1) * P_prev) / i
        P_prev = P_curr
        P_curr = P_next
        pred += params[i] * P_curr
        
    return pred
# EVOLVE-BLOCK-END
