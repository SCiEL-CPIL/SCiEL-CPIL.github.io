"""
Strategy: Learnable Weighted Gaussian Mixture Model.
The current strategy uses 5 Gaussian RBFs with fixed coefficient 1.0.
This limits the model to additive positive bumps. By learning 5 weights (amplitudes),
the model can subtract regions, scale individual features, and better approximate
functions with varying local amplitudes or negative regions.
Parameters:
p[0:5] -> Centers (c_i)
p[5:9] -> Widths (w_i)
p[9]   -> Global Scale (optional, but here we use individual weights)
Wait, we have 10 params.
Let's use:
p[0:5] -> Centers (c_i)
p[5:9] -> Widths (w_i)
p[9]   -> Global Amplitude Scale? No, let's use 5 weights.
But that's 15 params. We only have 10.

Revised Parameter Allocation for 10 params:
We need to balance centers, widths, and weights.
Option A: 3 Centers, 3 Widths, 3 Weights, 1 Global Offset, 1 Global Scale? 
Option B: 5 Centers, 5 Widths. Weights are fixed? That's the current strategy.
Option C: 4 Centers, 4 Widths, 2 Weights? No.

Let's look at the constraint: params length is 10.
If we want weights, we must reduce the number of kernels.
Let's use 3 Gaussian Kernels.
Params:
p[0:3] -> Centers (c0, c1, c2)
p[3:6] -> Widths (w0, w1, w2)
p[6:9] -> Weights (a0, a1, a2)
p[9]   -> Global Offset (b)

This gives us a model: y(x) = b + sum_{i=0}^{2} a_i * exp(-0.5 * ((x-c_i)/w_i)^2)
This is more flexible than 5 fixed-unit Gaussians because:
1. We can have negative weights (subtraction).
2. We can have an offset (non-zero mean).
3. We can scale the height of each bump.
The reduction from 5 to 3 kernels is likely worth the gain in flexibility per kernel.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    x = np.asarray(x).ravel()
    n = len(x)
    
    # Unpack parameters
    # 3 Centers, 3 Widths, 3 Weights, 1 Offset
    centers = params[0:3]
    widths = params[3:6]
    weights = params[6:9]
    offset = params[9]
    
    # Ensure widths are strictly positive and bounded to avoid numerical issues
    # Clipping to [0.01, 10.0] ensures stability
    safe_widths = np.clip(widths, 0.01, 10.0)
    
    # Initialize output with offset
    result = np.full_like(x, offset, dtype=float)
    
    # Compute sum of 3 Weighted Gaussian RBFs
    for i in range(3):
        c = centers[i]
        w = safe_widths[i]
        a = weights[i]
        
        # Check deadline
        if time.time() > deadline:
            break
            
        diff = x - c
        exponent = -0.5 * (diff * diff) / (w * w)
        
        # Clip exponent to prevent underflow, though exp(-large) is safe
        # We clip to -700 to avoid 0.0 which might lose gradient info? 
        # Actually, for BFGS, 0.0 is fine if it's far away.
        # But if we want smooth gradients, we might want to be careful.
        # Standard exp is fine.
        
        result += a * np.exp(exponent)
        
    return result
# EVOLVE-BLOCK-END
