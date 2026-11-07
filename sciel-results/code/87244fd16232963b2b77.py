"""
Strategy: Fixed-architecture Neural Network.
We use a small MLP (Multi-Layer Perceptron) with 2 hidden layers.
Input: 1 (x)
Hidden Layer 1: 10 units, tanh activation
Hidden Layer 2: 10 units, tanh activation
Output Layer: 1 unit, linear activation

Total parameters:
- W1: 10x1 (10 params)
- b1: 10 (fixed to 0? No, we only have 10 params total for the whole model? 
  Wait, the task says params is an array of length 10.
  If I use a 10-10-1 network, I need 10 + 10 + 10 + 10 = 40 params.
  I only have 10 params.
  
  Let's re-read carefully: "The evaluator fits params, an array of length 10".
  So I MUST fit exactly 10 parameters.
  
  Option A: 1 hidden layer, 10 units.
  Input -> 10 units -> Output.
  Params: W (10x1) + b (10) = 20 params. Still too many.
  
  Option B: 1 hidden layer, 5 units.
  Input -> 5 units -> Output.
  Params: W (5x1) + b (5) = 10 params.
  This works!
  
  Architecture:
  z = tanh(x * W + b)
  y = z @ v  # v is fixed? Or is v part of the 10 params?
  
  If y = sum_{i=1}^5 w_i * tanh(x * a_i + b_i), then params are [a_1, b_1, a_2, b_2, ..., a_5, b_5].
  This is a sum of 5 sigmoid/tanh basis functions with learnable centers and scales.
  This is much more flexible than a fixed basis because the "basis functions" themselves adapt to the data.
  
  Let's refine this:
  y = sum_{i=1}^5 c_i * tanh(x * s_i + o_i)
  Here we have 3 params per unit: c_i (scale/weight), s_i (slope), o_i (offset).
  Total 15 params. Too many.
  
  We have exactly 10 params.
  
  Let's try:
  y = sum_{i=1}^5 c_i * tanh(x * s_i)
  Params: c_1, s_1, c_2, s_2, c_3, s_3, c_4, s_4, c_5, s_5.
  Total 10 params.
  This allows us to fit 5 "bumps" or "saturations" with learnable widths and heights.
  tanh(x*s) is an odd function. It can approximate linear, step-like, or smooth transitions.
  A sum of these can approximate many functions.
  
  Alternative:
  y = sum_{i=1}^5 c_i * exp(-x^2 * s_i)
  Params: c_1, s_1, ..., c_5, s_5.
  This is a sum of 5 Gaussians. Excellent for local features.
  But exp(-x^2 * s) requires s > 0 for stability. BFGS might find negative s, causing explosion.
  We can use exp(-x^2 * |s_i|) or exp(-x^2 * (s_i^2)).
  Using s_i^2 ensures positivity.
  Params: c_i, s_i. 10 params total.
  y = sum_{i=1}^5 c_i * exp(-x^2 * s_i^2)
  
  Let's consider the domain [-3, 3].
  If s_i is small, the Gaussian is wide. If s_i is large, it's narrow.
  This is a RBF (Radial Basis Function) network with centers fixed at 0.
  This might be too restrictive if the target has features away from 0.
  
  Better: Learnable centers.
  y = sum_{i=1}^5 c_i * exp(-(x - m_i)^2 * s_i^2)
  Params: c_1, m_1, s_1, c_2, m_2, s_2, c_3, m_3, s_3, c_4, m_4, s_4...
  That's 15 params.
  
  We have 10 params.
  Can we do 3 RBFs?
  y = sum_{i=1}^3 c_i * exp(-(x - m_i)^2 * s_i^2)
  Params: 3 * 3 = 9. Plus 1 global bias?
  y = b + sum_{i=1}^3 c_i * exp(-(x - m_i)^2 * s_i^2)
  Total 10 params.
  This is a very powerful local basis. It can fit any continuous function reasonably well if it's smooth, by placing 3 bumps.
  
  However, 3 bumps might not be enough for oscillatory functions like sin(3x).
  
  What about a hybrid?
  1 polynomial term + 2 RBFs?
  y = a*x^2 + b + c1*exp(-(x-m1)^2*s1^2) + c2*exp(-(x-m2)^2*s2^2)
  Params: a, b, c1, m1, s1, c2, m2, s2. (8 params).
  Remaining 2?
  Maybe add a linear term?
  y = a*x + b + c1*exp(...) + c2*exp(...)
  Params: a, b, c1, m1, s1, c2, m2, s2. (8 params).
  
  Let's stick to the 3 RBF + bias idea, it's robust.
  Or, even better:
  Use a mixture of different basis types to cover more ground?
  No, the params are fixed length 10.
  
  Let's try a different angle.
  What if we use a single hidden layer with 10 units, but we fix the output weights to 1?
  y = sum_{i=1}^10 tanh(x * s_i + b_i)
  Params: s_1, b_1, ..., s_10, b_10.
  Total 20 params. Still too many.
  
  Wait, can we share parameters?
  No.
  
  Let's go with the **5-tanh** model:
  y = sum_{i=1}^5 c_i * tanh(x * s_i + b_i)
  This requires 15 params.
  
  How about:
  y = sum_{i=1}^5 c_i * tanh(x * s_i)
  10 params.
  This lacks offsets. tanh(x*s) is always 0 at x=0.
  The sum will be 0 at x=0.
  If the target function is not 0 at x=0, this is a severe constraint.
  We need a bias.
  
  Can we reserve 1 param for bias?
  y = b + sum_{i=1}^4 c_i * tanh(x * s_i)
  Params: b, c_1, s_1, c_2, s_2, c_3, s_3, c_4, s_4.
  Total 9 params.
  We have 1 left.
  Maybe make it 4.5 units? No.
  
  Let's try:
  y = b + sum_{i=1}^4 c_i * tanh(x * s_i + o_i)
  Params: b, c_1, s_1, o_1, c_2, s_2, o_2, c_3, s_3, o_3...
  1 + 4*3 = 13. Too many.
  
  Let's try:
  y = b + sum_{i=1}^3 c_i * tanh(x * s_i + o_i)
  Params: b, c_1, s_1, o_1, c_2, s_2, o_2, c_3, s_3, o_3.
  Total 10 params.
  
  This model:
  - Has a global bias.
  - Has 3 learnable "S-curves" (tanh units) with learnable center, scale, and amplitude.
  - This is a universal approximator (in the limit of more units, but 3 is small).
  - It can approximate polynomials, steps, and smooth curves.
  - It is numerically stable (tanh is bounded).
  - BFGS should be able to fit this.
  
  Let's verify the parameter count:
  1 (bias) + 3 * 3 (c, s, o) = 10. Correct.
  
  Implementation:
  params[0] = b
  params[1:4] = c1, s1, o1
  params[4:7] = c2, s2, o2
  params[7:10] = c3, s3, o3
  
  y = b + c1*tanh(x*s1 + o1) + c2*tanh(x*s2 + o2) + c3*tanh(x*s3 + o3)
  
  This is a very different strategy from the fixed Chebyshev basis. It adapts the basis functions to the data.
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a sum of 3 shifted/scaled tanh functions plus a bias.
    params: array of length 10.
    params[0]: bias
    params[1:4]: c1, s1, o1
    params[4:7]: c2, s2, o2
    params[7:10]: c3, s3, o3
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    b = params[0]
    
    c1, s1, o1 = params[1], params[2], params[3]
    c2, s2, o2 = params[4], params[5], params[6]
    c3, s3, o3 = params[7], params[8], params[9]
    
    # Compute terms
    # Use np.tanh for stability
    term1 = c1 * np.tanh(x * s1 + o1)
    term2 = c2 * np.tanh(x * s2 + o2)
    term3 = c3 * np.tanh(x * s3 + o3)
    
    y = b + term1 + term2 + term3
    
    return y
