# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Hybrid Fourier-MLP model.
    - 4 parameters for a global Fourier basis (constant, sin(x), cos(x), sin(y), cos(y) - actually 5 terms, 
      but we can combine or select. Let's use: 1, sin(x), cos(x), sin(y), cos(y) -> 5 params.
      Wait, we have 10 params total.
      Let's split:
      5 params for Fourier global terms: [1, sin(x), cos(x), sin(y), cos(y)]
      5 params for a small MLP local/nonlinear term.
      
      Actually, the current best (1.49) is a pure MLP.
      Inspiration 1 (4.47) is a linear basis with some trig.
      Inspiration 3 (2.16) is Hybrid Poly+RBF.
      
      Let's try a Hybrid Fourier + MLP.
      The MLP part handles complex non-linear interactions.
      The Fourier part handles global periodic trends which might be hard for a small MLP to fit from scratch 
      if the frequencies are not well-matched, or simply provides a better initialization landscape.
      
      However, BFGS is local. A pure MLP might get stuck. 
      A linear combination of Fourier terms is convex (if we just fit coefficients), but the MLP part is non-convex.
      
      Let's structure it as:
      output = FourierPart + MLPPart
      
      Fourier Part (5 params):
      f0 = 1
      f1 = sin(x)
      f2 = cos(x)
      f3 = sin(y)
      f4 = cos(y)
      Four = p[0]*f0 + p[1]*f1 + p[2]*f2 + p[3]*f3 + p[4]*f4
      
      MLP Part (5 params):
      We need a simple non-linear term. 
      Maybe a single hidden neuron with tanh, and a bias?
      h = tanh(w_x*x + w_y*y + b)
      out_mlp = w_out * h
      
      Total params: 5 (Fourier) + 5 (MLP: w_x, w_y, b, w_out, scale_mlp?) 
      Actually, let's just use 4 params for MLP to leave 6 for Fourier?
      Or stick to the 10 param limit strictly.
      
      Let's use:
      p[0:5] : Fourier coefficients [1, sin(x), cos(x), sin(y), cos(y)]
      p[5:9] : MLP weights [w_x, w_y, b] for tanh
      p[9]   : MLP output weight
      
      output = (p[0] + p[1]*sin(x) + p[2]*cos(x) + p[3]*sin(y) + p[4]*cos(y)) + p[9]*tanh(p[5]*x + p[6]*y + p[7])
      
      This combines the global periodic capability of Fourier with the local non-linear flexibility of a single-neuron MLP.
      This is structurally different from pure polynomial (Insp 1) or pure RBF (Insp 2) or pure MLP (Current).
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Global Fourier Basis
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    
    fourier_part = (p[0] + 
                    p[1] * sin_x + 
                    p[2] * cos_x + 
                    p[3] * sin_y + 
                    p[4] * cos_y)
    
    # Local Non-linear MLP Component (Single Hidden Neuron)
    # tanh(w_x * x + w_y * y + b) * w_out
    hidden = np.tanh(p[5] * x + p[6] * y + p[7])
    mlp_part = p[8] * hidden
    
    # Note: We have 10 params (0-9). 
    # I used 0-4 for Fourier (5 params).
    # I used 5,6,7,8 for MLP (4 params).
    # I have one param left: p[9].
    # Let's add a second interaction term or a quadratic Fourier term?
    # Or maybe scale the whole thing?
    # Let's add a second hidden neuron or a higher freq Fourier term.
    # Let's add sin(x+y) and cos(x-y) to the Fourier part?
    # That would make Fourier 7 params, MLP 3 params (w_x, w_y, b) and 1 output weight? 
    # No, MLP needs at least 4 params to be useful (3 input weights + 1 output).
    
    # Alternative split:
    # 4 Fourier params: [1, sin(x), sin(y), sin(x+y)]
    # 6 MLP params: [w_x1, w_y1, b1, w_x2, w_y2, b2] -> 2 hidden neurons?
    # Output weight?
    
    # Let's stick to a robust 5-5 split.
    # Fourier: 1, sin(x), cos(x), sin(y), cos(y)
    # MLP: tanh(w1x*x + w1y*y + b1) * w1_out + tanh(w2x*x + w2y*y + b2) * w2_out
    # That's 3+1 + 3+1 = 8 params for 2 neurons. Too many.
    
    # Let's go back to 5 Fourier + 5 MLP.
    # Fourier: 1, sin(x), cos(x), sin(y), cos(y)
    # MLP: Single neuron with tanh, but with a scale factor on the input?
    # h = tanh(s * (w_x*x + w_y*y + b))
    # out = w_out * h
    # Params: s, w_x, w_y, b, w_out -> 5 params.
    # This allows the optimizer to adjust the "steepness" of the tanh, which is helpful for fitting sharp transitions.
    
    s = p[5]
    w_x = p[6]
    w_y = p[7]
    b = p[8]
    w_out = p[9]
    
    hidden = np.tanh(s * (w_x * x + w_y * y + b))
    mlp_part = w_out * hidden
    
    return fourier_part + mlp_part
# EVOLVE-BLOCK-END
