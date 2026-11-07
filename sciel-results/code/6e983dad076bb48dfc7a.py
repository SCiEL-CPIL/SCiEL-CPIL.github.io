"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Shallow Neural Network (1 hidden unit)
    Instead of a linear combination of fixed basis functions, this model uses
    a non-linear activation (tanh) to create a learned feature 'h'.
    
    Architecture:
    Inputs: t, x, v
    Scaled Inputs:
      ts = t - 40.0   (centered)
      xs = x * 5.0    (scaled to ~O(1))
      vs = v * 3.0    (scaled to ~O(1))
    
    Hidden Unit:
      h = tanh(p[0]*xs + p[1]*vs + p[2]*ts + p[3])
    
    Output Layer (Linear combination of h and direct inputs):
      y = p[4]*h + p[5]*xs + p[6]*vs + p[7]*(xs*vs) + p[8]*(xs*xs) + p[9]
    
    Parameters (10 total):
      p[0]: weight x -> hidden
      p[1]: weight v -> hidden
      p[2]: weight t -> hidden
      p[3]: bias hidden
      p[4]: weight h -> output
      p[5]: weight x -> output
      p[6]: weight v -> output
      p[7]: weight xv -> output
      p[8]: weight x^2 -> output
      p[9]: bias output
    """
    p = params
    
    # Scale inputs for better numerical conditioning of the tanh and BFGS
    ts = t - 40.0
    xs = x * 5.0
    vs = v * 3.0
    
    # Hidden layer
    z_h = p[0] * xs + p[1] * vs + p[2] * ts + p[3]
    h = np.tanh(z_h)
    
    # Output layer
    # We include direct linear terms and some simple non-linear terms in the output
    # to allow the model to capture simple physics even if the hidden unit is weak
    y = (p[4] * h + 
         p[5] * xs + 
         p[6] * vs + 
         p[7] * (xs * vs) + 
         p[8] * (xs * xs) + 
         p[9])
         
    return y

# EVOLVE-BLOCK-END
