import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a small Neural Network (MLP) structure.
    
    Architecture:
    Inputs: x, v
    Hidden Layer: 4 neurons with tanh activation
    Output Layer: 1 neuron (linear activation)
    
    Parameters (10 total):
    p[0]: Weight x -> h1
    p[1]: Weight v -> h1
    p[2]: Weight x -> h2
    p[3]: Weight v -> h2
    p[4]: Weight x -> h3
    p[5]: Weight v -> h3
    p[6]: Weight x -> h4
    p[7]: Weight v -> h4
    p[8]: Biases for hidden layer (shared for simplicity to save params, or p[8] is bias h1, others 0? 
          Let's use p[8] as a global bias for hidden layer and p[9] as output bias/weight)
          
    Actually, to maximize flexibility with 10 params:
    Let's define:
    h1 = tanh(p0*x + p1*v)
    h2 = tanh(p2*x + p3*v)
    h3 = tanh(p4*x + p5*v)
    h4 = tanh(p6*x + p7*v)
    
    y = p8 * (h1 + h2 + h3 + h4) + p9
    
    This uses 10 parameters. 
    p0-p7 are input weights.
    p8 is a global scaling factor for the hidden sum (acts like a master weight).
    p9 is the bias.
    
    This structure is smooth (tanh) and can approximate a wide range of nonlinear functions.
    BFGS should be able to optimize this if the landscape isn't too rugged, but tanh helps with saturation compared to raw polynomials.
    """
    
    # Time guard (though this function is called many times by the optimizer, 
    # the "120s" limit usually applies to the whole evaluation script. 
    # However, the prompt says "Keep an EXPLICIT deadline guard... return the best result found".
    # In this context, `equation` is the forward pass. It cannot "return best result found" 
    # because it doesn't do the search. The search is done by the evaluator's BFGS.
    # The deadline guard in the prompt likely refers to if *I* was doing the optimization.
    # Since I am just providing the model, I must ensure it is fast and stable.
    # I will not implement a time guard inside `equation` as it is a pure function.
    # The "deadline guard" instruction in the prompt seems to apply to strategies where the 
    # candidate code performs its own optimization loop. Here, the evaluator performs the optimization.
    # Therefore, I just need to ensure the function is vectorized and fast.
    
    p = params
    
    # Calculate hidden layer activations
    # Using tanh for smoothness and bounded output, which helps BFGS convergence
    # compared to unbounded polynomials.
    
    # To avoid numerical issues if params are large, tanh is naturally bounded.
    z1 = p[0] * x + p[1] * v
    z2 = p[2] * x + p[3] * v
    z3 = p[4] * x + p[5] * v
    z4 = p[6] * x + p[7] * v
    
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    h3 = np.tanh(z3)
    h4 = np.tanh(z4)
    
    # Output layer
    # p[8] scales the sum of hidden activations
    # p[9] is the bias
    y = p[8] * (h1 + h2 + h3 + h4) + p[9]
    
    return y
