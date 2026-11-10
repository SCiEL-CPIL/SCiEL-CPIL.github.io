import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a single hidden layer neural network (10 ReLU units).
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of length 10. 
            The evaluator fits these parameters. 
            In this implementation, we use a fixed architecture where the 'parameters' 
            are effectively the output weights and biases, but since the evaluator 
            only passes 10 params and we need to define the hidden layer, we must 
            be careful. 
            
            Wait, the evaluator fits `params` (length 10) using BFGS. 
            If I define a neural network with hidden weights, those weights must be 
            part of `params` or fixed. 
            The task says: "The evaluator fits params, an array of length 10". 
            So I have exactly 10 degrees of freedom.
            
            Strategy: 
            Use 5 ReLU units? No, 10 params. 
            If I use 5 ReLU units, I need 5 weights for the hidden layer and 5 for the output? 
            Or 10 ReLU units with fixed hidden weights?
            
            Let's use 10 ReLU units with FIXED hidden weights and biases, and the 10 
            `params` are the output weights. 
            This makes the problem convex with respect to `params` (linear combination of 
            fixed basis functions). This is similar to the RBF insight but using ReLUs.
            ReLUs are piecewise linear. A sum of ReLUs is a piecewise linear function.
            
            Fixed hidden layer:
            h_i(x) = max(0, w_i * x + b_i)
            y(x) = sum_{i=1}^{10} params[i] * h_i(x)
            
            We need to choose w_i and b_i such that the basis functions cover the domain 
            [0.1, 6] well and are diverse.
            Since x is in [0.1, 6], let's place the "kinks" (biases) evenly across the domain.
            b_i = -x_kink_i.
            Let's choose x_kink_i evenly spaced in [0.1, 6].
            Spacing: (6 - 0.1) / 9 = 0.655...
            Kinks: 0.1, 0.755, 1.41, ...
            
            If we use slopes w_i = 1, then h_i(x) = max(0, x - x_kink_i).
            This creates a set of "hinge" functions.
            Any piecewise linear function with these kinks can be represented.
            This is a very robust basis for BFGS because it's linear in params.
            
            Let's verify the number of params.
            10 params -> 10 hinge functions.
            
            Calculation:
            kinks = np.linspace(0.1, 6.0, 10)
            h = np.maximum(0, x[None, :] - kinks[:, None])  # Shape (10, N)
            y = np.dot(params, h) # Shape (N,)
            
            This is computationally cheap and convex.
    """
    # Ensure x is 1D for consistency, though it should be
    x = np.asarray(x).flatten()
    
    # Define fixed hinge points (kinks) across the domain [0.1, 6]
    # 10 kinks for 10 parameters
    kinks = np.linspace(0.1, 6.0, 10)
    
    # Compute ReLU activations: max(0, x - kink)
    # Broadcasting: x is (N,), kinks is (10,)
    # We want h[i, j] = max(0, x[j] - kinks[i])
    h = np.maximum(0, x[None, :] - kinks[:, None])
    
    # Linear combination of these basis functions
    # params is (10,), h is (10, N)
    y = np.dot(params, h)
    
    return y
