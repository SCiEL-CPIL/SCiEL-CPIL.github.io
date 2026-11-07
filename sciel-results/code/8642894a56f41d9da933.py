import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters
    # Two tanh units with different orientations, plus linear x and constant
    w1, vx1, vy1, b1 = params[0], params[1], params[2], params[3]
    w2, vx2, vy2, b2 = params[4], params[5], params[6], params[7]
    lin_x, const = params[8], params[9]
    
    # Compute activations
    # Using tanh for bounded non-linearity
    unit1 = np.tanh(vx1 * x + vy1 * y + b1)
    unit2 = np.tanh(vx2 * x - vy2 * y + b2) # Note: minus sign for different orientation
    
    # Combine
    pred = w1 * unit1 + w2 * unit2 + lin_x * x + const
    
    return pred
