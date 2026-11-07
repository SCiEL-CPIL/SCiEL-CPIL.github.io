import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixture of 3 2D Gaussians + Bias.
    Parameters:
    0, 1, 2: tanh(params) -> centers x1, x2, x3
    3, 4, 5: tanh(params) -> centers y1, y2, y3
    6, 7, 8: weights w1, w2, w3
    9: exp(-param) -> width sigma
    """
    # Unpack parameters
    cx1 = np.tanh(params[0])
    cx2 = np.tanh(params[1])
    cx3 = np.tanh(params[2])
    
    cy1 = np.tanh(params[3])
    cy2 = np.tanh(params[4])
    cy3 = np.tanh(params[5])
    
    w1 = params[6]
    w2 = params[7]
    w3 = params[8]
    
    sigma = np.exp(-params[9])
    # Ensure sigma is not too small to avoid numerical issues, but exp is always > 0
    # If params[9] is large positive, sigma is small. If large negative, sigma is large.
    # Starting at 1, sigma = exp(-1) ~ 0.36.
    
    # Calculate squared distances
    # G1
    d1 = (x - cx1)**2 + (y - cy1)**2
    g1 = np.exp(-d1 / (2 * sigma**2))
    
    # G2
    d2 = (x - cx2)**2 + (y - cy2)**2
    g2 = np.exp(-d2 / (2 * sigma**2))
    
    # G3
    d3 = (x - cx3)**2 + (y - cy3)**2
    g3 = np.exp(-d3 / (2 * sigma**2))
    
    pred = w1 * g1 + w2 * g2 + w3 * g3
    
    return pred
