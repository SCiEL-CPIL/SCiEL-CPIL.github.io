import numpy as np

def heuristics(prizes, weights):
       # prizes: (5, 100)
       # weights: (5, 100)
       eps = 1e-9
       ratios = prizes / (weights + eps)
       
       # Soft-Min approximation
       beta = 5.0
       
       # v = -beta * ratios
       v = -beta * ratios
       
       # Numerical stability: subtract max
       v_max = np.max(v, axis=0, keepdims=True)
       
       exp_v = np.exp(v - v_max)
       sum_exp = np.sum(exp_v, axis=0)
       
       # soft_min = -1/beta * (v_max + log(sum_exp))
       soft_min = -1.0 / beta * (v_max + np.log(sum_exp))
       
       # The task handles non-positive values, but we return the raw values.
       # Ensure output shape is (100,)
       return soft_min.flatten()
