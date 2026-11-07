# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 4-unit tanh MLP with 10 parameters:
    # p0, p1: weight, bias for unit 0
    # p2, p3: weight, bias for unit 1
    # p4, p5: weight, bias for unit 2
    # p6, p7: weight, bias for unit 3
    # p8, p9: output weights for units 0, 1
    # Wait, that's only 8. Let me re-plan to use exactly 10 params.
    # 
    # Let's do 3 hidden units with a linear combination + bias:
    # p0, p1: weight, bias for unit 0
    # p2, p3: weight, bias for unit 1
    # p4, p5: weight, bias for unit 2
    # p6, p7, p8: output weights for units 0, 1, 2
    # p9: output bias
    #
    # This gives a model: f(x) = p6*tanh(p0*x+p1) + p7*tanh(p2*x+p3) + p8*tanh(p4*x+p5) + p9
    #
    # This is a universal approximator for continuous functions on bounded intervals.
    # The BFGS optimizer starting from all ones should be able to find good parameters
    # for smooth functions.
    
    z0 = np.tanh(params[0] * x + params[1])
    z1 = np.tanh(params[2] * x + params[3])
    z2 = np.tanh(params[4] * x + params[5])
    
    return params[6] * z0 + params[7] * z1 + params[8] * z2 + params[9]
# EVOLVE-BLOCK-END
