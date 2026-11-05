import numpy as np

N = 400

def construct():
    # Analytical solution:
    # The objective function is V = 2*N*max(C)/S^2.
    # By Cauchy-Schwarz, max(C) >= C[0] = sum(h^2) >= S^2/N.
    # Thus V >= 2*N*(S^2/N)/S^2 = 2.
    # The equality holds when h is constant (all elements equal).
    # Therefore, the constant vector is the global minimizer.
    return np.ones(N)
