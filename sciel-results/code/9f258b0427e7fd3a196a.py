import numpy as np
from scipy.optimize import minimize

# Try to find rank-10 decomposition of [2,2,3]
# C[i,k] = sum_j A[i,j] B[j,k]
# Need: sum_t u_t[i,j] * v_t[j,k] = w_t[i,k] (with appropriate structure)
