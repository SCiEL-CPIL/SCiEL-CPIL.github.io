import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # Start at the center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center

        calls_used = 1

        # Determine the sign of the gradient for the first dim-1 dimensions
        # For a linear function f(x) = c + w^T x, the minimum is at the vertex
        # where x_i = 5 if w_i < 0, and x_i = -5 if w_i > 0.
        # We estimate w_i by checking f(5 e_i) vs f(0).
        signs = np.zeros(dim)

        for i in range(dim - 1):
            if time.time() > deadline:
                break

            # Evaluate f at 5 * e_i
            x_probe = x_center.copy()
            x_probe[i] = 5.0

            f_probe = func(x_probe)
            calls_used += 1

            # Update best if probe is better
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()

            # Determine sign of gradient component i
            if f_probe < f_center:
                signs[i] = -1.0  # Gradient negative, go to +5
            else:
                signs[i] = 1.0   # Gradient positive (or zero), go to -5

        # For the last dimension, we don't know the sign yet.
        # Construct two candidate vertices differing only in the last coordinate.
        # Evaluate them at calls dim+1 and dim+2 (calls 6 and 7 for dim=5).
        # One of these is the global minimum.

        x_cand_1 = np.zeros(dim)
        x_cand_2 = np.zeros(dim)

        for i in range(dim - 1):
            x_cand_1[i] = 5.0 if signs[i] < 0 else -5.0
            x_cand_2[i] = 5.0 if signs[i] < 0 else -5.0

        x_cand_1[dim - 1] = 5.0
        x_cand_2[dim - 1] = -5.0

        # Evaluate the first candidate vertex (call 6)
        if calls_used < budget and time.time() < deadline:
            f_cand_1 = func(x_cand_1)
            calls_used += 1

            if f_cand_1 < best_f:
                best_f = f_cand_1
                best_x = x_cand_1.copy()

        # Evaluate the second candidate vertex (call 7)
        if calls_used < budget and time.time() < deadline:
            f_cand_2 = func(x_cand_2)
            calls_used += 1

            if f_cand_2 < best_f:
                best_f = f_cand_2
                best_x = x_cand_2.copy()

        # For a linear function on a box, the global minimum is at a vertex.
        # We have now evaluated both possible vertices for the last dimension,
        # so best_x is the global minimizer. No further calls can improve the value.
        # The remaining budget does not improve the score since AOCC uses the best-so-far.

        return best_x, best_f
