import time
from fractions import Fraction
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Compute LCM(2, 3, ..., 48)
    L = 1
    for d in denominators:
        L = L // gcd(L, d) * d

    # Integer weights: w_i = L / d_i
    w = [L // d for d in denominators]

    # Try MILP approach using scipy
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        if time.time() > deadline - 30:
            raise TimeoutError("Not enough time for MILP")

        # Objective: maximize sum(w_i * x_i) subject to sum(w_i * x_i) <= L - 1
        # So minimize -sum(w_i * x_i)
        c = np.array([-float(wi) for wi in w])

        # Constraint: sum(w_i * x_i) <= L - 1
        A_ub = np.array([float(wi) for wi in w])

        constraints = [LinearConstraint(A_ub.reshape(1, -1), -np.inf, float(L - 1))]

        # Binary variables
        integrality = np.ones(n)
        bounds = Bounds(0, 1)

        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 55, "mip_gap": 0.0}
        )

        if result.status == 0 and result.x is not None:
            selected = []
            for i in range(n):
                if result.x[i] > 0.5:
                    selected.append(i)

            if len(selected) > 0:
                # Verify with exact Fraction arithmetic
                total_frac = Fraction(0)
                for idx in selected:
                    total_frac += Fraction(1, denominators[idx])

                if total_frac < 1:
                    return selected
                else:
                    # If verification fails (floating point issue), try removing one element
                    # to get strictly below 1
                    for remove_idx in range(len(selected)):
                        trial = selected[:remove_idx] + selected[remove_idx+1:]
                        total_frac = Fraction(0)
                        for idx in trial:
                            total_frac += Fraction(1, denominators[idx])
                        if total_frac < 1 and len(trial) > 0:
                            return trial
                    pass

        # If MILP didn't give a valid result, fall through
    except Exception:
        pass

    # Fallback: greedy approach - take all that fit
    sel = []
    s = Fraction(0)
    for k in range(n):
        if s + Fraction(1, denominators[k]) < 1:
            sel.append(k)
            s += Fraction(1, denominators[k])
    return sel
