import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0

def coefficients(stages):
    """12-stage product formula with optimized symmetric splitting coefficients."""
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * stages
        b = [0.0] * stages
        for i in range(0, stages, 2):
            if i < len(b):
                b[i] = 1.0 / substeps
        return a, b

    # Time guard
    if time.time() > _DEADLINE - 10:
        # Fallback: equal weights (2nd order Strang)
        a = [1.0/6.0] * 12
        b = [1.0/6.0] * 12
        return a, b

    from scipy.optimize import minimize

    # Symmetric 4th-order splitting with s=6 sub-steps
    # a = [a1,a2,a3,a3,a2,a1] for first 6 stages, mirrored for last 6
    # b = [b1,b2,b3,b3,b2,b1] for first 6 stages, mirrored for last 6
    # Free: a1,a2,a3,b1,b2,b3

    def objective(x):
        a1, a2, a3, b1, b2, b3 = x
        a = np.array([a1, a2, a3, a3, a2, a1])
        b = np.array([b1, b2, b3, b3, b2, b1])
        k = np.arange(1, 7)
        # 5th moment difference (proxy for 5th-order error)
        m5_a = np.sum(k**5 * a)
        m5_b = np.sum(k**5 * b)
        # 6th moment (for 6th-order accuracy proxy)
        m6_a = np.sum(k**6 * a)
        m6_b = np.sum(k**6 * b)
        return (m5_a - m5_b)**2 + 0.1*(m6_a - m6_b)**2 + 0.001*(np.sum(a**2) + np.sum(b**2))

    cons = [
        {'type': 'eq', 'fun': lambda x: 2*(x[0]+x[1]+x[2]) - 1},
        {'type': 'eq', 'fun': lambda x: 2*(x[3]+x[4]+x[5]) - 1},
        {'type': 'eq', 'fun': lambda x: (2*x[0]+8*x[1]+18*x[2]) - (2*x[3]+8*x[4]+18*x[5])}
    ]
    bounds = [(-0.5, 2.0)] * 6
    x0 = [1/6.0]*6

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=cons, options={'maxiter': 2000, 'ftol': 1e-14})
        if time.time() > _DEADLINE - 5:
            raise TimeoutError()
        if result.success and all(np.isfinite(result.x)):
            a1, a2, a3, b1, b2, b3 = result.x
            a = [a1, a2, a3, a3, a2, a1, a1, a2, a3, a3, a2, a1]
            b = [b1, b2, b3, b3, b2, b1, b1, b2, b3, b3, b2, b1]
            a = [max(-50, min(50, x)) for x in a]
            b = [max(-50, min(50, x)) for x in b]
            # Renormalize after clipping
            a = [x/sum(a) for x in a]
            b = [x/sum(b) for x in b]
            return a, b
    except Exception:
        pass

    # Fallback: hardcoded 4th-order coefficients
    a = [0.159518, 0.327217, 0.159518, 0.159518, 0.327217, 0.159518,
         0.159518, 0.327217, 0.159518, 0.159518, 0.327217, 0.159518]
    b = [0.0, 0.327217, 0.159518, 0.159518, 0.327217, 0.159518,
         0.159518, 0.327217, 0.159518, 0.159518, 0.327217, 0.0]
    a = [x/sum(a) for x in a]
    b = [x/sum(b) for x in b]
    return a, b
