import numpy as np
import scipy.optimize

def construct(q, n, radius):
    N = q ** n
    indices = np.arange(N)
    digits = np.zeros((N, n), dtype=np.int8)
    for k in range(n):
        digits[:, k] = (indices // (q ** (n - 1 - k))) % q
        
    A = np.eye(N, dtype=np.int8)
    for p in range(n):
        step = q ** (n - 1 - p)
        current_digit = digits[:, p]
        for v in range(q):
            delta = (v - current_digit) * step
            new_indices = indices + delta
            A[indices, new_indices] = 1
            
    if hasattr(scipy.optimize, 'milp'):
        try:
            c = np.ones(N)
            constraints = [scipy.optimize.LinearConstraint(A, lb=1, ub=np.inf)]
            integrality = np.ones(N, dtype=int)
            bounds = scipy.optimize.Bounds(lb=0, ub=1)
            res = scipy.optimize.milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={'time_limit': 100})
            if res.status == 0:
                x_bin = (res.x > 0.5).astype(int)
                selected_indices = np.where(x_bin == 1)[0]
                return [digits[idx].tolist() for idx in selected_indices]
        except Exception:
            pass
            
    # Fallback: Greedy
    covered = np.zeros(N, dtype=bool)
    codewords = []
    import random
    random.seed(42)
    while not np.all(covered):
        # Find uncovered
        uncov = np.where(~covered)[0]
        if len(uncov) == 0: break
        # Pick random uncovered
        target = random.choice(uncov.tolist())
        # Find candidates covering target
        cands = np.where(A[target, :] == 1)[0]
        # Pick candidate that covers most uncovered words?
        # To be fast, just pick random
        chosen = random.choice(cands.tolist())
        codewords.append(chosen)
        covered[A[chosen, :] == 1] = True
        
    return [digits[idx].tolist() for idx in codewords]
