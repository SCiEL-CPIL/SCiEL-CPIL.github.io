import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
       h0 = np.full(N, 0.5)
       
       def objective(h):
           g = 1.0 - h
           c = np.correlate(h, g, mode="full")
           return np.max(c) / N * 2.0

       # Constraints
       # Equality: sum(h) - 50 = 0
       cons = (
           {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0},
       )
       
       # Bounds: 0 <= h <= 1
       bounds = [(0.0, 1.0) for _ in range(N)]
       
       # Use SLSQP
       # SLSQP uses finite differences for gradients by default if not provided.
       # This might be slow. We can provide a numerical gradient?
       # Or use method='COBYLA' which is derivative-free and handles inequalities?
       # COBYLA is derivative-free and handles constraints.
       
       # Let's try COBYLA.
       res = minimize(objective, h0, method='COBYLA', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'rhobeg': 0.1})
       
       h_opt = res.x
       
       # Final check and cleanup
       if not np.all(np.isfinite(h_opt)):
           return np.full(N, 0.5)
           
       # Ensure sum is exactly 50 and bounds are respected
       # COBYLA might not be perfect.
       h_opt = np.clip(h_opt, 0.0, 1.0)
       diff = N/2.0 - np.sum(h_opt)
       if abs(diff) > 1e-9:
           # Distribute diff
           # Add diff to the element with the most "room" towards the center
           # Simple: add to h[0] if possible, else distribute
           # To be safe, let's do a small adjustment loop
           # Or just return the clipped version, the error might be small enough?
           # The task says "sum(h) equal to len(h) / 2 within 1e-9 * len(h)".
           # 1e-9 * 100 = 1e-7.
           # If diff is larger, we need to fix it.
           
           # Let's try to adjust h[0]
           # If h[0] + diff is in [0,1], do it.
           if 0 <= h_opt[0] + diff <= 1:
               h_opt[0] += diff
           else:
               # Distribute evenly?
               # This might push some out of bounds.
               # Let's try to distribute to indices with h close to 0.5
               idxs = np.argsort(np.abs(h_opt - 0.5))
               for idx in idxs:
                   if abs(diff) < 1e-12: break
                   step = diff / (len(idxs) - idx)
                   if 0 <= h_opt[idx] + step <= 1:
                       h_opt[idx] += step
                       diff -= step
                   else:
                       # Can't take this step, try smaller?
                       # This is getting complex.
                       pass
       
       return h_opt
