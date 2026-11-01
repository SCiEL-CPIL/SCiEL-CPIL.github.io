import numpy as np
from scipy.optimize import minimize
import time

N = 21
NX = 7
NY = 3

def construct():
       start_time = time.time()
       deadline = start_time + 100.0
       
       best_value = 0.0
       best_result = None
       
       # Try different aspect ratios
       # The optimal aspect ratio for uniform grid is where w/NX == h/NY
       # w/7 = h/3 => w = 7h/3. w+h=2 => 10h/3=2 => h=0.6, w=1.4
       # Let's search around this.
       
       def objective(z):
           # z: [w, x1, x2, x3, x4, x5, x6, y1, y2]
           # h is determined by w
           w = z[0]
           h = 2.0 - w
           if w < 0.2 or h < 0.2:
               return 1e6
               
           # Grid lines
           # x: 0, x1...x6, w
           # y: 0, y1, y2, h
           
           # Extract
           xs = np.array([0.0, z[1], z[2], z[3], z[4], z[5], z[6], w])
           ys = np.array([0.0, z[7], z[8], h])
           
           # Check ordering
           if np.any(np.diff(xs) < 0) or np.any(np.diff(ys) < 0):
               return 1e6
               
           # Cell widths and heights
           wc = np.diff(xs)
           hc = np.diff(ys)
           
           # Radii for each cell
           # r_ij = min(wc[i], hc[j]) / 2
           radii = np.array([
               [min(wc[i], hc[j]) / 2.0 for j in range(NY)] 
               for i in range(NX)
           ])
           
           total_r = np.sum(radii)
           return -total_r
           
       # Initial guess
       w0 = 1.4
       h0 = 0.6
       xs0 = np.linspace(0, w0, NX+1)[1:-1]
       ys0 = np.linspace(0, h0, NY+1)[1:-1]
       z0 = np.concatenate([[w0], xs0, ys0])
       
       bounds = [
           (0.2, 1.8), # w
       ]
       # x bounds depend on w, so we can't use simple bounds for SLSQP easily if we want strict ordering.
       # But SLSQP can handle inequalities.
       # Alternatively, we can parameterize the gaps.
       # Let's stick to SLSQP with bounds and let it handle ordering via penalty or just hope it stays ordered from a good start.
       # Better: use a transformation or just simple bounds and check validity.
       
       # Simple bounds for x_i: (0, 2)
       # Simple bounds for y_i: (0, 2)
       for _ in range(NX-1):
           bounds.append((0.01, 1.99))
       for _ in range(NY-1):
           bounds.append((0.01, 1.99))
           
       # We need to ensure x1 < x2 ... < x6
       # SLSQP doesn't enforce ordering directly in bounds.
       # We can add penalty for ordering violation in objective.
       
       def objective_penalized(z):
           w = z[0]
           h = 2.0 - w
           if w < 0.2 or h < 0.2:
               return 1e6
           xs = np.array([0.0, z[1], z[2], z[3], z[4], z[5], z[6], w])
           ys = np.array([0.0, z[7], z[8], h])
           
           # Penalty for non-positive or non-ordered
           if np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0):
               return 1e6 + np.sum(np.diff(xs)[np.diff(xs)<=0]**2) + np.sum(np.diff(ys)[np.diff(ys)<=0]**2)
               
           wc = np.diff(xs)
           hc = np.diff(ys)
           radii = np.array([
               [min(wc[i], hc[j]) / 2.0 for j in range(NY)] 
               for i in range(NX)
           ])
           return -np.sum(radii)
           
       try:
           res = minimize(objective_penalized, z0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10})
           if res.success or res.fun < objective_penalized(z0):
               z_opt = res.x
               w_opt = z_opt[0]
               h_opt = 2.0 - w_opt
               xs_opt = np.array([0.0, z_opt[1], z_opt[2], z_opt[3], z_opt[4], z_opt[5], z_opt[6], w_opt])
               ys_opt = np.array([0.0, z_opt[7], z_opt[8], h_opt])
               
               if np.all(np.diff(xs_opt) > 0) and np.all(np.diff(ys_opt) > 0):
                   wc = np.diff(xs_opt)
                   hc = np.diff(ys_opt)
                   
                   # Construct circles
                   x_c = np.zeros(N)
                   y_c = np.zeros(N)
                   r_c = np.zeros(N)
                   
                   idx = 0
                   for i in range(NX):
                       for j in range(NY):
                           x_c[idx] = (xs_opt[i] + xs_opt[i+1]) / 2.0
                           y_c[idx] = (ys_opt[j] + ys_opt[j+1]) / 2.0
                           r_c[idx] = min(wc[i], hc[j]) / 2.0
                           idx += 1
                   
                   # Validate
                   # Check bounds
                   valid = True
                   for k in range(N):
                       if x_c[k] - r_c[k] < -1e-9 or x_c[k] + r_c[k] > w_opt + 1e-9:
                           valid = False
                       if y_c[k] - r_c[k] < -1e-9 or y_c[k] + r_c[k] > h_opt + 1e-9:
                           valid = False
                   
                   # Check disjointness (should be guaranteed by grid)
                   for i in range(N):
                       for j in range(i+1, N):
                           d2 = (x_c[i]-x_c[j])**2 + (y_c[i]-y_c[j])**2
                           rs = r_c[i] + r_c[j]
                           if d2 < rs*rs - 1e-9:
                               valid = False
                               
                   if valid:
                       val = np.sum(r_c)
                       if val > best_value:
                           best_value = val
                           best_result = (x_c, y_c, r_c, w_opt)
       except Exception:
           pass
           
       if best_result is None:
           # Fallback to uniform 7x3
           w = 1.4
           h = 0.6
           x_c = np.zeros(N)
           y_c = np.zeros(N)
           r_c = np.zeros(N)
           for i in range(NX):
               for j in range(NY):
                   idx = i * NY + j
                   x_c[idx] = (i + 0.5) * w / NX
                   y_c[idx] = (j + 0.5) * h / NY
                   r_c[idx] = min(w/NX, h/NY) / 2.0
           best_result = (x_c, y_c, r_c, w)
           best_value = np.sum(r_c)
           
       x, y, r, w = best_result
       # Slight shrink to ensure validity
       r = r * (1 - 1e-9)
       
       return np.column_stack((x, y, r))
