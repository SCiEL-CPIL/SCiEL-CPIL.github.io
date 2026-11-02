import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Variables: W, x1, y1, r1, ..., xN, yN, rN
    # Index 0: W
    # Index 1 + 3*i: xi
    # Index 2 + 3*i: yi
    # Index 3 + 3*i: ri
    
    def unpack(v):
        W = v[0]
        H = 2.0 - W
        coords = v[1:]
        # coords shape (3*N)
        xs = coords[0::3]
        ys = coords[1::3]
        rs = coords[2::3]
        return W, H, xs, ys, rs

    def objective(v):
        W, H, xs, ys, rs = unpack(v)
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(rs)

    def constraints(v):
        W, H, xs, ys, rs = unpack(v)
        cons = []
        
        # W bounds: 0 < W < 2
        # H bounds: 0 < H < 2 => 0 < 2-W < 2 => 0 < W < 2
        # We handle these with bounds in the optimizer, but let's add explicit ones if needed.
        # Actually, SLSQP handles bounds well.
        
        # Circle bounds
        # xi >= 0, xi <= W
        # yi >= 0, yi <= H
        # ri >= 0
        
        # Pairwise non-overlap
        # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        
        # Vectorized calculation for all pairs
        # This might be slow if done naively in python loops. 
        # N=21, pairs = 210. It's small enough for a loop, but let's try to be efficient.
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dr = rs[i] + rs[j]
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[1+3*i] - v[1+3*j])**2 + (v[2+3*i] - v[2+3*j])**2 - (v[3+3*i] + v[3+3*j])**2
                })
        return cons

    def bounds():
        b = [(1e-6, 2.0 - 1e-6)] # W
        for i in range(N):
            b.append((0.0, 2.0)) # x (max possible width is 2 if H~0, but practically limited by W)
            b.append((0.0, 2.0)) # y
            b.append((0.0, 1.0)) # r (max radius in 2x2 rect is 1)
        return b

    # Initialization
    # Start with a square 1x1 and a grid of points
    W0 = 1.0
    H0 = 1.0
    side = int(np.ceil(np.sqrt(N)))
    xs0 = np.zeros(N)
    ys0 = np.zeros(N)
    rs0 = np.zeros(N)
    
    # Fill grid
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            xs0[idx] = (i + 0.5) / side
            ys0[idx] = (j + 0.5) / side
            rs0[idx] = 0.5 / side - 0.01 # Slightly smaller to avoid initial violation
            idx += 1
            
    # Pad if N is not a perfect square (21 is 5x5=25, so we use 21 of them)
    # The above loop fills 21 items in a 5x5 grid.
    
    v0 = np.zeros(1 + 3*N)
    v0[0] = W0
    for i in range(N):
        v0[1+3*i] = xs0[i]
        v0[2+3*i] = ys0[i]
        v0[3+3*i] = rs0[i]
        
    best_result = None
    best_value = -np.inf
    
    # Try a few random restarts if time permits
    # But first, try the deterministic start
    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds(), constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
        if res.success:
            best_value = -res.fun
            best_result = res.x
    except Exception as e:
        pass

    # If time remains, do random restarts
    # Generate random points in a 1x1 square
    while time.time() < deadline - 10:
        W0 = np.random.uniform(0.8, 1.2)
        H0 = 2.0 - W0
        xs0 = np.random.uniform(0, W0, N)
        ys0 = np.random.uniform(0, H0, N)
        # Small initial radii to ensure feasibility
        rs0 = np.full(N, 0.05)
        
        v0 = np.zeros(1 + 3*N)
        v0[0] = W0
        for i in range(N):
            v0[1+3*i] = xs0[i]
            v0[2+3*i] = ys0[i]
            v0[3+3*i] = rs0[i]
            
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds(), constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success and -res.fun > best_value:
                best_value = -res.fun
                best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Extract result
    W = best_result[0]
    H = 2.0 - W
    xs = best_result[1::3]
    ys = best_result[2::3]
    rs = best_result[3::3]
    
    # Final validation and cleanup
    # Ensure circles are inside
    valid = True
    for i in range(N):
        if xs[i] < 0 or xs[i] > W or ys[i] < 0 or ys[i] > H or rs[i] < 0:
            valid = False
            break
        # Check overlap
        for j in range(i+1, N):
            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if dist < rs[i] + rs[j] - 1e-9:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
         # If invalid, try to shrink radii slightly or fall back
         # Shrink radii to fix overlaps
         for i in range(N):
             for j in range(i+1, N):
                 dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                 if dist < rs[i] + rs[j]:
                     # Scale down both radii
                     factor = dist / (rs[i] + rs[j]) * 0.99
                     rs[i] *= factor
                     rs[j] *= factor
                     # Re-check bounds
                     rs[i] = max(0, min(rs[i], min(xs[i], W-xs[i], ys[i], H-ys[i])))
                     rs[j] = max(0, min(rs[j], min(xs[j], W-xs[j], ys[j], H-ys[j])))
    
    result = np.column_stack((xs, ys, rs))
    return result
