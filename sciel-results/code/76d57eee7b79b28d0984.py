"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin within 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios (w/h)
    # w + h = 2, so h = 2 - w
    # Aspect ratio ranges from very narrow to very wide
    aspect_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        # w + h = 2, w/h = ar => w = ar*h, ar*h + h = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try multiple initial configurations
        for init_type in ['grid', 'random1', 'random2']:
            if time.time() > deadline - 3.0:
                break
            
            # Initialize positions and radii
            if init_type == 'grid':
                # Grid initialization
                cols = int(np.ceil(np.sqrt(N * w / h))) if w >= h else int(np.ceil(np.sqrt(N * h / w)))
                if cols < 1:
                    cols = 1
                rows = int(np.ceil(N / cols))
                
                xs = np.zeros(N)
                ys = np.zeros(N)
                rs = np.zeros(N)
                for i in range(N):
                    c = i % cols
                    r = i // cols
                    xs[i] = (c + 0.5) * w / cols
                    ys[i] = (r + 0.5) * h / rows
                    rs[i] = min(w / cols, h / rows) * 0.4
            else:
                # Random initialization
                np.random.seed(42 if init_type == 'random1' else 123)
                xs = np.random.uniform(0.1, w - 0.1, N)
                ys = np.random.uniform(0.1, h - 0.1, N)
                rs = np.random.uniform(0.05, 0.2, N)
            
            # Variables: [x0, y0, r0, x1, y1, r1, ...]
            x0 = np.concatenate([xs, ys, rs])
            
            def objective(v):
                if time.time() > deadline - 1.0:
                    return 0.0
                xs_v = v[0:N]
                ys_v = v[N:2*N]
                rs_v = v[2*N:3*N]
                return -np.sum(rs_v)
            
            def constraints(v):
                if time.time() > deadline - 1.0:
                    return []
                xs_v = v[0:N]
                ys_v = v[N:2*N]
                rs_v = v[2*N:3*N]
                cons = []
                
                # Boundary constraints: circles inside rectangle
                for i in range(N):
                    # x_i - r_i >= 0
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N + i]})
                    # x_i + r_i <= w
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[i] - v[2*N + i]})
                    # y_i - r_i >= 0
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N + i] - v[2*N + i]})
                    # y_i + r_i <= h
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: h - v[N + i] - v[2*N + i]})
                    # r_i >= 0.001
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N + i] - 0.001})
                
                # Non-overlap constraints
                for i in range(N):
                    for j in range(i + 1, N):
                        cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                                     (v[i] - v[j])**2 + (v[N+i] - v[N+j])**2 - (v[2*N+i] + v[2*N+j])**2})
                
                return cons
            
            try:
                res = minimize(objective, x0, method='SLSQP', 
                              constraints=constraints,
                              bounds=[(0, w)] * N + [(0, h)] * N + [(0.001, min(w, h)/2)] * N,
                              options={'maxiter': 200, 'ftol': 1e-6})
                
                if res.success and res.fun < best_score:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        # Extract and verify
                        xs_v = res.x[0:N]
                        ys_v = res.x[N:2*N]
                        rs_v = res.x[2*N:3*N]
                        
                        # Verify validity
                        valid = True
                        for i in range(N):
                            if xs_v[i] - rs_v[i] < -1e-6 or xs_v[i] + rs_v[i] > w + 1e-6:
                                valid = False
                                break
                            if ys_v[i] - rs_v[i] < -1e-6 or ys_v[i] + rs_v[i] > h + 1e-6:
                                valid = False
                                break
                        if valid:
                            for i in range(N):
                                for j in range(i + 1, N):
                                    dist = np.sqrt((xs_v[i] - xs_v[j])**2 + (ys_v[i] - ys_v[j])**2)
                                    if dist < rs_v[i] + rs_v[j] - 1e-6:
                                        valid = False
                                        break
                                if not valid:
                                    break
                        
                        if valid:
                            result = np.column_stack([xs_v, ys_v, rs_v])
                            if best_result is None or score > best_score:
                                best_result = result
                                best_score = score
            except Exception as e:
                pass
    
    # If no valid result found, fall back to grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
