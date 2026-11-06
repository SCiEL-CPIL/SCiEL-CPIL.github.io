import numpy as np

def _robertson_jacobian(y):
    J = np.zeros((3, 3))
    J[0, 0] = -0.04
    J[0, 1] = 10000.0 * y[2]
    J[0, 2] = 10000.0 * y[1]
    J[1, 0] = 0.04
    J[1, 1] = -10000.0 * y[2] - 6000.0 * y[1]
    J[1, 2] = -10000.0 * y[1]
    J[2, 0] = 0.0
    J[2, 1] = 6000.0 * y[1]
    J[2, 2] = 0.0
    return J

def _newton_solve_3x3(f, jac, t, h, A_coeff, y_prev):
    y = y_prev.copy()
    iters = 0
    for _ in range(10):
        fy = f(t, y)
        J = jac(y)
        R = y - y_prev - h * A_coeff * fy
        JR = np.eye(3) - h * A_coeff * J
        try:
            delta = np.linalg.solve(JR, -R)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(JR, -R, rcond=None)[0]
        y += delta
        iters += 1
        if np.linalg.norm(delta) < 1e-12 * (1 + np.linalg.norm(y)):
            break
    return y, iters

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    c = np.array([1/6, 1/2, 1.0])
    A = np.array([
        [1/6, 0, 0],
        [3/4, -1/4, 0],
        [19/30, 1/2, -1/30]
    ])
    b = np.array([19/30, 1/2, -1/30])
    
    h = 1e-4
    max_h = (t1 - t0) / 5.0
    min_h = 1e-12
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        
        t1_stage = t + c[0] * h
        Y1, it1 = _newton_solve_3x3(f, _robertson_jacobian, t1_stage, h, A[0, 0], y)
        
        t2_stage = t + c[1] * h
        f1 = f(t1_stage, Y1)
        y_prev_eff2 = y + h * A[1, 0] * f1
        Y2, it2 = _newton_solve_3x3(f, _robertson_jacobian, t2_stage, h, A[1, 1], y_prev_eff2)
        
        t3_stage = t + c[2] * h
        f2 = f(t2_stage, Y2)
        y_prev_eff3 = y + h * A[2, 0] * f1 + h * A[2, 1] * f2
        Y3, it3 = _newton_solve_3x3(f, _robertson_jacobian, t3_stage, h, A[2, 2], y_prev_eff3)
        
        f3 = f(t3_stage, Y3)
        y_new = y + h * (b[0] * f1 + b[1] * f2 + b[2] * f3)
        
        # Heuristic step size control
        max_it = max(it1, it2, it3)
        if max_it <= 3:
            factor = 2.0
        elif max_it <= 5:
            factor = 1.5
        else:
            factor = 0.5
            
        h_new = h * factor
        h_new = np.clip(h_new, min_h, max_h)
        
        y = y_new
        t += h
        h = min(h_new, t1 - t)
        
    return y
