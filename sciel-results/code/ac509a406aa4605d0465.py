import numpy as np

N = 21

def construct():
    def compute_cost_and_grad(x):
        xs = x[0:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        w = x[3*N]
        h = 2.0 - w
        
        cost = -np.sum(rs)
        grad = np.zeros_like(x)
        grad[2*N:3*N] = -1.0
        
        overlap_pen = 0.0
        d_overlap_dx = np.zeros(N)
        d_overlap_dy = np.zeros(N)
        d_overlap_dr = np.zeros(N)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-8:
                    dist = 1e-8
                    dx = 1e-8
                    dy = 0.0
                min_dist = rs[i] + rs[j]
                delta = min_dist - dist
                if delta > 0:
                    overlap_pen += delta * delta
                    factor = 2.0 * delta / dist
                    d_overlap_dx[i] += factor * dx
                    d_overlap_dx[j] -= factor * dx
                    d_overlap_dy[i] += factor * dy
                    d_overlap_dy[j] -= factor * dy
                    d_overlap_dr[i] += 2.0 * delta
                    d_overlap_dr[j] += 2.0 * delta
        
        cost += 100.0 * overlap_pen
        grad[0:N] += 100.0 * d_overlap_dx
        grad[N:2*N] += 100.0 * d_overlap_dy
        grad[2*N:3*N] += 100.0 * d_overlap_dr
        
        bound_pen = 0.0
        d_bound_dx = np.zeros(N)
        d_bound_dy = np.zeros(N)
        d_bound_dr = np.zeros(N)
        d_bound_dw = 0.0
        
        for i in range(N):
            if xs[i] < rs[i]:
                val = rs[i] - xs[i]
                bound_pen += val * val
                d_bound_dx[i] += -2.0 * val
                d_bound_dr[i] += 2.0 * val
            elif xs[i] > w - rs[i]:
                val = xs[i] - (w - rs[i])
                bound_pen += val * val
                d_bound_dx[i] += 2.0 * val
                d_bound_dr[i] += -2.0 * val
                d_bound_dw += -2.0 * val
                
            if ys[i] < rs[i]:
                val = rs[i] - ys[i]
                bound_pen += val * val
                d_bound_dy[i] += -2.0 * val
                d_bound_dr[i] += 2.0 * val
            elif ys[i] > h - rs[i]:
                val = ys[i] - (h - rs[i])
                bound_pen += val * val
                d_bound_dy[i] += 2.0 * val
                d_bound_dr[i] += -2.0 * val
                d_bound_dw += 2.0 * val
        
        cost += 100.0 * bound_pen
        grad[0:N] += 100.0 * d_bound_dx
        grad[N:2*N] += 100.0 * d_bound_dy
        grad[2*N:3*N] += 100.0 * d_bound_dr
        grad[3*N] += 100.0 * d_bound_dw
        
        return cost, grad

    def run_gradient_descent(x0, iterations=500):
        x = x0.copy()
        lr = 0.01
        best_cost, _ = compute_cost_and_grad(x)
        best_x = x.copy()
        velocity = np.zeros_like(x)
        momentum = 0.9
        
        for it in range(iterations):
            cost, grad = compute_cost_and_grad(x)
            velocity = momentum * velocity - lr * grad
            x_new = x + velocity
            x_new[3*N] = np.clip(x_new[3*N], 0.5, 1.5)
            x_new[2*N:3*N] = np.clip(x_new[2*N:3*N], 0.01, 1.0)
            
            cost_new, _ = compute_cost_and_grad(x_new)
            
            if cost_new < best_cost:
                best_cost = cost_new
                best_x = x_new.copy()
                x = x_new
            else:
                lr = lr * 0.5
                velocity = 0.5 * velocity
                if lr < 1e-5:
                    break
                    
        return best_x, best_cost

    def is_valid(xs, ys, rs, w, h):
        for i in range(N):
            if xs[i] < rs[i] - 1e-6 or xs[i] > w - rs[i] + 1e-6:
                return False
            if ys[i] < rs[i] - 1e-6 or ys[i] > h - rs[i] + 1e-6:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - 1e-6:
                    return False
        return True

    side = 5
    xs_init = []
    ys_init = []
    for i in range(side):
        for j in range(side):
            if len(xs_init) >= N:
                break
            xs_init.append((i + 0.5) / side)
            ys_init.append((j + 0.5) / side)
    xs_init = np.array(xs_init)
    ys_init = np.array(ys_init)
    r_init = np.full(N, 0.15)
    w_init = np.array([1.0])
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    for restart in range(5):
        noise = np.random.normal(0, 0.01, N)
        x0 = np.concatenate([xs_init + noise, ys_init + noise, r_init, w_init])
        x_res, cost_res = run_gradient_descent(x0)
        
        if cost_res < best_overall_cost:
            best_overall_cost = cost_res
            best_overall_x = x_res.copy()
            
    if best_overall_x is None:
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.02) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    xs_res = best_overall_x[0:N]
    ys_res = best_overall_x[N:2*N]
    rs_res = best_overall_x[2*N:3*N]
    w_res = np.clip(best_overall_x[3*N], 0.5, 1.5)
    h_res = 2.0 - w_res
    
    if not is_valid(xs_res, ys_res, rs_res, w_res, h_res):
        for _ in range(10):
            rs_res *= 0.95
            if is_valid(xs_res, ys_res, rs_res, w_res, h_res):
                break
        if not is_valid(xs_res, ys_res, rs_res, w_res, h_res):
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.02) for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    out = np.zeros((N, 3))
    out[:, 0] = xs_res
    out[:, 1] = ys_res
    out[:, 2] = rs_res
    
    return out
