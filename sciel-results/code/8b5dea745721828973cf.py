import numpy as np

N = 26

# EVOLVE-BLOCK-START
import numpy as np

def _resolve_vec(x, y, r, iters=500):
    n = len(r)
    for _ in range(iters):
        moved = 0.0
        # Wall constraints
        for i in range(n):
            if x[i] < r[i]:
                moved += r[i] - x[i]
                x[i] = r[i]
            elif x[i] > 1 - r[i]:
                moved += x[i] - (1 - r[i])
                x[i] = 1 - r[i]
            if y[i] < r[i]:
                moved += r[i] - y[i]
                y[i] = r[i]
            elif y[i] > 1 - r[i]:
                moved += y[i] - (1 - r[i])
                y[i] = 1 - r[i]
        
        # Pairwise: vectorized
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        req = r[None, :] + r[:, None]
        overlap = req - dist
        overlap = np.maximum(overlap, 0)
        if np.max(overlap) < 1e-10:
            break
        moved += np.sum(overlap)
        
        # Apply corrections
        for i in range(n):
            for j in range(i + 1, n):
                ov = overlap[i, j]
                if ov > 1e-10:
                    d = dist[i, j]
                    if d < 1e-12:
                        # Randomly nudge if coincident
                        x[i] += np.random.uniform(-0.001, 0.001)
                        y[i] += np.random.uniform(-0.001, 0.001)
                        continue
                    half = ov / 2 + 1e-8
                    ux = dx[i, j] / d
                    uy = dy[i, j] / d
                    x[i] -= half * ux
                    y[i] -= half * uy
                    x[j] += half * ux
                    y[j] += half * uy
    
    # Final clamp
    for i in range(n):
        x[i] = min(max(x[i], r[i]), 1 - r[i])
        y[i] = min(max(y[i], r[i]), 1 - r[i])
    return x, y

def _valid(x, y, r, tol=1e-6):
    n = len(r)
    for i in range(n):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
            return False
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    req = r[None, :] + r[:, None]
    if np.any(dist < req - tol):
        return False
    return True

def _score(x, y, r):
    if not _valid(x, y, r, tol=1e-5):
        return -1e10
    return np.sum(r)

def _grow(x, y, r, max_steps=300, growth=0.005):
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = _score(best_x, best_y, best_r)
    
    for _ in range(max_steps):
        r_new = r * (1 + growth)
        x_try, y_try = _resolve_vec(x.copy(), y.copy(), r_new, iters=300)
        s = _score(x_try, y_try, r_new)
        if s > best_score - 1e-9:
            best_x, best_y, best_r = x_try, y_try, r_new
            best_score = s
            r = r_new
            x = x_try
            y = y_try
        else:
            # Try smaller growth
            found = False
            for g in [0.002, 0.001, 0.0005]:
                r_try = r * (1 + g)
                x_try, y_try = _resolve_vec(x.copy(), y.copy(), r_try, iters=300)
                s = _score(x_try, y_try, r_try)
                if s > best_score - 1e-9:
                    best_x, best_y, best_r = x_try, y_try, r_try
                    best_score = s
                    r = r_try
                    x = x_try
                    y = y_try
                    found = True
                    break
            if not found:
                break
    
    return best_x, best_y, best_r, best_score

def _hex_init(config):
    x_list = []
    y_list = []
    ref_r = 0.1
    dx = ref_r * np.sqrt(3)
    dy = ref_r * 1.5
    
    total = 0
    for i, row_len in enumerate(config):
        if total >= N: break
        y = (i + 0.5) * dy
        for j in range(row_len):
            if total >= N: break
            x = (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2.0
            x_list.append(x)
            y_list.append(y)
            total += 1
            
    x = np.array(x_list[:N])
    y = np.array(y_list[:N])
    
    # Center and scale
    cx = (np.min(x) + np.max(x)) / 2
    cy = (np.min(y) + np.max(y)) / 2
    x -= cx
    y -= cy
    
    # Scale to fit in 0.8 box
    x_max = np.max(np.abs(x))
    y_max = np.max(np.abs(y))
    scale = 0.4 / max(x_max, y_max)
    x *= scale
    y *= scale
    
    x += 0.5
    y += 0.5
    
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    
    return x, y

def construct():
    configs = [
        [5, 6, 5, 5, 5],
        [6, 5, 6, 5, 4],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 5, 5],
    ]
    
    best_x, best_y, best_r, best_score = None, None, None, -1e10
    
    for config in configs:
        x, y = _hex_init(config)
        r = np.full(N, 0.015)
        x, y, r, score = _grow(x, y, r, max_steps=200, growth=0.005)
        if score > best_score:
            best_x, best_y, best_r, best_score = x, y, r, score
    
    if best_x is None or best_score <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final polish
    best_x, best_y = _resolve_vec(best_x, best_y, best_r, iters=500)
    
    if not _valid(best_x, best_y, best_r, tol=1e-5):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack((best_x, best_y, best_r))
# EVOLVE-BLOCK-END
