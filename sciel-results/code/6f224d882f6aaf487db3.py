"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Uses a structural change: SDP-relaxation-inspired initialization via
distance-matrix embedding, combined with a contact-graph-aware penalty
optimization and a multi-restart local search with diverse aspect ratios.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=800, tol=1e-9):
    """Relaxation to resolve overlaps and boundary violations."""
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False
        # Boundary
        for i in range(n):
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]; moved = True
            if pos[i, 0] > w - rad[i]:
                pos[i, 0] = w - rad[i]; moved = True
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]; moved = True
            if pos[i, 1] > h - rad[i]:
                pos[i, 1] = h - rad[i]; moved = True
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j,0] - pos[i,0]
                dy = pos[j,1] - pos[i,1]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rad[i] + rad[j]
                if d < min_d - 1e-12:
                    if d < 1e-10:
                        angle = np.random.uniform(0, 2*np.pi)
                        ux, uy = np.cos(angle), np.sin(angle)
                    else:
                        ux, uy = dx/d, dy/d
                    push = (min_d - d) / 2.0
                    pos[i,0] -= ux*push; pos[i,1] -= uy*push
                    pos[j,0] += ux*push; pos[j,1] += uy*push
                    moved = True
        if not moved:
            break
    # Final clamp
    for i in range(n):
        pos[i,0] = np.clip(pos[i,0], rad[i], w - rad[i])
        pos[i,1] = np.clip(pos[i,1], rad[i], h - rad[i])
    new_circles = np.column_stack((pos, rad))
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    return circles

def _sdp_inspired_init(n, w, h, seed=0):
    """
    SDP-relaxation-inspired initialization:
    1. Create a random distance matrix consistent with Euclidean geometry.
    2. Embed into 2D via eigen-decomposition (PCoA).
    3. Scale to fit rectangle, set radii from half-min-distances.
    This produces a configuration where pairwise distances are 'aware' of
    global geometry, unlike pure grid init.
    """
    rng = np.random.RandomState(seed)
    
    # Random initial positions in unit square
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    
    # Compute distance matrix
    D2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d2 = (pos[i]-pos[j])**2
            D2[i,j] = d2
            D2[j,i] = d2
    
    # Double centering (PCoA)
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ D2 @ J
    
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 principal components
    embedded = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    
    # Scale to fit rectangle with margin
    margin = 0.15
    ex, ey = embedded[:, 0], embedded[:, 1]
    ex_min, ex_max = ex.min(), ex.max()
    ey_min, ey_max = ey.min(), ey.max()
    ex_range = max(ex_max - ex_min, 1e-6)
    ey_range = max(ey_max - ey_min, 1e-6)
    
    scale_x = (w - 2*margin) / ex_range
    scale_y = (h - 2*margin) / ey_range
    scale = min(scale_x, scale_y)
    
    new_ex = (ex - ex_min) * scale + margin
    new_ey = (ey - ey_min) * scale + margin
    
    # Set radii: half of minimum distance to any other circle, capped
    radii = np.zeros(n)
    for i in range(n):
        min_d = np.inf
        for j in range(n):
            if i != j:
                d = np.sqrt((new_ex[i]-new_ex[j])**2 + (new_ey[i]-new_ey[j])**2)
                min_d = min(min_d, d)
        radii[i] = 0.48 * min_d  # slight gap
    
    # Cap radii so circles fit in rectangle
    for i in range(n):
        max_r = min(new_ex[i], w - new_ex[i], new_ey[i], h - new_ey[i])
        radii[i] = min(radii[i], max_r * 0.95)
    
    circles = np.column_stack((new_ex, new_ey, radii))
    return circles

def _hex_init(n, w, h, rows=None):
    """Hexagonal grid initialization."""
    if rows is None:
        rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        offset = 0.5 * (w / max(rows[0],1)) if row_i % 2 == 1 else 0
        for col_j in range(count):
            x = (col_j + 0.5) * w / max(rows[0],1) + offset
            r = 0.4 * min(w / max(rows[0],1), h / total_rows)
            circles.append([x, y, r])
    return np.array(circles[:n])

def _penalty_optimize(circles, w, h, deadline, beta_start=5.0, beta_end=500.0, n_beta_steps=8):
    """Penalty-based L-BFGS-B optimization."""
    if time.time() > deadline - 5.0:
        return None
    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    bounds = [(1e-7, w), (1e-7, h)] * n + [(1e-7, min(w,h)/2)] * n

    def obj_and_grad(x, beta):
        cx, cy, r = x[:n], x[n:2*n], x[2*n:]
        obj = -np.sum(r)
        grad = np.zeros(3*n)
        grad[2*n:] = -1.0
        # Boundary
        for i in range(n):
            for c, lim in [(cx[i], 0), (cx[i], w), (cy[i], 0), (cy[i], h)]:
                if c - r[i] < 0 and lim == 0:
                    v = c - r[i]
                    obj += beta * v*v
                    grad[i if lim==0 or lim==w else n+i] += 2*beta*v
                    grad[2*n+i] += -2*beta*v
                if c + r[i] > lim and lim > 0:
                    v = c + r[i] - lim
                    obj += beta * v*v
                    grad[i if lim==w else n+i] += 2*beta*v
                    grad[2*n+i] += 2*beta*v
        # Overlap
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            d2 = dx*dx + dy*dy
            md = r[i] + r[i+1:]
            mask = d2 < md*md
            if np.any(mask):
                pos_idx = np.where(mask)[0]
                d = np.sqrt(d2[mask] + 1e-12)
                ov = md[mask] - d
                for k in range(len(ov)):
                    j = i + 1 + pos_idx[k]
                    if ov[k] > 0:
                        obj += beta * ov[k]**2
                        ux, uy = dx[pos_idx[k]]/d[k], dy[pos_idx[k]]/d[k]
                        grad[i] += -2*beta*ov[k]*ux
                        grad[j] += 2*beta*ov[k]*ux
                        grad[n+i] += -2*beta*ov[k]*uy
                        grad[n+j] += 2*beta*ov[k]*uy
                        grad[2*n+i] += 2*beta*ov[k]
                        grad[2*n+j] += 2*beta*ov[k]
        return obj, grad

    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    x = x0.copy()
    for beta in betas:
        if time.time() > deadline - 3.0:
            break
        try:
            res = minimize(lambda x: obj_and_grad(x, beta)[0], x,
                           jac=lambda x: obj_and_grad(x, beta)[1],
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success:
                x = res.x
        except:
            pass
    new_c = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    new_c = _relax_circles(new_c, w, h, iterations=300)
    if _validate(new_c, w, h, tol=1e-5):
        return new_c
    return None

def _slsqp_polish(circles, w, h, deadline):
    """SLSQP with hard constraints."""
    if time.time() > deadline - 10.0:
        return None
    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    cons = []
    for i in range(n):
        cons.append({'type':'ineq','fun':lambda x,i=i: x[i]-x[2*n+i]})
        cons.append({'type':'ineq','fun':lambda x,i=i: w-x[i]-x[2*n+i]})
        cons.append({'type':'ineq','fun':lambda x,i=i: x[n+i]-x[2*n+i]})
        cons.append({'type':'ineq','fun':lambda x,i=i: h-x[n+i]-x[2*n+i]})
    for i in range(n):
        for j in range(i+1,n):
            cons.append({'type':'ineq','fun':lambda x,i=i,j=j:
                         np.sqrt((x[i]-x[j])**2+(x[n+i]-x[n+j])**2)-x[2*n+i]-x[2*n+j]})
    bounds = [(1e-7,w),(1e-7,h)]*n + [(1e-7,min(w,h)/2)]*n
    try:
        res = minimize(lambda x: -np.sum(x[2*n:]), x0, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter':100,'ftol':1e-12})
        if res.success:
            nc = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            if _validate(nc, w, h, tol=1e-5):
                return nc
    except:
        pass
    return None

def _linearized_refine(circles, w, h, deadline, max_iter=12):
    """LP-based convex refinement with linearized non-overlap constraints."""
    if time.time() > deadline - 12.0:
        return circles
    n = len(circles)
    cx, cy, r = circles[:,0].copy(), circles[:,1].copy(), circles[:,2].copy()
    best_val = _total_radius(circles)
    best_c = circles.copy()
    for _ in range(max_iter):
        if time.time() > deadline - 14.0:
            break
        c_obj = np.zeros(3*n)
        c_obj[2*n:] = -1.0
        A_rows, b_rows = [], []
        for i in range(n):
            row = np.zeros(3*n); row[i]=-1; row[2*n+i]=1; A_rows.append(row); b_rows.append(0)
            row = np.zeros(3*n); row[i]=1; row[2*n+i]=1; A_rows.append(row); b_rows.append(w)
            row = np.zeros(3*n); row[n+i]=-1; row[2*n+i]=1; A_rows.append(row); b_rows.append(0)
            row = np.zeros(3*n); row[n+i]=1; row[2*n+i]=1; A_rows.append(row); b_rows.append(h)
        for i in range(n):
            for j in range(i+1,n):
                dx0, dy0 = cx[i]-cx[j], cy[i]-cy[j]
                d0 = np.sqrt(dx0*dx0+dy0*dy0)
                if d0 < 1e-12:
                    dx0, dy0, d0 = 1.0, 0.0, 1.0
                row = np.zeros(3*n)
                row[i]=-dx0; row[j]=dx0; row[n+i]=-dy0; row[n+j]=dy0
                row[2*n+i]=1; row[2*n+j]=1
                A_rows.append(row); b_rows.append(0)
        A_ub = np.array(A_rows); b_ub = np.array(b_rows)
        bl = [(1e-8,w-1e-8),(1e-8,h-1e-8)]*n + [(1e-8,min(w,h)/2)]*n
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bl, method='highs')
            if res.status == 0:
                nc = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
                if _validate(nc, w, h, tol=1e-6):
                    v = _total_radius(nc)
                    if v > best_val + 1e-10:
                        best_val = v; best_c = nc.copy()
                        cx, cy, r = nc[:,0].copy(), nc[:,1].copy(), nc[:,2].copy()
                    else:
                        break
                else:
                    rel = _relax_circles(nc, w, h, iterations=100)
                    if _validate(rel, w, h, tol=1e-6):
                        v = _total_radius(rel)
                        if v > best_val + 1e-10:
                            best_val = v; best_c = rel.copy()
                            cx, cy, r = rel[:,0].copy(), rel[:,1].copy(), rel[:,2].copy()
                        else:
                            break
                    else:
                        break
            else:
                break
        except:
            break
    return best_c

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    best_val = 0.0
    best_res = None

    # Broader aspect ratio search
    ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.4, 2.5, 0.35]

    for ar in ratios:
        if time.time() > deadline - 20.0:
            break
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 20.0:
                break
            try:
                # Multiple initializations
                inits = []
                # Hexagonal
                inits.append(_hex_init(N, ww, hh))
                # SDP-inspired with different seeds
                inits.append(_sdp_inspired_init(N, ww, hh, seed=42))
                inits.append(_sdp_inspired_init(N, ww, hh, seed=123))

                for init_c in inits:
                    if time.time() > deadline - 20.0:
                        break
                    c = _relax_circles(init_c, ww, hh, iterations=500)
                    if not _validate(c, ww, hh):
                        continue
                    val = _total_radius(c)

                    # Penalty optimize
                    if time.time() < deadline - 25.0:
                        opt = _penalty_optimize(c, ww, hh, deadline)
                        if opt is not None:
                            c = opt
                            val = _total_radius(c)

                    # SLSQP polish
                    if time.time() < deadline - 15.0:
                        pol = _slsqp_polish(c, ww, hh, deadline)
                        if pol is not None:
                            c = pol
                            val = _total_radius(c)

                    # LP refinement
                    if time.time() < deadline - 13.0:
                        ref = _linearized_refine(c, ww, hh, deadline, max_iter=10)
                        if ref is not None:
                            c = ref
                            val = _total_radius(c)

                    if val > best_val:
                        best_val = val
                        best_res = c.copy()
            except:
                continue

    if best_res is None:
        # Fallback: equal circles in 1x1
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i+0.5)*0.2, (j+0.5)*0.2, 0.09])
        return np.array(cells[:N])

    return best_res
