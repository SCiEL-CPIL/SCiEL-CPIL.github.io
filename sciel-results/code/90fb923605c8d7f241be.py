import time
import numpy as np
import random

N = 12
DEADLINE = 140.0
_t0 = time.time()

def _hex_vertices(cx, cy, ang_deg, side=1.0):
    a = np.deg2rad(ang_deg)
    th = a + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + side * np.cos(th), cy + side * np.sin(th)])

def _all_vertices(inner):
    pts = [_hex_vertices(x, y, ang) for (x, y, ang) in inner]
    return np.vstack(pts)

def _enclosing_side(V, theta_cands):
    """Calculate minimal enclosing hexagon side for given vertices and candidate angles."""
    c = V.mean(axis=0)
    best = np.inf
    for th in theta_cands:
        a = np.deg2rad(th)
        normals = np.column_stack([np.cos(a + np.arange(6) * np.pi / 3.0),
                                   np.sin(a + np.arange(6) * np.pi / 3.0)])
        proj = V @ normals.T
        # Distance from center to furthest projection
        # The side length S of a regular hexagon with apothem A is S = 2A/sqrt(3).
        # Here, we need the hexagon to contain V.
        # The support function h(n) = max(v.n).
        # For a hexagon centered at c with side S and orientation phi,
        # the width in direction n is 2 * S * cos(angle(n, normal)) / sqrt(3) ? No.
        # A regular hexagon with side S has apothem A = S * sqrt(3) / 2.
        # The distance from center to edge is A.
        # So we need max(v.n - c.n) <= A for all 6 normals.
        # Thus A >= max(v.n - c.n).
        # S = 2 * A / sqrt(3).
        m = proj.max(axis=0) - normals @ c
        A = m.max()
        s = (2.0 / np.sqrt(3.0)) * A
        if s < best:
            best = s
    return best

def _sat_penetration(A, B):
    """Minimum translation depth if convex hexagons A,B overlap, else 0."""
    axes = []
    for P in (A, B):
        for i in range(6):
            e = P[(i + 1) % 6] - P[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                axes.append(n / nn)
    min_pen = np.inf
    for ax in axes:
        pa = A @ ax
        pb = B @ ax
        overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if overlap <= 0.0:
            return 0.0
        if overlap < min_pen:
            min_pen = overlap
    return min_pen

def _total_penetration(inner):
    verts = [_hex_vertices(x, y, ang) for (x, y, ang) in inner]
    tot = 0.0
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            p = _sat_penetration(verts[i], verts[j])
            tot += p
    return tot

def _initial_inner():
    d = np.sqrt(3.0)
    pos = []
    for row in range(3):
        for col in range(4):
            x = col * d + (0.5 * d if row % 2 else 0.0)
            y = row * 1.5
            pos.append((x, y))
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    mx = (min(xs) + max(xs)) / 2.0
    my = (min(ys) + max(ys)) / 2.0
    pos = [(x - mx, y - my) for (x, y) in pos]
    return [[x, y, 30.0] for (x, y) in pos]

def _flatten(inner):
    return [v for row in inner for v in row]

def _unflatten(flat):
    return [[flat[i], flat[i + 1], flat[i + 2]] for i in range(0, len(flat), 3)]

def construct():
    global _t0
    theta_cands = np.linspace(0.0, 60.0, 13)[:-1]
    
    # Start with honeycomb
    x0 = _flatten(_initial_inner())
    current_flat = list(x0)
    current_inner = _unflatten(current_flat)
    
    # Ensure initial state is valid (it should be, but let's verify)
    # If not, we might need to perturb. But honeycomb is exact.
    
    best_flat = list(current_flat)
    best_side = _enclosing_side(_all_vertices(current_inner), theta_cands)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.99
    steps = 5000
    
    # We need to be fast. 5000 steps * 12 hexes * SAT checks might be slow.
    # Let's optimize SAT check or reduce steps.
    # 12 hexes -> 66 pairs.
    # 5000 * 66 = 330k SAT checks.
    # Each SAT check is ~12 axis projections.
    # Should be fine in Python if vectorized? No, it's a loop.
    # Let's try 2000 steps.
    
    for step in range(2000):
        if time.time() - _t0 > DEADLINE - 5:
            break
            
        # Pick a random hexagon
        idx = random.randint(0, N - 1)
        i = idx * 3
        
        # Propose change
        new_flat = list(current_flat)
        # Small random perturbation
        new_flat[i] += random.gauss(0, 0.1)
        new_flat[i+1] += random.gauss(0, 0.1)
        new_flat[i+2] += random.gauss(0, 5.0)
        
        new_inner = _unflatten(new_flat)
        
        # Check overlap
        # Optimization: Only check pairs involving idx?
        # But moving one hex can only affect pairs involving it.
        # So we only need to check penetration for (idx, j) for j != idx.
        
        # Calculate total penetration efficiently?
        # We can just calculate full penetration for simplicity first, 
        # but that's O(N^2). 
        # Let's stick to full for correctness in the SA loop, but it might be slow.
        # Let's try to make it faster.
        
        pen = _total_penetration(new_inner)
        
        if pen > 1e-9:
            # Reject overlap
            continue
            
        new_side = _enclosing_side(_all_vertices(new_inner), theta_cands)
        
        delta = new_side - best_side # We want to minimize best_side? 
        # Actually, we should track the current side and compare to current, 
        # and update best.
        
        # Let's track current_side
        # I missed current_side in the loop.
        
        # Re-evaluating current_side is expensive if we don't store it.
        # Let's store current_side.
        
        # I need to refactor the loop to keep current_side.
        
        pass # Placeholder for logic re-structure
        
    # The above logic was flawed. Let's write the correct SA loop.
    
    current_flat = list(x0)
    current_inner = _unflatten(current_flat)
    current_side = _enclosing_side(_all_vertices(current_inner), theta_cands)
    
    best_flat = list(current_flat)
    best_side = current_side
    
    T = 0.5
    cooling = 0.995
    max_steps = 3000
    
    for step in range(max_steps):
        if time.time() - _t0 > DEADLINE - 5:
            break
            
        idx = random.randint(0, N - 1)
        i = idx * 3
        
        new_flat = list(current_flat)
        # Perturbation size
        scale = 0.5 * T + 0.01
        new_flat[i] += random.gauss(0, scale)
        new_flat[i+1] += random.gauss(0, scale)
        new_flat[i+2] += random.gauss(0, scale * 10)
        
        new_inner = _unflatten(new_flat)
        
        # Check validity
        # To speed up, we can check only pairs with idx.
        # But _total_penetration is simple. Let's see if it's fast enough.
        # 3000 steps * 66 pairs * ~10us/pair = ~2s. Should be ok.
        
        pen = _total_penetration(new_inner)
        
        if pen > 1e-9:
            # Reject
            T *= cooling
            continue
            
        new_side = _enclosing_side(_all_vertices(new_inner), theta_cands)
        
        delta = new_side - current_side
        
        if delta < 0:
            # Improvement
            current_flat = new_flat
            current_inner = new_inner
            current_side = new_side
            
            if new_side < best_side:
                best_side = new_side
                best_flat = list(new_flat)
        else:
            # Worse, accept with probability
            if random.random() < np.exp(-delta / T):
                current_flat = new_flat
                current_inner = new_inner
                current_side = new_side
                # Do not update best if it's worse
                if new_side < best_side:
                    best_side = new_side
                    best_flat = list(new_flat)
                    
        T *= cooling
        
    # Final validation and polishing
    # Ensure best_flat is valid
    best_inner = _unflatten(best_flat)
    if _total_penetration(best_inner) > 1e-9:
        # Fallback to initial
        best_inner = _unflatten(x0)
        best_side = _enclosing_side(_all_vertices(best_inner), theta_cands)
        
    # Calculate final outer hexagon parameters
    V = _all_vertices(best_inner)
    # Use the best angle from theta_cands
    c = V.mean(axis=0)
    best_s = np.inf
    best_th = 0
    for th in theta_cands:
        a = np.deg2rad(th)
        normals = np.column_stack([np.cos(a + np.arange(6) * np.pi / 3.0),
                                   np.sin(a + np.arange(6) * np.pi / 3.0)])
        proj = V @ normals.T
        m = proj.max(axis=0) - normals @ c
        s = (2.0 / np.sqrt(3.0)) * m.max()
        if s < best_s:
            best_s = s
            best_th = th
            
    return best_inner, (float(c[0]), float(c[1])), float(best_s), float(best_th)

