"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
from scipy.optimize import minimize

def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    if n < 2:
        return 0.5
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    # Wall distance
    # min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    # x in [-0.5, 0.5] => 0.5 - x in [0, 1], x + 0.5 in [0, 1]
    # min is min(0.5 - |x|, 0.5 - |y|)
    # Actually min(0.5 - x, x + 0.5) = 0.5 - |x|
    rw = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])).min()
    return min(rp, rw)

def _init_hexagonal():
    """Hexagonal lattice of ~50 points inside the square."""
    pts = []
    s = 0.14
    xs = np.arange(-0.5 + s / 2, 0.5, s)
    for row, y in enumerate(np.arange(-0.5 + s / 2, 0.5, s * np.sqrt(3) / 2)):
        offset = (s / 2) if row % 2 else 0.0
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
    pts = np.array(pts)
    if len(pts) >= N:
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists)[:N]
        return pts[idx].copy()
    else:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        g = np.array([(x, y) for y in ticks for x in ticks])
        return g[:N].copy()

def _expand_radius(c, R0, R1, step, deadline, t0):
    """Incrementally grow radius, resolving collisions by pushing apart."""
    c = c.copy()
    R = R0
    n = len(c)
    while R < R1:
        if (time.time() - t0) > deadline - 10:
            break
        for _ in range(200):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    diff = c[i] - c[j]
                    d = np.linalg.norm(diff)
                    if d < 2 * R - 1e-12:
                        if d < 1e-12:
                            diff = np.array([1e-6, 1e-6])
                            d = np.linalg.norm(diff)
                        push = (2 * R - d) / d * 0.5
                        c[i] += push * diff
                        c[j] -= push * diff
                        moved = True
            for i in range(n):
                c[i, 0] = np.clip(c[i, 0], -0.5 + R, 0.5 - R)
                c[i, 1] = np.clip(c[i, 1], -0.5 + R, 0.5 - R)
            if not moved:
                break
        R += step
    return c

def _delaunay_refine(c, deadline, t0):
    """
    Use Delaunay triangulation to identify contact graph.
    Solve a linear system to maximize the minimum distance among active contacts.
    """
    n = len(c)
    if n < 3:
        return c
    
    current_r = _radius(c)
    if current_r <= 0:
        return c
        
    # We want to increase r.
    # Identify pairs that are "close" (within some factor of 2*r)
    # Or just use Delaunay neighbors.
    
    try:
        tri = scipy.spatial.Delaunay(c)
    except Exception:
        return c
        
    # Build adjacency list from Delaunay
    # tri.simplices gives triangles.
    adj = {i: set() for i in range(n)}
    for simplex in tri.simplices:
        i, j, k = simplex
        adj[i].add(j); adj[i].add(k)
        adj[j].add(i); adj[j].add(k)
        adj[k].add(i); adj[k].add(j)
        
    # Also add wall contacts if close to wall
    # But let's stick to inter-particle contacts for the linear solve first.
    # We can handle walls in the post-processing or add them to the system.
    
    # Let's define the active set as all Delaunay edges.
    # We want to find displacement dx such that:
    # 1. For all active pairs (i,j), the distance increases (or stays same)
    # 2. We maximize the "average" increase or the "minimum" increase.
    
    # Linearization:
    # d_{ij} \approx ||c_i - c_j + dx_i - dx_j||
    # If c_i - c_j = u, then ||u + du|| \approx ||u|| + (u dot du) / ||u||
    # We want to maximize min( (u_ij dot du_ij) / ||u_ij|| )
    # This is a linear programming problem (maximize t subject to A t <= b).
    # But we can use a simpler heuristic:
    # Minimize the squared displacement norm subject to:
    # (u_ij dot du_ij) / ||u_ij|| >= delta_r
    # where delta_r is a small step.
    
    # Actually, a better approach for "rigid body" expansion:
    # We want to move points apart.
    # Let's try to solve for dx that satisfies:
    # For all active edges (i,j): (c_i - c_j) dot (dx_i - dx_j) >= 2 * current_r * epsilon
    # This ensures the distance increases by at least 2*current_r*epsilon.
    
    # Variables: dx (2n)
    # Constraints:
    # A dx >= b
    # Where row for (i,j) is: (c_i - c_j) in cols 2i, 2i+1 and -(c_i - c_j) in cols 2j, 2j+1
    # b = 2 * current_r * epsilon
    
    # To make it well-posed, we minimize ||dx||^2 subject to constraints.
    # This is a QP. We can solve it using scipy.optimize.linprog (if we relax) or just use a simple iterative projection.
    
    # Given time constraints, let's use a simpler "Repulsive Force" method guided by Delaunay.
    # Apply force only along Delaunay edges.
    
    c_new = c.copy()
    steps = 50
    for _ in range(steps):
        if (time.time() - t0) > deadline - 5:
            break
            
        forces = np.zeros_like(c_new)
        
        # Recalculate Delaunay? It might be expensive. Let's use the previous one or just check distances.
        # For efficiency, let's just check all pairs but weight by Delaunay adjacency if available?
        # No, let's just do a standard repulsion but maybe smarter.
        
        # Actually, the "structural change" requested is using the contact graph.
        # Let's apply repulsive forces ONLY between Delaunay neighbors.
        # This prevents "long range" interference that might destabilize local structure.
        
        # Rebuild Delaunay every few steps?
        # For now, let's just use the static graph from the start of this function.
        # This might fail if topology changes drastically.
        
        for i in range(n):
            for j in adj[i]:
                if i < j: # avoid double counting
                    diff = c_new[i] - c_new[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-6:
                        continue
                    # Repulsion force proportional to 1/d or constant?
                    # We want to increase d.
                    # Force magnitude:
                    # If we assume we want to increase r by dr, and d is current distance.
                    # Ideal force is to push them to 2*r.
                    # If d < 2*current_r (which shouldn't happen if valid), push hard.
                    # If d >= 2*current_r, we still want to push to increase r.
                    
                    # Let's use a spring-like force: F = k * (2*target_r - d)
                    # But we don't know target_r.
                    # Let's just push them apart slightly.
                    
                    # A better metric: We want to maximize the minimum distance.
                    # Gradient of min distance is dominated by the closest pairs.
                    # Delaunay edges are likely the closest.
                    
                    # Apply force:
                    # f = (diff / d) * strength
                    strength = 1.0 # constant push
                    # Damping?
                    
                    f = (diff / d) * strength
                    forces[i] += f
                    forces[j] -= f
        
        # Wall forces
        # Push away from walls if close
        for i in range(n):
            # x
            if c_new[i, 0] < -0.5 + current_r:
                forces[i, 0] += ( -0.5 + current_r - c_new[i, 0]) * 10.0
            elif c_new[i, 0] > 0.5 - current_r:
                forces[i, 0] -= (c_new[i, 0] - (0.5 - current_r)) * 10.0
            
            # y
            if c_new[i, 1] < -0.5 + current_r:
                forces[i, 1] += ( -0.5 + current_r - c_new[i, 1]) * 10.0
            elif c_new[i, 1] > 0.5 - current_r:
                forces[i, 1] -= (c_new[i, 1] - (0.5 - current_r)) * 10.0

        # Update
        # Use a small step size to avoid instability
        step_size = 0.01
        c_new += forces * step_size
        
        # Clip to square
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Check if radius improved
        r_new = _radius(c_new)
        if r_new > current_r:
            c = c_new
            current_r = r_new
        else:
            # If it got worse, revert or damp
            # For simplicity, keep c_new but maybe reduce step?
            # Let's just break if we stagnate?
            # Or try a smaller step.
            pass
            
    return c

def _slsqp_polish(c, t0, deadline):
    """SLSQP optimization to maximize r subject to constraints."""
    n = c.shape[0]
    x0 = np.concatenate([c.flatten(), [_radius(c)]])
    
    def obj(x):
        return -x[-1]

    def grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    iu = np.triu_indices(n, k=1)
    # Use all pair constraints? 1225 constraints.
    # SLSQP might struggle.
    # Let's try to use only the "active" ones (closest pairs) to speed up.
    # But active set changes.
    # For polish, let's just use all. If it's too slow, we'll catch the exception.
    
    constraints = []
    for i, j in zip(iu[0], iu[1]):
        def con(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            d = np.linalg.norm(ci - cj)
            return r - d / 2.0
        constraints.append({'type': 'ineq', 'fun': con})
    for i in range(n):
        def con(x, i=i):
            ci = x[2*i:2*i+2]
            r = x[-1]
            w = np.min([0.5 - ci[0], ci[0] + 0.5, 0.5 - ci[1], ci[1] + 0.5])
            return r - w
        constraints.append({'type': 'ineq', 'fun': con})

    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.0, 0.5)]
    
    try:
        res = minimize(obj, x0, method='SLSQP', jac=grad, bounds=bounds,
                       constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
        if res.success:
            c_new = res.x[:2*n].reshape(n, 2)
            r_new = _radius(c_new)
            r_old = _radius(c)
            if r_new > r_old:
                return c_new
    except Exception:
        pass
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    # 1. Init
    c = _init_hexagonal()
    
    # 2. Expand
    c = _expand_radius(c, 0.05, 0.072, 0.001, deadline, t0)
    
    # 3. Delaunay Refine (New Strategy)
    if (time.time() - t0) < deadline - 30:
        c = _delaunay_refine(c, deadline, t0)
        
    # 4. Polish
    if (time.time() - t0) < deadline - 10:
        c = _slsqp_polish(c, t0, deadline)
        
    # 5. Safety
    c = np.clip(c, -0.5 + 1e-15, 0.5 - 1e-15)
    if not np.all(np.isfinite(c)) or _radius(c) <= 0:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return c
# EVOLVE-BLOCK-END
