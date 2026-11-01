<<<<<<< SEARCH
def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _optimize_sphere(pts0, max_iters=2000, tol=1e-10):
    """
    Riemannian Gradient Descent with Backtracking Line Search.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    for it in range(max_iters):
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = _retract(pts, alpha * g_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * g_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-12:
                break
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Slowly decay learning rate
        lr = max(0.001, lr * 0.999)
        
    return best_pts, best_energy
=======
def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _fibonacci_spiral():
    """Generate points using Fibonacci spiral on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _icosahedron_subdivide():
    """Generate points by subdividing icosahedron."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide each face multiple times
    def subdivide(verts, faces, depth):
        if depth == 0:
            return verts
        new_faces = []
        mid_cache = {}
        for f in faces:
            a, b, c = verts[f]
            mids = []
            for i, j in [(0,1), (1,2), (2,0)]:
                key = tuple(sorted((f[i], f[j])))
                if key not in mid_cache:
                    mid = (verts[f[i]] + verts[f[j]]) / 2
                    mid = mid / np.linalg.norm(mid)
                    mid_cache[key] = len(verts)
                    verts = np.vstack([verts, mid])
                mids.append(mid_cache[key])
            new_faces.extend([
                [f[0], mids[0], mids[2]],
                [f[1], mids[1], mids[0]],
                [f[2], mids[2], mids[1]],
                [mids[0], mids[1], mids[2]]
            ])
        return subdivide(verts, new_faces, depth - 1)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide 3 times: 20 * 4^3 = 1280 faces, but we need 100 points
    # Use 2 subdivisions: 20 * 4^2 = 320 points, then select 100
    verts, _ = subdivide(verts.copy(), faces, 2)
    
    # Select 100 points uniformly from the subdivided vertices
    rng = np.random.default_rng(123)
    indices = rng.choice(len(verts), N, replace=False)
    return verts[indices]

def _perturbed_points(base_pts, sigma=0.1, seed=42):
    """Add random perturbation to base points."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(base_pts.shape) * sigma
    pts = base_pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _polish_lbfgs(pts0, max_iter=200):
    """Polish using L-BFGS-B."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts0.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-15})
        if res.success or res.fun < _energy_and_grad(pts0)[0]:
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            return x_opt
    except Exception:
        pass
    return pts0
>>>>>>> REPLACE

