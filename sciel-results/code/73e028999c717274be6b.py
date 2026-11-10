"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(pts):
    """Compute covering radius in radians. Returns (radius, worst_facet_index, worst_point)."""
    try:
        hull = ConvexHull(pts)
        n_sites = len(pts)
        max_ang = 0.0
        worst_facet = -1
        worst_pt = None
        
        # Vectorized computation of circumcenters for all facets
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        # Normal vectors to each triangle
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return 0.0, -1, None
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        # Ensure normals point outward (away from origin)
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        # Compute angular distance from each circumcenter to nearest site
        dots = np.clip(pts @ normals.T, -1.0, 1.0)
        dists = np.arccos(dots)  # shape: (n_sites, n_facets)
        min_dists = np.min(dists, axis=0)  # shape: (n_facets,)
        
        max_idx = np.argmax(min_dists)
        max_ang = min_dists[max_idx]
        worst_facet = np.where(valid)[0][max_idx]
        worst_pt = normals[max_idx].copy()
        
        return max_ang, worst_facet, worst_pt
    except Exception:
        return np.pi, -1, None

def _soft_covering_radius_and_grad(pts, beta=100.0):
    """
    Compute the soft-max covering radius and its gradient.
    Returns (value, grad).
    """
    try:
        hull = ConvexHull(pts)
        tris = hull.simplices
        n_facets = len(tris)
        
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        # Compute normals
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return 0.0, np.zeros_like(pts)
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        # Ensure normals point outward
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        # Map valid facet indices to original indices
        valid_indices = np.where(valid)[0]
        
        # Compute distances from sites to facet circumcenters
        # dots: (N, n_valid_facets)
        dots = np.clip(pts @ normals.T, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # min_dists: (n_valid_facets,)
        min_dists = np.min(dists, axis=0)
        
        # Soft-max weights
        # L = 1/beta * log(sum(exp(beta * min_dists)))
        # To avoid overflow, subtract max
        max_d = np.max(min_dists)
        exps = np.exp(beta * (min_dists - max_d))
        sum_exps = np.sum(exps)
        weights = exps / sum_exps  # w_i
        
        value = max_d + (1.0 / beta) * np.log(sum_exps)
        
        # Gradient calculation
        # grad = sum_i w_i * d(theta_i)/d(p_k)
        # theta_i = arccos(p_{n(i)} . n_i)
        # d(theta_i)/d(p_k) = -1/sqrt(1-dot^2) * (delta_{k,n(i)} n_i + p_{n(i)} dn_i/dp_k)
        
        grad = np.zeros_like(pts)
        
        # Identify nearest site for each facet
        nearest_sites = np.argmin(dists, axis=0) # (n_valid_facets,)
        
        # Precompute 1/sqrt(1-dot^2) for the nearest sites
        nearest_dots = dots[nearest_sites, np.arange(n_facets) if len(exps)==n_facets else np.arange(len(exps))]
        # Note: dots shape is (N, n_valid). nearest_sites corresponds to columns.
        # dots[nearest_sites, j] is the dot product for facet j with its nearest site.
        # Let's be careful with indices.
        # dists is (N, n_valid). min_dists is (n_valid,).
        # nearest_sites[j] is the site index for facet j.
        # nearest_dots[j] = dots[nearest_sites[j], j]
        
        # Compute derivative of arccos: -1 / sqrt(1 - x^2)
        # Clip to avoid division by zero or NaNs
        safe_dots = np.clip(nearest_dots, -0.999999, 0.999999)
        inv_sqrt = -1.0 / np.sqrt(1.0 - safe_dots**2) # (n_valid,)
        
        # Term 1: delta_{k, n(i)} n_i
        # For each facet i, add w_i * inv_sqrt_i * n_i to grad[nearest_sites[i]]
        # Vectorized update:
        # grad[nearest_sites[i]] += weights[i] * inv_sqrt[i] * normals[i]
        np.add.at(grad, nearest_sites, weights[:, np.newaxis] * inv_sqrt[:, np.newaxis] * normals)
        
        # Term 2: p_{n(i)} * dn_i/dp_k
        # dn_i/dp_k is non-zero only if p_k is a vertex of facet i.
        # n_i = (b-a) x (c-a) / ||...||
        # This is complex to vectorize perfectly for all vertices at once,
        # but we can iterate over facets if N is small (130) and facets ~ 2N.
        # 260 facets * 3 vertices = 780 updates. This is fast.
        
        # We need dn/dp for each vertex of each facet.
        # Let u = b-a, v = c-a. n = u x v / ||u x v||.
        # dn/du = (v (u x v) - u ((u x v) . v)) / ||u x v||^2 ... wait, standard vector calculus.
        # Actually, we can use the fact that n is the circumcenter.
        # A more robust way: use the explicit formula for circumcenter C of triangle A,B,C.
        # C = (|A|^2 (B x C) + |B|^2 (C x A) + |C|^2 (A x B)) / (2 (A x (B x C)))
        # Since points are on unit sphere, |A|^2=1.
        # C = (B x C + |B|^2 C x A + |C|^2 A x B) / (2 A . (B x C))
        # This is getting messy. Let's use the property that n_i is the normal.
        # The gradient of the normal with respect to vertices can be derived.
        # However, for a "soft" max, the contribution from the normal derivative is often smaller 
        # or can be approximated. But let's try to be precise.
        
        # Alternative: Just use the term 1 (nearest site movement) and ignore term 2?
        # Term 2 accounts for how the circumcenter moves when vertices move.
        # If we ignore it, the gradient is incomplete.
        
        # Let's compute dn/dp_k analytically for the 3 vertices of each facet.
        # n = (b-a) x (c-a) / norm
        # Let w = (b-a) x (c-a). n = w / |w|.
        # dn = (dw |w| - w (dw . w) / |w|) / |w|^2 = (dw |w|^2 - w (dw . w)) / |w|^3
        # dw/da = -(c-a) x (c-a) - (b-a) x (c) ... wait.
        # w = b x c - b x a - a x c + a x a = b x c - b x a - a x c.
        # dw/db = c x b + a x b = (a - c) x b ? No.
        # d/db (b x c) = 0 (if c fixed) ... no, b is variable.
        # d/db (b x c) = 0? No. (b x c) derivative wrt b is not standard.
        # Let's use: d/dx (x x y) = 0? No.
        # Identity: d/dx (x x y) = (dx x y) + (x x dy). If y is const, dx x y.
        # So dw/db = (db x c) - (db x a) = db x (c - a).
        # dw/dc = (b x dc) - (dc x a) = (b + a) x dc? No.
        # d/dc (b x c) = b x dc. d/dc (-a x c) = -a x dc.
        # dw/dc = (b - a) x dc.
        # dw/da = d/da (-b x a) - d/da (a x c) = -b x da - da x c = (c - b) x da.
        
        # So:
        # dw/da = (c - b) x da
        # dw/db = (c - a) x db
        # dw/dc = (b - a) x dc
        
        # For each facet i, we have vertices a,b,c (indices ia, ib, ic).
        # We need to add to grad[ia] the contribution from dn/da.
        # Contribution to grad[ia] from term 2:
        # p_{n(i)} * (w_i * inv_sqrt_i) * (dn_i/dp_{ia})
        
        # This is 260 facets * 3 vertices = 780 cross products and dot products.
        # Doable in a loop.
        
        for j in range(len(weights)):
            if weights[j] < 1e-10: continue
            
            facet = valid_indices[j]
            ia, ib, ic = tris[facet]
            
            pa = pts[ia]
            pb = pts[ib]
            pc = pts[ic]
            
            n_i = normals[j]
            w_j = weights[j] * inv_sqrt[j]
            p_nearest = pts[nearest_sites[j]]
            
            # Compute dw/dp for each vertex
            # w = (pb - pa) x (pc - pa)
            # dw/da = (pc - pb) x da  -> (pc - pb)
            # dw/db = (pc - pa) x db  -> (pc - pa)
            # dw/dc = (pb - pa) x dc  -> (pb - pa)
            
            # Let u = pb - pa, v = pc - pa. w = u x v.
            # dw/da = (v - u) x da? No.
            # w = (b-a) x (c-a).
            # d/da: (0) x (c-a) + (b-a) x (-da) = - (b-a) x da.
            # Wait, d/da (b-a) = -da. d/da (c-a) = -da.
            # d/da ((b-a) x (c-a)) = (-da) x (c-a) + (b-a) x (-da) = -da x (c-a) - (b-a) x da.
            # Using a x b = - b x a:
            # = (c-a) x da - da x (b-a) = (c-a) x da + (b-a) x da = (b+c-2a) x da.
            # Let's check with u x v. u=b-a, v=c-a.
            # d/da (u x v) = du x v + u x dv = (-da) x v + u x (-da) = -da x v - u x da.
            # = v x da - u x da = (v-u) x da = (c-b) x da.
            # So dw/da = (c-b) x da.
            
            # Similarly:
            # dw/db = db x v + u x 0 = db x (c-a) = (c-a) x db? No, db x (c-a).
            # dw/dc = 0 x dc + u x dc = (b-a) x dc.
            
            # We need dn = (dw |w|^2 - w (dw . w)) / |w|^3
            # Let N = |w|^2. dn = (dw N - w (dw . w)) / sqrt(N)^3
            
            # We will compute the vector part: p_nearest * (dn)
            # And add to grad[vertex]
            
            w_vec = np.cross(pb - pa, pc - pa)
            N_val = np.dot(w_vec, w_vec)
            if N_val < 1e-12: continue
            N_sqrt = np.sqrt(N_val)
            N_cubed = N_val * N_sqrt
            
            # For vertex a
            dw_da = np.cross(pc - pb, np.ones(3)) # This is wrong, da is a vector variable.
            # We need the linear operator.
            # dn/da is a 3x3 matrix.
            # Let's just compute the directional derivative? No, we need the full gradient.
            # grad[ia] += p_nearest * (w_j * inv_sqrt_j) * (dn/da)
            
            # Let's compute dn/da as a matrix M_a such that dn = M_a da.
            # dw/da = (c-b) x da. Let u = c-b.
            # dw = u x da.
            # dw . w = (u x da) . w = da . (w x u).
            # So dw . w is linear in da: da . (w x u).
            # Let v_w = w x u. Then (dw . w) = da . v_w.
            # dn = ( (u x da) N_val - w (da . v_w) ) / N_cubed
            # dn = (1/N_cubed) [ N_val (u x da) - w (v_w . da) ]
            # We can write this as M_a da where M_a is a matrix.
            # u x da = - da x u = - (skew(u) da) = (skew(-u) da)?
            # u x da = (da x u)^T? No.
            # u x da = - (u x da)^T ...
            # u x da = [u]_x da.
            # So term 1: N_val [u]_x da.
            # Term 2: - w (v_w . da) = - w v_w^T da.
            # M_a = (1/N_cubed) ( N_val [u]_x - w v_w^T )
            
            u_a = pc - pb
            v_w_a = np.cross(w_vec, u_a)
            M_a = (N_val * np.cross(np.eye(3), u_a) - np.outer(w_vec, v_w_a)) / N_cubed
            # Note: [u]_x is the skew-symmetric matrix for cross product with u.
            # np.cross(np.eye(3), u) gives the matrix such that M @ x = u @ x? No.
            # Standard skew:
            # [[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]]
            # Let's construct it manually to be safe.
            ux, uy, uz = u_a
            skew_u = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
            M_a = (N_val * skew_u - np.outer(w_vec, v_w_a)) / N_cubed
            
            grad[ia] += p_nearest * (w_j) * M_a
            
            # For vertex b
            # dw/db = (c-a) x db. Let u = c-a.
            u_b = pc - pa
            v_w_b = np.cross(w_vec, u_b)
            ux, uy, uz = u_b
            skew_u = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
            M_b = (N_val * skew_u - np.outer(w_vec, v_w_b)) / N_cubed
            grad[ib] += p_nearest * (w_j) * M_b
            
            # For vertex c
            # dw/dc = (b-a) x dc. Let u = b-a.
            u_c = pb - pa
            v_w_c = np.cross(w_vec, u_c)
            ux, uy, uz = u_c
            skew_u = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
            M_c = (N_val * skew_u - np.outer(w_vec, v_w_c)) / N_cubed
            grad[ic] += p_nearest * (w_j) * M_c
        
        # Project gradient to tangent space
        for i in range(len(pts)):
            radial = np.dot(grad[i], pts[i]) * pts[i]
            grad[i] -= radial
            
        return value, grad
        
    except Exception:
        return np.pi, np.zeros_like(pts)

def _fibonacci_start():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_start(rng):
    """Generate random initial points on the sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _lloyd_step(pts, n_iter=30):
    """Perform Lloyd's algorithm iterations to minimize covering radius."""
    pts = pts.copy()
    for _ in range(n_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Vectorized circumcenter computation
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            break
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        circumcenters = normals
        
        new_pts = pts.copy()
        for i in range(len(pts)):
            dists = np.linalg.norm(circumcenters - pts[i], axis=1)
            k = min(5, len(circumcenters))
            idx = np.argsort(dists)[:k]
            centroid = circumcenters[idx].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 1e-10:
                centroid = centroid / nc
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
                new_pts[i] /= np.linalg.norm(new_pts[i])
        
        pts = new_pts
    
    return pts

def _local_optimize(pts, n_iter=100, base_alpha=0.05):
    """Local optimization targeting the worst facet with adaptive step size."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        a_idx, b_idx, c_idx = tri[0], tri[1], tri[2]
        
        # Adaptive step size: reduce if improvement is small
        alpha = base_alpha * (0.5 ** (it // 20))
        
        for idx in [a_idx, b_idx, c_idx]:
            new_pt = pts[idx] + alpha * (wpt - pts[idx])
            new_pt /= np.linalg.norm(new_pt)
            pts[idx] = new_pt
    
    return pts

def _polish(pts, n_iter=50):
    """High-precision polish phase with adaptive step-size reduction."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        
        # Try multiple step sizes
        for alpha_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
            alpha = 0.01 * alpha_mult
            trial = pts.copy()
            for idx in tri:
                new_pt = trial[idx] + alpha * (wpt - trial[idx])
                new_pt /= np.linalg.norm(new_pt)
                trial[idx] = new_pt
            
            r_trial, _, _ = _covering_radius(trial)
            if r_trial < radius:
                pts = trial
                break
    
    return pts

def _gradient_descent_step(pts, n_iter=100, lr=0.01, beta=100.0):
    """Use analytical gradient descent on the soft-max surrogate to improve covering radius."""
    pts = pts.copy()
    
    for it in range(n_iter):
        # Compute current soft-max value and gradient
        current_val, grad = _soft_covering_radius_and_grad(pts, beta)
        
        if np.any(np.isnan(grad)):
            break
            
        # Project gradient to tangent space (already done in _soft_covering_radius_and_grad, but safe to re-do)
        for i in range(len(pts)):
            radial = np.dot(grad[i], pts[i]) * pts[i]
            grad[i] -= radial
        
        # Normalize gradient to avoid huge steps if gradient magnitude varies
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Adaptive learning rate
        lr_eff = lr * (0.95 ** (it // 10))
        
        # Update
        new_pts = pts - lr_eff * grad
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check if improved
        new_val, _ = _soft_covering_radius_and_grad(new_pts, beta)
        if new_val < current_val:
            pts = new_pts
        else:
            # Try smaller step
            new_pts = pts - lr_eff * 0.1 * grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_val, _ = _soft_covering_radius_and_grad(new_pts, beta)
            if new_val < current_val:
                pts = new_pts
            else:
                break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate multiple initial configurations
    best_pts = None
    best_radius = np.pi
    
    # Fibonacci start
    pts_fib = _fibonacci_start()
    r_fib, _, _ = _covering_radius(pts_fib)
    if r_fib < best_radius:
        best_radius = r_fib
        best_pts = pts_fib.copy()
    
    # Several random starts
    for _ in range(8):
        if time.time() > deadline - 60:
            break
        pts_rand = _random_start(rng)
        r_rand, _, _ = _covering_radius(pts_rand)
        if r_rand < best_radius:
            best_radius = r_rand
            best_pts = pts_rand.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_start()
        best_radius, _, _ = _covering_radius(best_pts)
    
    pts = best_pts.copy()
    
    # Phase 2: Lloyd's algorithm for global improvement
    while time.time() < deadline - 40:
        pts_lloyd = _lloyd_step(pts, n_iter=20)
        r_lloyd, _, _ = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
            pts = pts_lloyd.copy()
        else:
            pts_lloyd = _lloyd_step(pts, n_iter=5)
            r_lloyd, _, _ = _covering_radius(pts_lloyd)
            if r_lloyd < best_radius:
                best_radius = r_lloyd
                best_pts = pts_lloyd.copy()
                pts = pts_lloyd.copy()
            else:
                break
    
    # Phase 3: Local optimization targeting worst facets
    while time.time() < deadline - 15:
        pts_local = _local_optimize(pts, n_iter=50)
        r_local, _, _ = _covering_radius(pts_local)
        if r_local < best_radius:
            best_radius = r_local
            best_pts = pts_local.copy()
            pts = pts_local.copy()
        else:
            pts_local = _local_optimize(pts, n_iter=20)
            r_local, _, _ = _covering_radius(pts_local)
            if r_local < best_radius:
                best_radius = r_local
                best_pts = pts_local.copy()
                pts = pts_local.copy()
            else:
                break
                
    # Phase 3.5: Gradient descent on soft-max surrogate
    if time.time() < deadline - 10:
        pts_grad = _gradient_descent_step(pts, n_iter=100, lr=0.01, beta=100.0)
        r_grad, _, _ = _covering_radius(pts_grad)
        if r_grad < best_radius:
            best_radius = r_grad
            best_pts = pts_grad.copy()
            pts = pts_grad.copy()
            
    # Phase 4: High-precision polish
    while time.time() < deadline - 5:
        pts_polished = _polish(pts, n_iter=30)
        r_polished, _, _ = _covering_radius(pts_polished)
        if r_polished < best_radius:
            best_radius = r_polished
            best_pts = pts_polished.copy()
            pts = pts_polished.copy()
        else:
            break
    
    # Ensure output is valid
    pts_out = best_pts.copy()
    norms = np.linalg.norm(pts_out, axis=1)
    pts_out = pts_out / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_out[i] - pts_out[j]) < 1e-8:
                pts_out[j] += np.random.randn(3) * 0.01
                pts_out[j] /= np.linalg.norm(pts_out[j])
    
    return pts_out
# EVOLVE-BLOCK-END
