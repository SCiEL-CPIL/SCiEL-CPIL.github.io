import numpy as np
import time
import math
from scipy.spatial import Delaunay, cKDTree

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy(pts):
    """Compute the logarithmic energy of points on the sphere."""
    n = len(pts)
    # Compute pairwise squared distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-300)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # d_ij = sqrt(dist_sq_ij)
    # log(d_ij) = 0.5 * log(dist_sq_ij)
    # Sum_{i<j} log(1/d_ij) = -0.5 * Sum_{i<j} log(d_ij) 
    # Wait, the task says: sum_{i<j} log(1 / |x_i - x_j|)
    # Let d = |x_i - x_j|.
    # Term is log(1/d) = -log(d).
    # Sum_{i<j} -log(d_ij).
    
    # Using dist_sq:
    # log(d_ij) = 0.5 * log(dist_sq_ij)
    # So sum_{i<j} -log(d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    
    # We have the full matrix dist_sq.
    # Sum over i<j is 0.5 * (Sum over all i!=j).
    # So Sum_{i<j} log(dist_sq_ij) = 0.5 * (Sum_{i!=j} log(dist_sq_ij))
    # Since diagonal is 1 (log 1 = 0), Sum_{i!=j} is Sum_{all}.
    
    log_d_sq = np.log(dist_sq)
    total_log_sq = np.sum(log_d_sq)
    
    # Energy = -0.5 * (0.5 * total_log_sq) = -0.25 * total_log_sq
    energy = -0.25 * total_log_sq
    
    return float(energy)

def _laplacian_flow_step(pts, triangle_indices, step_size):
    """
    Perform one step of Laplacian flow.
    The discrete Laplacian of a point x_i in a triangulated mesh is:
    L_i = sum_{j in neighbors(i)} (cot(alpha_ij) + cot(beta_ij)) (x_i - x_j)
    where alpha, beta are angles opposite to edge (i,j) in the two triangles.
    Alternatively, a simpler approximation for energy minimization on sphere
    is to move points away from neighbors weighted by 1/dist^2 or similar.
    
    However, for logarithmic energy, the gradient is proportional to 1/d^2 (x_i - x_j).
    The Laplacian flow with cotangent weights approximates the gradient of the Dirichlet energy,
    but for logarithmic energy, the "force" is 1/d.
    
    Let's use a specific force law derived from the gradient:
    F_i = - sum_{j} (x_i - x_j) / ||x_i - x_j||^2
    This is exactly the negative gradient of 0.5 * sum log(d^2).
    
    To make it "Laplacian-like" and stable, we use the triangle structure to define neighbors,
    but the force is computed from all pairs? No, that's O(N^2).
    The prompt suggests approximating energy by adjacent mesh edges.
    If we only use edges, we lose the global log term.
    
    Let's stick to the exact gradient but use the triangulation for a "relaxation" step
    that respects topology, or just use the gradient with adaptive step size.
    
    Actually, the "Laplacian Flow" usually refers to minimizing the Dirichlet energy 
    (sum of squared edge lengths). Log energy is different.
    
    Let's implement a Hybrid:
    1. Compute exact gradient (O(N^2)).
    2. Use the Delaunay triangulation to identify "close" neighbors for a local repulsion boost
       or to detect instability.
    3. Apply gradient descent with a step size adapted to the local geometry.
    
    Wait, the directive says: "approximated by a sum over adjacent mesh edges... allowing use of discrete differential geometry techniques like Laplacian flow".
    If I approximate E_log by E_edge = sum_edges log(1/L_e), then the gradient is local.
    Let's try this local approximation first. It is O(N) per step (if neighbors are precomputed).
    
    E_local = sum_{(i,j) in edges} log(1/d_ij)
    Grad_i = - sum_{j in nbrs(i)} (x_i - x_j) / d_ij^2
    
    This is much faster. Let's see if it converges to a good global minimum.
    The global minimum of log energy is likely close to the global minimum of edge-length log energy
    if the triangulation is Delaunay (which maximizes minimum angle).
    """
    n = len(pts)
    # Compute forces
    forces = np.zeros_like(pts)
    
    # Get edges from triangulation
    # triangle_indices is (M, 3)
    edges = set()
    for tri in triangle_indices:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[2], tri[0]))))
        
    for i, j in edges:
        diff = pts[i] - pts[j]
        d_sq = np.dot(diff, diff)
        if d_sq < 1e-12:
            continue
        # Gradient of log(1/d) wrt x_i is - (x_i - x_j) / d^2
        # We want to minimize energy, so we move in direction of -grad.
        # grad_E = sum - (x_i - x_j)/d^2
        # -grad_E = sum (x_i - x_j)/d^2
        # So force is (x_i - x_j) / d^2 ?
        # Let's check: E = -log(d). dE/dx_i = -(1/d) * (x_i-x_j)/d = -(x_i-x_j)/d^2.
        # Descent step: x_i -= step * grad = x_i + step * (x_i-x_j)/d^2.
        # This moves x_i away from x_j. Correct.
        
        force_vec = diff / d_sq
        forces[i] += force_vec
        forces[j] -= force_vec # By Newton's 3rd law / antisymmetry
        
    # Project forces to tangent space
    # f_tangent = f - (f . x) x
    proj = np.sum(forces * pts, axis=1, keepdims=True)
    forces_tangent = forces - proj * pts
    
    # Update
    new_pts = pts + step_size * forces_tangent
    
    # Normalize to sphere
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    
    return new_pts

def _retriangulate(pts):
    """
    Create a Delaunay triangulation of the points projected onto the sphere.
    Since Delaunay in 3D is for convex hull, and points are on sphere,
    we can use the convex hull faces.
    scipy.spatial.Delaunay works on 3D points and returns simplices (tetrahedra).
    For points on a sphere, the convex hull is the set of points.
    The faces of the convex hull correspond to the triangulation on the sphere.
    """
    try:
        hull = Delaunay(pts)
        # hull.simplices are tetrahedra. We need the boundary faces.
        # A face is on the boundary if it appears in only one tetrahedron.
        faces = []
        face_count = {}
        for tet in hull.simplices:
            # 4 faces per tet
            f0 = tuple(sorted((tet[1], tet[2], tet[3])))
            f1 = tuple(sorted((tet[0], tet[2], tet[3])))
            f2 = tuple(sorted((tet[0], tet[1], tet[3])))
            f3 = tuple(sorted((tet[0], tet[1], tet[2])))
            for f in [f0, f1, f2, f3]:
                face_count[f] = face_count.get(f, 0) + 1
        
        for f, count in face_count.items():
            if count == 1:
                faces.append(f)
        
        if len(faces) == 0:
            return None
            
        return np.array(faces)
    except Exception:
        return None

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_laplacian(pts, max_iter=500, initial_step=0.01, decay=0.999):
    """
    Optimize using Laplacian flow on the Delaunay triangulation.
    """
    current_pts = pts.copy()
    current_energy = _log_energy(current_pts)
    
    # Initial triangulation
    tris = _retriangulate(current_pts)
    if tris is None:
        return current_pts, current_energy
        
    step_size = initial_step
    
    for k in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        new_pts = _laplacian_flow_step(current_pts, tris, step_size)
        
        # Check for validity
        if not np.all(np.isfinite(new_pts)):
            step_size *= 0.5
            continue
            
        # Check if points got too close (collision)
        # This is a heuristic. If energy increases significantly, we might be unstable.
        new_energy = _log_energy(new_pts)
        
        if new_energy < current_energy:
            current_pts = new_pts
            current_energy = new_energy
            # Accept step, maybe increase step size slightly if we are converging?
            # Or keep constant.
        else:
            # Reject step, reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
        # Retriangulate periodically (e.g., every 10 steps) or if topology changes
        # For simplicity, every 20 steps.
        if k % 20 == 0 and k > 0:
            new_tris = _retriangulate(current_pts)
            if new_tris is not None:
                tris = new_tris
                
    return current_pts, current_energy

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci
    fib = _fibonacci_sphere(N)
    inits.append(fib)
    
    # 2. Random
    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    rand1 = np.column_stack([x, y, z])
    inits.append(rand1)
    
    # 3. Another random
    np.random.seed(123)
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    rand2 = np.column_stack([x, y, z])
    inits.append(rand2)
    
    # 4. Icosahedron-like subdivision (approx)
    # Just use a rotated Fibonacci
    rot_fib = fib.copy()
    # Rotate around z
    angle = np.pi / 5
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rot_fib = rot_fib @ R.T
    inits.append(rot_fib)

    for init in inits:
        if time.time() > DEADLINE - 20:
            break
            
        try:
            # Run Laplacian optimization
            opt_pts, e = _optimize_laplacian(init, max_iter=1000, initial_step=0.05, decay=0.999)
            
            # Verify
            e_check = _log_energy(opt_pts)
            
            if e_check < best_energy:
                best_energy = e_check
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
            
    # If we have time, do a final polish with a very small step size on the best result
    if best_pts is not None and time.time() < DEADLINE - 10:
        try:
            # Final polish
            tris = _retriangulate(best_pts)
            if tris is not None:
                # Use a smaller step size for final convergence
                final_pts, final_e = _optimize_laplacian(best_pts, max_iter=500, initial_step=0.005, decay=0.99)
                if final_e < best_energy:
                    best_energy = final_e
                    best_pts = final_pts
        except Exception:
            pass

    if best_pts is None:
        best_pts = fib

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
