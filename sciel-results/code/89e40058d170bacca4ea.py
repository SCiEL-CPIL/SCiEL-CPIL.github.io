import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, LinearConstraint
from scipy.spatial.transform import Rotation

N = 14
D = 3
DEADLINE = 140.0  # Safety margin for 150s limit

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    max_d = np.max(sq_dists)
    return max_d / min_d

def objective_direct(x):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return 1e10
    max_d = np.max(sq_dists)
    # We want to minimize max/min. 
    # Using log ratio is smooth if min/max > 0.
    # But max/min is not differentiable at min=0.
    # Let's use a smooth approximation or just optimize log(max) - log(min)
    # To handle non-differentiability, we can use a soft version or just rely on global search
    # and a final local polish with a smooth objective.
    # For DE, a robust objective is better.
    # Let's use: log(soft_max) - log(soft_min)
    # Or simply: (max - min) + penalty? No, ratio is the goal.
    # Let's use: max_d / min_d + 1e-6 * variance(sq_dists) ?
    # Actually, minimizing log(max) - log(min) is equivalent to minimizing max/min.
    # Let's use a smoothed version for the local step, but for DE, 
    # the non-smoothness is less of an issue as it's derivative-free.
    # However, if min_d is very small, the gradient is huge.
    # Let's return max_d / min_d.
    return max_d / min_d

def objective_smooth(x, tau=20.0):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Return log ratio (equivalent to log of soft ratio)
    return np.log(soft_max) - np.log(soft_min)

def generate_tetrahedron():
    # Regular tetrahedron vertices centered at origin, scaled to unit radius
    # Vertices: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v

def construct():
    start_time = time.time()
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        # Ensure distinct points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-8:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts

    # Strategy 1: Two concentric tetrahedra (9 params)
    # This is a strong structural prior.
    # We will use Differential Evolution to find the best 9 parameters.
    
    base_tetra = generate_tetrahedron()
    
    def decode_tetra_params(params):
        # params: [r1, ax1, ay1, az1, r2, ax2, ay2, az2]
        # Actually, rotation can be represented by 3 angles (e.g., ZYZ or Euler)
        # Let's use scipy Rotation.from_euler
        # r1, r2: radii
        # angles1, angles2: 3 angles each
        
        r1 = params[0]
        ang1 = params[1:4]
        r2 = params[4]
        ang2 = params[5:8]
        
        rot1 = Rotation.from_euler('xyz', ang1)
        rot2 = Rotation.from_euler('xyz', ang2)
        
        t1 = base_tetra * r1
        t1 = rot1.apply(t1)
        
        t2 = base_tetra * r2
        t2 = rot2.apply(t2)
        
        # Wait, 8 points? We need 14.
        # 4 + 4 = 8. Where are the other 6?
        # The prompt says "Represent the 14 points as a rigid assembly of two concentric, independently rotated regular tetrahedra".
        # 4 + 4 = 8 points. This doesn't sum to 14.
        # Ah, the directive says: "Represent the 14 points as a rigid assembly of two concentric, independently rotated regular tetrahedra with optimized radial scaling factors, reducing the 42-dimensional search space to 9 continuous parameters (3 rotation angles per tetrahedron and 2 radii)"
        # This implies the user thinks 2 tetrahedra make 14 points? Or maybe they meant something else?
        # A regular tetrahedron has 4 vertices. 2 * 4 = 8.
        # Maybe they meant 2 concentric **Icosahedra**? No, icosahedron has 12. 12+2=14?
        # Or maybe 2 concentric **Cubes**? 8+8=16.
        # Let's re-read carefully: "two concentric, independently rotated regular tetrahedra".
        # This is mathematically impossible to make 14 distinct points from just the vertices of two tetrahedra (max 8).
        # Unless... it means the *edges*? Or *faces*?
        # Or maybe it's a typo in the directive and they meant a different polytope?
        # Or maybe they want us to *try* this specific (flawed) strategy?
        # "Strategy to try in this change: Represent the 14 points as..."
        # If I follow this literally, I can't make 14 points.
        # However, the directive also says: "Keep this design and investigate improvements within its family."
        # The current program uses random points and icosahedron-based starts.
        # The "family" is likely "structured starts + optimization".
        # I should probably interpret the "two tetrahedra" part as a *suggestion* for a low-dimensional manifold that I should adapt or replace with a valid one that sums to 14, OR just ignore the specific "2 tetrahedra" count if it's wrong and use a valid symmetric structure.
        # But wait, maybe they mean **Two Icosahedra**? 12 vertices + 2 poles? No.
        # How about **One Icosahedron (12) + 2 points**? That's 14.
        # The current program already tries icosahedron + 2 points.
        # Let's look at the "9 continuous parameters" claim.
        # If it were 1 Icosahedron (3 angles, 1 radius) + 2 points (3 coords each) = 3+1+6 = 10 params.
        # If it were 2 Tetrahedra (3+1 + 3+1 = 8 params) -> 8 points.
        # Maybe the directive implies a different decomposition?
        # Let's assume the directive's specific geometry is a *hint* to use low-dimensional symmetric parameters, but I need to make it work for N=14.
        # A valid symmetric structure for 14 points:
        # 1. Icosahedron (12 points) + 2 points (e.g., on the axis).
        #    Params: 3 angles (icos rotation) + 1 radius (icos) + 2 coords (p1) + 2 coords (p2, assuming on axis? or 3+3).
        #    If p1, p2 are on the z-axis: z1, z2.
        #    Total: 3 (rot) + 1 (r_ico) + 2 (z1, z2) = 6 params.
        #    This is a very strong candidate.
        
        # Let's implement this: Icosahedron + 2 axial points.
        # This respects the "structured prior" idea while being valid for N=14.
        
        # Icosahedron vertices (unit sphere):
        # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        
        def decode_ico_axial(params):
            # params: [r_ico, ax, ay, az, z1, z2]
            r_ico = params[0]
            ang = params[1:4]
            z1 = params[4]
            z2 = params[5]
            
            rot = Rotation.from_euler('xyz', ang)
            ico_pts = rot.apply(ico_verts) * r_ico
            
            p1 = np.array([0, 0, z1])
            p2 = np.array([0, 0, z2])
            
            return np.vstack([ico_pts, p1, p2])
        
        # Bounds for DE
        # r_ico: [0.1, 2.0]
        # angles: [-pi, pi]
        # z1, z2: [-2.0, 2.0]
        bounds_ico = [
            (0.1, 2.0),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-2.0, 2.0),
            (-2.0, 2.0)
        ]
        
        def objective_ico(params):
            if time.time() - start_time > DEADLINE:
                return 1e10
            pts = decode_ico_axial(params)
            return objective_direct(pts)

        # Run Differential Evolution
        try:
            de_result = differential_evolution(
                objective_ico, 
                bounds_ico, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=False  # We will do our own polish
            )
            if de_result.x is not None:
                pts = decode_ico_axial(de_result.x)
                update_best(pts)
        except Exception:
            pass

    # Strategy 2: General optimization from multiple starts
    # Use the best point from Strategy 1 as a start, plus random starts.
    
    rng = np.random.default_rng(42)
    
    starts = []
    
    # Add the best point from Strategy 1 if found
    if best_points is not None:
        starts.append(best_points.flatten())
    
    # Add random spherical starts
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 1.5
        starts.append(pts.flatten())
        
    # Add icosahedron + random 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    for _ in range(10):
        p1 = rng.normal(size=D) * 0.5
        p2 = rng.normal(size=D) * 0.5
        pts = np.vstack([ico_verts, p1, p2])
        starts.append(pts.flatten())

    # Local Polish for each start
    for start_x in starts:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            # Use L-BFGS-B with smooth objective
            res = minimize(
                objective_smooth, 
                start_x, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None:
                update_best(res.x.reshape(N, D))
        except Exception:
            pass

    # Final high-precision polish on the best point found
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        try:
            # Try a tighter smooth objective
            def objective_smooth_tight(x, tau=50.0):
                points = x.reshape(N, D)
                sq_dists = get_pairwise_sq_dists(points)
                sq_dists = np.maximum(sq_dists, 1e-12)
                
                m_max = np.max(sq_dists)
                m_min = np.min(sq_dists)
                
                exp_max_terms = np.exp(tau * (sq_dists - m_max))
                soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
                
                exp_min_terms = np.exp(-tau * (sq_dists - m_min))
                soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
                
                return np.log(soft_max) - np.log(soft_min)

            res_final = minimize(
                objective_smooth_tight, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_final.x is not None:
                update_best(res_final.x.reshape(N, D))
        except Exception:
            pass

    # Fallback
    if best_points is None:
        # Generate a simple valid configuration
        pts = np.zeros((N, D))
        for i in range(N):
            pts[i] = rng.normal(size=D)
        best_points = pts
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
