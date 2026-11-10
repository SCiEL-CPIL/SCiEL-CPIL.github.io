import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    time_limit = 590.0  # Safety margin before 630s kill

    def check_time():
        return time.time() - start_time < time_limit

    def score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def log_ratio(points):
        """Calculate log(d_min/d_max) for smoother optimization"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-10 or d_max <= 1e-10:
            return -1e10
        return np.log(d_min) - np.log(d_max)

    def objective(params):
        """Minimize -log_ratio to maximize log_ratio"""
        points = params.reshape(n, 2)
        lr = log_ratio(points)
        if lr < -1e9:
            return 1e10
        return -lr

    def generate_hex_cluster():
        """Generate a hexagonal cluster of 16 points"""
        s3 = np.sqrt(3)
        # Hexagonal arrangement for 16 points
        # Center + 6 around + 9 more
        pts = [(0, 0)]
        # Ring 1: 6 points
        for k in range(6):
            angle = k * np.pi / 3
            pts.append((np.cos(angle), np.sin(angle) * s3))
        # Ring 2: 12 points, but we only need 9 more
        # Let's use a different approach: hexagonal lattice
        # Actually, let's just use a proper hexagonal packing
        # For 16 points, a common arrangement is:
        # Row 0: 4 points
        # Row 1: 4 points
        # Row 2: 4 points
        # Row 3: 4 points
        # With hexagonal offset
        
        # Better: use hexagonal lattice points
        pts = []
        # Center
        pts.append([0, 0])
        # First ring: 6 points
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Second ring: 12 points, take 9
        for k in range(12):
            if len(pts) >= 16:
                break
            angle = k * np.pi / 6 + np.pi / 6  # offset
            # Actually, second ring in hex is at distance 2
            # Angles: 0, pi/6, pi/3, pi/2, ... 
            # Let's use proper hex coordinates
            pass
        
        # Simpler: just use a 4x4 grid with hexagonal offset
        # Or use the known optimal-ish configurations
        # Let's use a hexagonal lattice pattern
        pts = []
        # Hexagonal lattice: (i + j/2, j*sqrt(3)/2)
        # We need 16 points
        # Let's try:
        # y=0: x = -1.5, -0.5, 0.5, 1.5 (4 points)
        # y=s3/2: x = -1, 0, 1, 2 (4 points)
        # y=s3: x = -1.5, -0.5, 0.5, 1.5 (4 points)
        # y=3*s3/2: x = -1, 0, 1, 2 (4 points)
        
        s3 = np.sqrt(3)
        for j in range(4):
            y = j * s3 / 2
            for i in range(4):
                x = i - 1.5 + (j % 2) * 0.5
                pts.append([x, y])
        
        arr = np.array(pts)
        center = arr.mean(axis=0)
        arr -= center
        return arr

    def generate_square_grid():
        """Generate a 4x4 square grid"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def generate_rings(n_inner, n_outer):
        """Generate ring configuration"""
        p_center = np.array([0.0, 0.0])
        r1 = 0.5
        angles1 = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        p_inner = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        r2 = 1.0
        angles2 = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        angles2 += np.pi / n_outer
        p_outer = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([p_center, p_inner, p_outer])

    def generate_pentagon_plus():
        """Generate a configuration based on 5-fold symmetry"""
        # 1 center, 5 in inner ring, 10 in outer ring
        pts = [[0, 0]]
        r1 = 0.6
        for k in range(5):
            angle = k * 2 * np.pi / 5
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        r2 = 1.0
        for k in range(10):
            angle = k * 2 * np.pi / 10 + np.pi / 10
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        arr = np.array(pts)
        center = arr.mean(axis=0)
        arr -= center
        return arr

    def generate_fibonacci_sphere_projected():
        """Generate points using Fibonacci sphere and project to 2D"""
        golden = (1 + np.sqrt(5)) / 2
        pts = []
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        arr = np.array(pts)
        center = arr.mean(axis=0)
        arr -= center
        return arr

    best_points = None
    best_score = -1.0

    # List of initial configurations
    inits = [
        generate_hex_cluster(),
        generate_square_grid(),
        generate_rings(6, 9),
        generate_rings(5, 10),
        generate_rings(7, 8),
        generate_pentagon_plus(),
        generate_fibonacci_sphere_projected(),
    ]
    
    # Add some random inits
    for _ in range(8):
        rand_pts = np.random.rand(n, 2) * 2 - 1
        inits.append(rand_pts)

    # Optimization using Powell's method (good for this type of problem)
    for i, init_pts in enumerate(inits):
        if not check_time():
            break
            
        current_pts = init_pts.copy()
        current_score = score(current_pts)
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()

        # Use Powell's method for optimization
        try:
            result = minimize(
                objective,
                current_pts.flatten(),
                method='Powell',
                options={'maxiter': 500, 'maxfev': 5000}
            )
            
            if result.success or result.fun < 1e9:
                opt_pts = result.x.reshape(n, 2)
                opt_score = score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
        except Exception:
            pass

        if not check_time():
            break

        # Also try Nelder-Mead for some inits
        if i % 2 == 0:
            try:
                result2 = minimize(
                    objective,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'maxfev': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if result2.success or result2.fun < 1e9:
                    opt_pts = result2.x.reshape(n, 2)
                    opt_score = score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
            except Exception:
                pass

    # If we have time, do a local refinement with gradient-based method
    if best_points is not None and check_time():
        try:
            # Use L-BFGS-B for refinement
            result3 = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfev': 10000}
            )
            
            if result3.success or result3.fun < 1e9:
                opt_pts = result3.x.reshape(n, 2)
                opt_score = score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
        except Exception:
            pass

    # If no valid points were found, return a default
    if best_points is None:
        best_points = generate_hex_cluster()

    return best_points
