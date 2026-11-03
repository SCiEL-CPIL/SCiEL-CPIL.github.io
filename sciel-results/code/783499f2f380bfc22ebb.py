"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import scipy.optimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq > 4.0 + 1e-9:
            return True
        
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        all_edges = []
        for i in range(6):
            all_edges.append(v1[i] - v1[(i+1) % 6])
            all_edges.append(v2[i] - v2[(i+1) % 6])
        
        for edge in all_edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            if np.max(proj1) < np.min(proj2) - 1e-12 or np.max(proj2) < np.min(proj1) - 1e-12:
                return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Compute minimum outer side length required to contain all inner hexagons."""
        max_extent = 0.0
        
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        offsets = all_verts - np.array(center)
        
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / SQRT3

    def spectral_to_positions(coeffs):
        """Convert spectral coefficients to hexagon center positions.
        coeffs[0:12] = real parts of Fourier modes 1-6 for x
        coeffs[12:24] = imaginary parts of Fourier modes 1-6 for y
        coeffs[24:36] = rotations in radians
        Returns array of shape (12, 3) with (x, y, angle_rad)
        """
        n_coeffs = 6
        real_x = coeffs[0:12]
        imag_x = coeffs[12:24]
        real_y = coeffs[24:36]
        imag_y = coeffs[36:48]
        rotations = coeffs[48:60]
        
        # Build positions using truncated Fourier series
        # position_k = sum_{m=1}^{6} (a_m * cos(m*theta_k) + b_m * sin(m*theta_k))
        theta = np.arange(N) * 2 * np.pi / N
        
        positions = np.zeros((N, 2))
        for m in range(1, n_coeffs + 1):
            a_x = real_x[m-1]
            b_x = imag_x[m-1]
            a_y = real_y[m-1]
            b_y = imag_y[m-1]
            positions[:, 0] += a_x * np.cos(m * theta) + b_x * np.sin(m * theta)
            positions[:, 1] += a_y * np.cos(m * theta) + b_y * np.sin(m * theta)
        
        result = np.zeros((N, 3))
        result[:, 0] = positions[:, 0]
        result[:, 1] = positions[:, 1]
        result[:, 2] = rotations
        return result

    def smooth_overlap_penalty(inner_positions, beta=20.0):
        """Compute smooth penalty for overlapping hexagons using log-sum-exp.
        inner_positions: array of shape (N, 3) with (x, y, angle_rad)
        """
        n_pairs = 0
        log_sum = 0.0
        
        for i in range(N):
            for j in range(i+1, N):
                dx = inner_positions[i, 0] - inner_positions[j, 0]
                dy = inner_positions[i, 1] - inner_positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Minimum distance for non-overlap of two unit hexagons is 2.0
                # Use smooth approximation: penalty = max(0, 2 - dist)^2 approximated smoothly
                excess = 2.0 - dist
                if excess > 0:
                    # Log-sum-exp style smooth max
                    log_sum += np.log(np.exp(beta * excess) + 1e-100)
                    n_pairs += 1
        
        if n_pairs == 0:
            return 0.0
        
        # Average and scale
        return (np.log(1 + np.exp(beta * 0.01)) / beta) * (log_sum / n_pairs)

    def compute_side_from_positions(inner_positions, center, angle_rad):
        """Compute outer side length from positions."""
        all_verts = np.vstack([hexagon_vertices(inner_positions[i, 0], inner_positions[i, 1], 1.0, np.degrees(inner_positions[i, 2])) for i in range(N)])
        
        offsets = all_verts - np.array(center)
        
        max_extent = 0.0
        for k in range(6):
            normal_angle = angle_rad + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / SQRT3

    def objective_from_coeffs(coeffs):
        """Full objective: minimize side length + overlap penalty."""
        inner_positions = spectral_to_positions(coeffs)
        center = (0.0, 0.0)
        angle_rad = 0.0
        
        side = compute_side_from_positions(inner_positions, center, angle_rad)
        penalty = smooth_overlap_penalty(inner_positions, beta=20.0)
        
        return side + 100.0 * penalty

    def initial_coeffs():
        """Create initial spectral coefficients for hexagonal packing."""
        coeffs = np.zeros(60)
        
        # Ring 1: 6 hexagons at radius 2, angles 0, 60, ..., 300
        # This corresponds to first harmonic with amplitude 2
        coeffs[0] = 2.0  # a_1 for x
        coeffs[12] = 0.0 # b_1 for x
        
        # For y: sin component
        coeffs[24] = 0.0 # a_1 for y  
        coeffs[36] = 2.0 # b_1 for y
        
        # Add small perturbation for ring 2
        # Second harmonic for 5 additional hexagons in gaps
        # Place 5 hexagons at angles 30, 90, 150, 210, 270 at radius 2*sqrt(3)
        # This is more complex; start with just ring 1 and let optimizer place the rest
        
        # Actually, let's use a different approach: initialize with known good positions
        # and fit spectral coefficients to them
        
        # Known good initial configuration:
        # Center: (0,0)
        # Ring 1: 6 at radius 2, angles 0,60,120,180,240,300
        # Ring 2: 5 at radius 2*sqrt(3), angles 30,90,150,210,270
        
        target_positions = []
        target_positions.append((0.0, 0.0, 0.0))
        for i in range(6):
            theta = i * np.pi / 3
            target_positions.append((2.0*np.cos(theta), 2.0*np.sin(theta), 0.0))
        for i in range(5):
            theta = np.radians(30 + i*60)
            target_positions.append((2.0*SQRT3*np.cos(theta), 2.0*SQRT3*np.sin(theta), 0.0))
        
        # Fit Fourier coefficients to these positions
        theta = np.arange(N) * 2 * np.pi / N
        
        for m in range(1, 7):
            # a_m = (2/N) * sum(x_k * cos(m*theta_k))
            # b_m = (2/N) * sum(x_k * sin(m*theta_k))
            a_x = (2.0/N) * np.sum([target_positions[k][0] * np.cos(m*theta[k]) for k in range(N)])
            b_x = (2.0/N) * np.sum([target_positions[k][0] * np.sin(m*theta[k]) for k in range(N)])
            a_y = (2.0/N) * np.sum([target_positions[k][1] * np.cos(m*theta[k]) for k in range(N)])
            b_y = (2.0/N) * np.sum([target_positions[k][1] * np.sin(m*theta[k]) for k in range(N)])
            
            coeffs[m-1] = a_x
            coeffs[12+m-1] = b_x
            coeffs[24+m-1] = a_y
            coeffs[36+m-1] = b_y
        
        # Set rotations to 0
        coeffs[48:60] = 0.0
        
        return coeffs

    best_coeffs = initial_coeffs()
    best_score = objective_from_coeffs(best_coeffs)
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    # Simulated annealing on spectral coefficients
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    step_size = 0.1
    max_steps = 50000
    
    step_count = 0
    while time.time() < deadline - 30.0 and step_count < max_steps:
        step_count += 1
        new_coeffs = current_coeffs.copy()
        
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, 59)
            current_step = step_size * (0.5 + 0.5 * T)
            new_coeffs[idx] += random.gauss(0, current_step)
        
        new_score = objective_from_coeffs(new_coeffs)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / max(T, T_min)):
            current_coeffs = new_coeffs
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Polish with L-BFGS-B using smooth objective
    if time.time() < deadline - 10.0:
        try:
            def smooth_objective(coeffs):
                inner_positions = spectral_to_positions(coeffs)
                center = (0.0, 0.0)
                angle_rad = 0.0
                
                side = compute_side_from_positions(inner_positions, center, angle_rad)
                penalty = smooth_overlap_penalty(inner_positions, beta=30.0)
                
                return side + 50.0 * penalty
            
            result = scipy.optimize.minimize(
                smooth_objective, 
                best_coeffs, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            p = result.x
            inner_positions = spectral_to_positions(p)
            
            # Validate
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (inner_positions[i, 0], inner_positions[i, 1], np.degrees(inner_positions[i, 2]))
                    h2 = (inner_positions[j, 0], inner_positions[j, 1], np.degrees(inner_positions[j, 2]))
                    if not hexagons_disjoint_fast(h1, h2):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                new_side = compute_side_from_positions(inner_positions, (0.0, 0.0), 0.0)
                if new_side < best_score:
                    best_score = new_side
                    best_coeffs = p
        except Exception:
            pass

    # Extract best
    inner_positions = spectral_to_positions(best_coeffs)
    inner = []
    for i in range(N):
        inner.append([float(inner_positions[i, 0]), float(inner_positions[i, 1]), float(np.degrees(inner_positions[i, 2]))])
    
    center = (0.0, 0.0)
    angle = 0.0
    side = compute_side_from_positions(inner_positions, center, np.radians(angle))
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback to simple hexagonal packing
        inner = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = i * np.pi / 3
            inner.append([2.0*np.cos(theta), 2.0*np.sin(theta), 0.0])
        for i in range(5):
            theta = np.radians(30 + i*60)
            inner.append([2.0*SQRT3*np.cos(theta), 2.0*SQRT3*np.sin(theta), 0.0])
        center = (0.0, 0.0)
        side = 6.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
