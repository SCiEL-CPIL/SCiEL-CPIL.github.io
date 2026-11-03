import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
SQRT3 = math.sqrt(3)
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute normals for overlap checking (axis-aligned hexagons)
    OVERLAP_NORMALS = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])

    # Precompute outer normals for 6 rotation angles (beta = 0, 30, 60, 90, 120, 150 deg)
    OUTER_NORMALS = []
    for i in range(6):
        beta = i * (math.pi / 6)
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        OUTER_NORMALS.append(normals)

    def calc_S(pts, beta_idx, cx, cy):
        """Calculate outer side length S for given points, beta index, center."""
        normals = OUTER_NORMALS[beta_idx]
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = np.max(np.abs(projs))
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S

    def check_overlap(pts):
        """Check if any two hexagons overlap."""
        n_pts = len(pts)
        if n_pts < 2:
            return True
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        projs = np.abs(diffs @ OVERLAP_NORMALS.T)
        min_projs = np.min(projs, axis=2)
        iu = np.triu_indices(n_pts, k=1)
        return np.all(min_projs[iu] >= SQRT3 - 1e-9)

    def compute_energy(pts, beta_idx, cx, cy):
        """Compute total energy: overlap penalty + containment penalty."""
        # Overlap penalty
        n_pts = len(pts)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        projs = np.abs(diffs @ OVERLAP_NORMALS.T)
        min_projs = np.min(projs, axis=2)
        iu = np.triu_indices(n_pts, k=1)
        violations = np.maximum(0, SQRT3 - min_projs[iu])
        overlap_energy = np.sum(violations) * 100.0
        
        # Containment penalty
        normals = OUTER_NORMALS[beta_idx]
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = np.max(np.abs(projs))
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        containment_energy = S * S * 10.0
        
        return overlap_energy + containment_energy, S

    # Initial configurations
    def make_ring_config():
        """Central hexagon with 6 around it, plus 4 more in a ring."""
        pts = np.zeros((N, 2))
        # Center
        pts[0] = [0, 0]
        # 6 around center
        for i in range(6):
            ang = i * math.pi / 3
            pts[i+1] = [SQRT3 * math.cos(ang), SQRT3 * math.sin(ang)]
        # 4 more
        pts[7] = [2*SQRT3, 0]
        pts[8] = [0, 3]
        pts[9] = [-SQRT3, 3]
        pts[10] = [SQRT3, -3]
        return pts

    def make_row_config():
        """3-4-3-1 pattern."""
        pts = np.zeros((N, 2))
        # Row 0: 3
        for i in range(3):
            pts[i] = [i*SQRT3, 0]
        # Row 1: 4
        for i in range(4):
            pts[3+i] = [(i+0.5)*SQRT3, 1.5]
        # Row 2: 3
        for i in range(3):
            pts[7+i] = [i*SQRT3, 3]
        # Row 3: 1
        pts[10] = [SQRT3/2, 4.5]
        return pts

    def make_compact_config():
        """Compact diamond-like."""
        pts = np.zeros((N, 2))
        pts[0] = [0, 0]
        pts[1] = [SQRT3, 0]
        pts[2] = [-SQRT3, 0]
        pts[3] = [0, 1.5]
        pts[4] = [0, -1.5]
        pts[5] = [2*SQRT3, 0]
        pts[6] = [-2*SQRT3, 0]
        pts[7] = [SQRT3/2, 3]
        pts[8] = [-SQRT3/2, 3]
        pts[9] = [SQRT3/2, -3]
        pts[10] = [-SQRT3/2, -3]
        return pts

    initial_configs = [make_ring_config(), make_row_config(), make_compact_config()]

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta_idx = 0
    best_global_center = None

    for config_idx, pts_init in enumerate(initial_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # For each initial config, try different beta values
        for beta_idx in range(6):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            pts = pts_init.copy()
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            
            # Langevin dynamics with annealing
            n_steps = 2000
            dt = 0.01
            T_start = 1.0
            T_end = 0.001
            cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
            T = T_start
            
            for step in range(n_steps):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                # Compute forces
                # Overlap forces
                forces = np.zeros_like(pts)
                n_pts = len(pts)
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        diff = pts[i] - pts[j]
                        projs = np.abs(diff @ OVERLAP_NORMALS.T)
                        min_proj = np.min(projs)
                        if min_proj < SQRT3:
                            # Find which normal is closest
                            min_idx = np.argmin(projs)
                            normal = OVERLAP_NORMALS[min_idx]
                            # Force direction: push apart
                            force_mag = 100.0 * (SQRT3 - min_proj)
                            # Determine sign based on dot product
                            dot = np.dot(diff, normal)
                            if dot > 0:
                                forces[i] += force_mag * normal
                                forces[j] -= force_mag * normal
                            else:
                                forces[i] -= force_mag * normal
                                forces[j] += force_mag * normal
                
                # Containment forces
                normals = OUTER_NORMALS[beta_idx]
                pts_centered = pts - np.array([cx, cy])
                projs = pts_centered @ normals.T
                max_proj = np.max(np.abs(projs))
                D = max_proj
                S = (D + APOTHEM) / APOTHEM
                
                # Force to reduce S: push points inward
                for k in range(6):
                    proj_vals = projs[:, k]
                    for i in range(n_pts):
                        if proj_vals[i] > 0:
                            # Push inward
                            forces[i] -= 10.0 * normals[k] * proj_vals[i]
                        elif proj_vals[i] < 0:
                            forces[i] += 10.0 * normals[k] * (-proj_vals[i])
                
                # Update positions with Langevin dynamics
                noise = np.random.normal(0, np.sqrt(2 * T * dt), size=pts.shape)
                pts += dt * forces + noise
                
                # Cool down
                T *= cooling_rate
                
                # Occasionally re-center
                if step % 100 == 0:
                    cx = np.mean(pts[:, 0])
                    cy = np.mean(pts[:, 1])
                    pts = pts - np.array([cx, cy]) + np.array([cx, cy])
            
            # Final evaluation
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Check overlap
            if check_overlap(pts):
                S, _ = compute_energy(pts, beta_idx, cx, cy)
                if S < best_global_S:
                    best_global_S = S
                    best_global_pts = pts.copy()
                    best_global_beta_idx = beta_idx
                    best_global_center = (cx, cy)
            
            # Also try without re-centering
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            if check_overlap(pts):
                S, _ = compute_energy(pts, beta_idx, 0, 0)
                if S < best_global_S:
                    best_global_S = S
                    best_global_pts = pts.copy()
                    best_global_beta_idx = beta_idx
                    best_global_center = (0, 0)

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = initial_configs[0]
        best_global_center = (0.0, 0.0)
        best_global_beta_idx = 0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = best_global_beta_idx * 30.0  # 6 * 30 = 180, but we use 0-150

    # Scale slightly to ensure no overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Recalculate S
    normals = OUTER_NORMALS[best_global_beta_idx]
    pts_centered = pts_scaled - np.array([cx, cy])
    projs = pts_centered @ normals.T
    max_proj = np.max(np.abs(projs))
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
