"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Rigid-body assembly approach: circle centers are nodes in a contact graph.
    Iteratively enforce distance and boundary constraints via projection.
    """
    
    def hex_initial(n):
        """Hexagonal lattice, centered and scaled to fit in [-0.5,0.5]^2."""
        pts = []
        # Generate hex lattice rows
        for row in range(12):
            for col in range(12):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2.0
                pts.append([x, y])
        pts = np.array(pts[:n])
        pts -= pts.mean(axis=0)
        max_abs = np.max(np.abs(pts))
        pts *= (0.48 / max_abs)
        return pts
    
    def grid_initial(n):
        """Grid layout with slight jitter."""
        m = int(np.ceil(np.sqrt(n)))
        xs, ys = np.meshgrid(np.linspace(-0.45, 0.45, m), np.linspace(-0.45, 0.45, m))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts
    
    def jittered_hex(n, seed):
        """Hex with random jitter."""
        np.random.seed(seed)
        pts = hex_initial(n)
        pts += np.random.randn(n, 2) * 0.02
        return np.clip(pts, -0.49, 0.49)
    
    def pairwise_dists(pts):
        """Compute pairwise distance matrix."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        return dist
    
    def compute_radius(pts):
        """Compute the packing radius for a configuration."""
        dist = pairwise_dists(pts)
        r_pair = np.min(dist) / 2.0
        r_bound = 0.5 - np.max(np.abs(pts), axis=1)
        r_bound = np.min(r_bound)
        return min(r_pair, r_bound)
    
    def build_contacts(pts, k=6):
        """Build contact graph: k nearest neighbors for each node."""
        dist = pairwise_dists(pts)
        n = len(pts)
        contacts = set()
        for i in range(n):
            nn = np.argsort(dist[i])[:k]
            for j in nn:
                a, b = min(i, j), max(i, j)
                contacts.add((a, b))
        return list(contacts)
    
    def solve_constraints(pts, r, contacts, n_iter=1000):
        """
        Iteratively project to satisfy:
        1. dist(i,j) >= 2r for all contacts
        2. pts in [-0.5, 0.5]^2
        Uses momentum to escape local traps.
        """
        pos = pts.copy()
        vel = np.zeros_like(pos)
        damping = 0.9
        n = len(pos)
        
        for it in range(n_iter):
            moved = False
            
            # Enforce contact distance constraints
            for (i, j) in contacts:
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                if dist < 2 * r:
                    if dist > 1e-10:
                        needed = 2 * r - dist
                        correction = d / dist * needed * 0.5
                        pos[i] -= correction
                        pos[j] += correction
                        vel[i] -= correction * 0.3
                        vel[j] += correction * 0.3
                        moved = True
                    else:
                        # Coincident: separate randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        offset = np.array([np.cos(angle), np.sin(angle)]) * r
                        pos[i] -= offset
                        pos[j] += offset
                        vel[i] -= offset * 0.3
                        vel[j] += offset * 0.3
                        moved = True
            
            # Enforce boundary: push inward if outside
            for i in range(n):
                for c in range(2):
                    if pos[i, c] < -0.5:
                        pos[i, c] = -0.5
                        vel[i, c] = 0.0
                        moved = True
                    elif pos[i, c] > 0.5:
                        pos[i, c] = 0.5
                        vel[i, c] = 0.0
                        moved = True
            
            # Apply damped velocity
            vel *= damping
            pos += vel * 0.1
            # Re-clip after velocity update
            pos = np.clip(pos, -0.5, 0.5)
            
            if not moved and np.max(np.abs(vel)) < 1e-10:
                break
        
        return pos
    
    def try_radius(pts, r, n_iter=800):
        """Try to find a feasible configuration for radius r."""
        contacts = build_contacts(pts, k=6)
        pos = solve_constraints(pts, r, contacts, n_iter=n_iter)
        # Verify feasibility
        dist = pairwise_dists(pos)
        if np.min(dist) < 2 * r - 1e-8:
            return None, False
        if np.any(np.abs(pos) > 0.5 + 1e-8):
            return None, False
        return pos, True
    
    # Try multiple initial configurations
    best_pos = None
    best_r = 0.0
    
    configs = [
        hex_initial(N),
        grid_initial(N),
        jittered_hex(N, 42),
        jittered_hex(N, 123),
        jittered_hex(N, 456),
    ]
    
    for cfg in configs:
        r0 = compute_radius(cfg)
        
        # Binary search for max radius starting from r0
        r_lo = r0 * 0.95
        r_hi = r0 * 1.15
        
        # First, try to improve at r0
        contacts = build_contacts(cfg, k=6)
        pos_imp = solve_constraints(cfg, r0, contacts, n_iter=1000)
        r_imp = compute_radius(pos_imp)
        r_lo = max(r_lo, r_imp)
        
        for _ in range(25):
            r_mid = (r_lo + r_hi) / 2
            pos_try, feasible = try_radius(cfg, r_mid, n_iter=600)
            if feasible:
                r_lo = r_mid
                if r_mid > best_r:
                    best_r = r_mid
                    best_pos = pos_try.copy()
            else:
                r_hi = r_mid
            
            if r_hi - r_lo < 1e-8:
                break
    
    # Final local refinement on best configuration
    if best_pos is not None:
        # Try to further improve with more iterations
        r_cur = compute_radius(best_pos)
        contacts = build_contacts(best_pos, k=8)
        pos_final = solve_constraints(best_pos, r_cur, contacts, n_iter=2000)
        r_final = compute_radius(pos_final)
        if r_final > best_r:
            best_r = r_final
            best_pos = pos_final
    
    # Safety: ensure valid output
    if best_pos is None or np.any(~np.isfinite(best_pos)):
        # Fallback grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    # Final check: no coincident centers
    dist = pairwise_dists(best_pos)
    if np.min(dist) < 1e-10:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
