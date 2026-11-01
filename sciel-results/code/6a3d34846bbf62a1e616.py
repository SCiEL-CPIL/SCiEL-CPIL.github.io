"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize
import sys

N = 55

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Construct the icosahedron vertices
    # Standard icosahedron vertices for radius R
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a regular icosahedron centered at origin
    # (0, +/-1, +/-phi), (+/-1, +/-phi, 0), (+/-phi, 0, +/-1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 2. Define the inner shell (12 atoms)
    # The inner shell forms an icosahedron.
    # Let the inner shell radius be R_in.
    # In the Mackay icosahedron, the inner shell atoms are at distance R_in from center.
    # The pair distance between adjacent inner shell atoms is R_in * 2 * sin(pi/5) = R_in * 1.902...
    # Wait, the Mackay icosahedron is built by adding shells.
    # Inner shell: 12 atoms at vertices of icosahedron.
    # Outer shell: 42 atoms.
    #   12 at vertices of a larger icosahedron (same directions).
    #   30 at midpoints of the edges of the inner icosahedron, pushed out to the outer radius.
    
    # Let's parameterize by R_in (inner radius) and R_out (outer radius).
    # Actually, the structure is defined by the geometry.
    # Inner atoms: i * R_in
    # Outer atoms:
    #   12 vertices: i * R_out
    #   30 edge midpoints: normalized((v_i + v_j)/2) * R_out
    
    # We need to find the optimal R_in and R_out.
    # Often R_in is close to 2^(1/6) ~ 1.122.
    # R_out is roughly 2 * R_in? No, it's the second shell.
    # Let's optimize R_in and R_out.
    
    def get_positions(r_in, r_out):
        # Inner shell
        inner_pos = verts * r_in
        
        # Outer shell vertices
        outer_verts = verts * r_out
        
        # Outer shell edge midpoints
        # Identify edges of the icosahedron
        # An edge exists if distance between vertices is 2 * sin(pi/5) in unit sphere
        # cos(theta) = 1/phi^2 ? No.
        # Distance between adjacent vertices on unit icosahedron:
        # d = sqrt((0-1)^2 + (1-phi)^2 + (phi-0)^2) = sqrt(1 + (phi-1)^2 + phi^2)
        # phi^2 = phi+1. (phi-1)^2 = phi^2 - 2phi + 1 = 2phi - 1.
        # d^2 = 1 + 2phi - 1 + phi + 1 = 3phi + 1.
        # phi = 1.618. 3*1.618 + 1 = 5.854. sqrt(5.854) = 2.419.
        # Let's just compute pairwise distances and pick the shortest 30 pairs.
        
        v_norms = np.linalg.norm(verts, axis=1)
        # Compute distance matrix
        dists = np.zeros((12, 12))
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                dists[i, j] = d
                dists[j, i] = d
        
        # Find the 30 shortest distances (edges)
        # There are 12*11/2 = 66 pairs.
        # The 30 edges are the ones with the minimum distance.
        flat_dists = dists.flatten()
        flat_dists = flat_dists[flat_dists > 0]
        min_dist = np.min(flat_dists)
        
        edge_pairs = []
        for i in range(12):
            for j in range(i+1, 12):
                if abs(dists[i, j] - min_dist) < 1e-5:
                    edge_pairs.append((i, j))
        
        # Should be 30 pairs
        outer_mids = []
        for i, j in edge_pairs:
            mid_vec = (verts[i] + verts[j]) / 2.0
            mid_vec = mid_vec / np.linalg.norm(mid_vec)
            outer_mids.append(mid_vec * r_out)
            
        outer_mids = np.array(outer_mids)
        
        # Combine
        # Center atom
        center = np.array([[0, 0, 0]])
        
        all_pos = np.vstack([center, inner_pos, outer_verts, outer_mids])
        return all_pos

    def energy_from_positions(pos):
        # pos is (55, 3)
        n = len(pos)
        energy = 0.0
        for i in range(n):
            for j in range(i+1, n):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < 0.5: # Invalid
                    return 1e10
                r6 = r**-6
                energy += 4 * (r6*r6 - r6)
        return energy

    # Initial guesses
    # Inner shell radius ~ 1.12
    # Outer shell radius ~ 2.2? 
    # In Mackay icosahedron, the outer shell is not just a scaled icosahedron.
    # The 30 midpoints are pushed out.
    # Let's try R_in = 1.12, R_out = 2.24
    
    def objective(params):
        r_in, r_out = params
        if r_in < 0.5 or r_out < r_in + 0.5:
            return 1e10
        pos = get_positions(r_in, r_out)
        return energy_from_positions(pos)
    
    # Optimization
    best_pos = None
    best_energy = 1e10
    
    # 1. Coarse search
    # We can use scipy minimize on 2 variables
    x0 = np.array([1.12, 2.24])
    
    # Check time
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success or res.fun < best_energy:
                best_energy = res.fun
                best_pos = get_positions(res.x[0], res.x[1])
        except Exception as e:
            pass
            
    # 2. Fine search / Polish with full relaxation
    # The analytic Mackay icosahedron is symmetric. 
    # A local relaxation (L-BFGS) of this structure should converge to the global minimum 
    # if the initial guess is close enough.
    
    if best_pos is None:
        # Fallback to standard cubic lattice if optimization failed
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
        best_energy = energy_from_positions(best_pos)
        
    # 3. Final Relaxation
    # Use the provided relax function if available, otherwise do a simple gradient step or trust region.
    # The prompt says: "science_chemistry.relax(positions) (L-BFGS local relaxation) is available to candidates."
    # However, I cannot import "science_chemistry" directly as it's an evaluator interface.
    # I must implement the relaxation or assume the evaluator calls it?
    # "The evaluator independently computes the energy... science_chemistry.relax(positions) ... is available to candidates."
    # This implies I can call it. But usually in these tasks, I have to implement the logic.
    # Wait, "available to candidates" usually means I can use it if I import it? 
    # No, standard practice in these "rewrite" tasks is that I only have numpy/scipy.
    # I will implement a local relaxation using scipy.optimize.minimize on the full 165D space.
    
    if time.time() < deadline - 5:
        def full_energy_flat(x):
            pos = x.reshape(-1, 3)
            return energy_from_positions(pos)
        
        x0_flat = best_pos.flatten()
        
        try:
            # Use L-BFGS-B for efficiency
            res_relax = minimize(full_energy_flat, x0_flat, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            if res_relax.fun < best_energy:
                best_pos = res_relax.x.reshape(-1, 3)
                best_energy = res_relax.fun
        except Exception as e:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
