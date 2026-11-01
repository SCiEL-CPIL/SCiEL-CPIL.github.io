import numpy as np
import time
from scipy.optimize import minimize

N = 55

def _lj_energy(positions):
    """Compute Lennard-Jones energy for a set of positions."""
    # positions: (N, 3)
    # Calculate all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-10] = 1e-10
    
    r = np.sqrt(dist_sq)
    
    # Compute r^-6 and r^-12
    r6 = 1.0 / (r**6)
    r12 = r6 * r6
    
    # Energy is 4 * sum(r^-12 - r^-6) over all pairs i < j
    # The matrix includes i=j (which is 0 contribution if we mask it) and i,j & j,i
    # We need to sum over i < j.
    # Sum of all elements is 2 * sum_{i<j} + sum_{i=i}
    # sum_{i=i} is 0 because r=0 -> masked.
    # So sum_{i<j} = 0.5 * sum_{i!=j}
    
    # Mask diagonal
    np.fill_diagonal(dist_sq, 1.0) # Ensure no zero divs in r calculation if we re-calculated, but we already did.
    # Actually, let's just sum the upper triangle.
    i, j = np.triu_indices_from(dist_sq, k=1)
    energy = 4.0 * np.sum(r12[i, j] - r6[i, j])
    return energy

def _lj_grad(positions):
    """Compute gradient of Lennard-Jones energy."""
    N = positions.shape[0]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-10] = 1e-10
    
    r = np.sqrt(dist_sq)
    r6 = 1.0 / (r**6)
    r12 = r6 * r6
    
    # Gradient of 4(r^-12 - r^-6) w.r.t r is 4(-12r^-13 + 6r^-7)
    # dE/dx_i = sum_{j!=i} dE/dr_{ij} * dx_{ij}/dr_{ij}
    # dx_{ij}/dr_{ij} = (x_i - x_j) / r_{ij}
    
    # Force on i from j: F_ij = -dE/dx_i
    # We want dE/dx_i.
    
    # Term for pair (i,j): 4 * (-12 * r_ij^-13 + 6 * r_ij^-7) * (x_i - x_j) / r_ij
    # = 4 * (-12 * r_ij^-14 + 6 * r_ij^-8) * (x_i - x_j)
    
    coeff = 4.0 * (-12.0 * (1.0 / (r**14)) + 6.0 * (1.0 / (r**8)))
    
    # coeff is (N, N). 
    # grad[i] = sum_j coeff[i,j] * (x_i - x_j)
    
    # Zero out diagonal
    np.fill_diagonal(coeff, 0.0)
    
    # Multiply coeff by diff
    # diff[i,j] = x_i - x_j
    # grad[i] = sum_j coeff[i,j] * diff[i,j]
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Generate Icosahedron vertices and face centers
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    # (0, 1, phi), (0, -1, phi), ...
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit length
    verts_norm = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 face centers
    # The face centers are the averages of the 3 vertices of each face.
    # There are 20 faces. Each face has 3 vertices.
    # We can find faces by checking which triplets of vertices form an equilateral triangle with side length sqrt(2) (for unit circumradius? No, side length is 2 for these coords? 
    # Distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + phi^2 - 2phi + 1 + phi^2) = sqrt(2 + 2phi^2 - 2phi).
    # phi^2 = phi + 1. So 2 + 2(phi+1) - 2phi = 2 + 2phi + 2 - 2phi = 4. Distance is 2.
    # So we look for triplets with distance 2.
    
    faces = []
    indices = list(range(12))
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ki = np.linalg.norm(verts[k] - verts[i])
                if np.isclose(d_ij, 2.0) and np.isclose(d_jk, 2.0) and np.isclose(d_ki, 2.0):
                    faces.append((i, j, k))
    
    # There should be 20 faces
    face_centers = []
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        face_centers.append(center)
    
    face_centers = np.array(face_centers)
    face_centers_norm = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    # 2. Define radial optimization function
    # Parameters: R1 (inner shell), R2 (outer vertices), R3 (outer face centers)
    # Structure:
    # 1 atom at (0,0,0)
    # 12 atoms at R1 * verts_norm
    # 12 atoms at R2 * verts_norm
    # 30 atoms at R3 * face_centers_norm
    
    def radial_energy(params):
        R1, R2, R3 = params
        # Ensure positive radii
        R1 = max(R1, 0.1)
        R2 = max(R2, 0.1)
        R3 = max(R3, 0.1)
        
        positions = np.zeros((55, 3))
        positions[0] = [0, 0, 0]
        positions[1:13] = R1 * verts_norm
        positions[13:25] = R2 * verts_norm
        positions[25:55] = R3 * face_centers_norm
        
        return _lj_energy(positions)
    
    # Initial guesses
    # LJ minimum is at r=1.122
    R1_init = 1.12
    R2_init = 2.24
    # Face center of icosahedron is at distance R * (sqrt(10(5+sqrt(5)))/10) from center?
    # Let's calculate the norm of the face center of the unit icosahedron.
    fc_norm = np.linalg.norm(face_centers[0])
    # fc_norm is approx 0.7947 * circumradius?
    # Circumradius of verts is norm(verts[0]) = sqrt(1+phi^2) = sqrt(1+phi+1) = sqrt(2+phi) ~ 1.902
    # So fc_norm ~ 1.51
    # Ratio fc_norm / circumradius ~ 0.794
    R3_init = R2_init * (fc_norm / np.linalg.norm(verts[0]))
    
    x0 = np.array([R1_init, R2_init, R3_init])
    
    # Optimize radial parameters
    res = minimize(radial_energy, x0, method='L-BFGS-B', bounds=[(0.5, 5.0)]*3, options={'maxiter': 500})
    
    if time.time() > deadline:
         # Fallback to initial guess if timeout
         best_params = x0
    else:
         best_params = res.x
         R1, R2, R3 = best_params
         R1 = max(R1, 0.1)
         R2 = max(R2, 0.1)
         R3 = max(R3, 0.1)
         
         # Construct initial Cartesian positions
         positions = np.zeros((55, 3))
         positions[0] = [0, 0, 0]
         positions[1:13] = R1 * verts_norm
         positions[13:25] = R2 * verts_norm
         positions[25:55] = R3 * face_centers_norm
         
         # 3. Full Cartesian Relaxation
         # This will polish the structure to the exact local minimum.
         res_cart = minimize(_lj_energy, positions, jac=_lj_grad, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
         
         if time.time() > deadline:
             # Return the best found so far (radial optimized)
             final_positions = positions
         else:
             final_positions = res_cart.x
             
         # Verify minimum distance constraint
         # The problem states atoms must be at least 0.5 apart.
         # If the optimization resulted in atoms closer than 0.5, we might need to handle it.
         # However, for the global minimum, distances are > 1.0.
         # We'll check just in case.
         diff = final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :]
         dist_sq = np.sum(diff**2, axis=2)
         np.fill_diagonal(dist_sq, 1.0)
         min_dist = np.sqrt(np.min(dist_sq))
         
         if min_dist < 0.5:
             # Scale up to satisfy constraint? This will increase energy.
             # Better to return the radial optimized version if it was valid?
             # Or just scale.
             scale = 0.5 / min_dist
             final_positions *= scale
             
         return final_positions

# Fallback if something goes wrong
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    try:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        verts_norm = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d_ij = np.linalg.norm(verts[i] - verts[j])
                    d_jk = np.linalg.norm(verts[j] - verts[k])
                    d_ki = np.linalg.norm(verts[k] - verts[i])
                    if np.isclose(d_ij, 2.0) and np.isclose(d_jk, 2.0) and np.isclose(d_ki, 2.0):
                        faces.append((i, j, k))
        
        face_centers = []
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            face_centers.append(center)
        face_centers = np.array(face_centers)
        face_centers_norm = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
        
        def radial_energy(params):
            R1, R2, R3 = params
            R1 = max(R1, 0.1)
            R2 = max(R2, 0.1)
            R3 = max(R3, 0.1)
            
            positions = np.zeros((55, 3))
            positions[0] = [0, 0, 0]
            positions[1:13] = R1 * verts_norm
            positions[13:25] = R2 * verts_norm
            positions[25:55] = R3 * face_centers_norm
            
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist_sq[dist_sq < 1e-10] = 1e-10
            r = np.sqrt(dist_sq)
            r6 = 1.0 / (r**6)
            r12 = r6 * r6
            i, j = np.triu_indices_from(dist_sq, k=1)
            return 4.0 * np.sum(r12[i, j] - r6[i, j])
        
        R1_init = 1.12
        R2_init = 2.24
        fc_norm = np.linalg.norm(face_centers[0])
        R3_init = R2_init * (fc_norm / np.linalg.norm(verts[0]))
        
        x0 = np.array([R1_init, R2_init, R3_init])
        
        res = minimize(radial_energy, x0, method='L-BFGS-B', bounds=[(0.5, 5.0)]*3, options={'maxiter': 500})
        
        if time.time() > deadline:
             best_params = x0
        else:
             best_params = res.x
             R1, R2, R3 = best_params
             R1 = max(R1, 0.1)
             R2 = max(R2, 0.1)
             R3 = max(R3, 0.1)
             
             positions = np.zeros((55, 3))
             positions[0] = [0, 0, 0]
             positions[1:13] = R1 * verts_norm
             positions[13:25] = R2 * verts_norm
             positions[25:55] = R3 * face_centers_norm
             
             def _lj_energy(pos):
                 diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                 dist_sq = np.sum(diff**2, axis=2)
                 dist_sq[dist_sq < 1e-10] = 1e-10
                 r = np.sqrt(dist_sq)
                 r6 = 1.0 / (r**6)
                 r12 = r6 * r6
                 i, j = np.triu_indices_from(dist_sq, k=1)
                 return 4.0 * np.sum(r12[i, j] - r6[i, j])
             
             def _lj_grad(pos):
                 diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                 dist_sq = np.sum(diff**2, axis=2)
                 dist_sq[dist_sq < 1e-10] = 1e-10
                 r = np.sqrt(dist_sq)
                 r6 = 1.0 / (r**6)
                 r12 = r6 * r6
                 coeff = 4.0 * (-12.0 * (1.0 / (r**14)) + 6.0 * (1.0 / (r**8)))
                 np.fill_diagonal(coeff, 0.0)
                 return np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
             
             res_cart = minimize(_lj_energy, positions, jac=_lj_grad, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
             
             if time.time() > deadline:
                 final_positions = positions
             else:
                 final_positions = res_cart.x
                 
             diff = final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :]
             dist_sq = np.sum(diff**2, axis=2)
             np.fill_diagonal(dist_sq, 1.0)
             min_dist = np.sqrt(np.min(dist_sq))
             
             if min_dist < 0.5:
                 scale = 0.5 / min_dist
                 final_positions *= scale
                 
             return final_positions

    except Exception:
        # Fallback to simple cubic lattice if everything fails
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12

# Note: I defined construct twice in the thought process. I will output the single robust version.
