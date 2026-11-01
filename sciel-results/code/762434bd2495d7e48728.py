import numpy as np
from scipy.optimize import minimize

N = 55

def construct():
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    # Using the standard coordinates (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find adjacent pairs (edges)
    # In an icosahedron, each vertex is connected to 5 others.
    # The dot product of adjacent unit vectors is constant.
    # Let's compute dot products
    dots = verts_unit @ verts_unit.T
    # The max dot product (excluding self) corresponds to adjacent vertices
    # For unit icosahedral vertices, adjacent dot product is 1/phi = phi-1 ≈ 0.618
    # Non-adjacent (opposite) is -1.
    # Let's find the 5 nearest neighbors for each vertex
    
    edges = []
    for i in range(12):
        # Get indices of 5 largest dot products (excluding i)
        d = dots[i].copy()
        d[i] = -np.inf
        # Find indices of top 5
        # Since it's a regular icosahedron, the 5 neighbors have the same dot product
        # We just need to pick them
        neighbor_indices = np.argsort(d)[-5:]
        for j in neighbor_indices:
            if i < j:
                edges.append((i, j))
                
    # Construct outer shell vectors (42 atoms)
    # 12 atoms: 2 * v_i
    # 30 atoms: v_i + v_j (for adjacent i,j)
    outer_vecs = []
    for i in range(12):
        v = 2 * verts_unit[i]
        outer_vecs.append(v / np.linalg.norm(v))
        
    for i, j in edges:
        v = verts_unit[i] + verts_unit[j]
        outer_vecs.append(v / np.linalg.norm(v))
        
    outer_vecs = np.array(outer_vecs)
    
    # Construct inner shell vectors (12 atoms)
    inner_vecs = verts_unit
    
    # Initial radii
    # LJ minimum is at r = 2^(1/6) ≈ 1.12246
    # For icosahedron, inner shell radius is typically close to 1.12
    # Outer shell radius is typically around 2.2
    R1 = 1.12
    R2 = 2.2
    
    # Build initial positions
    # Center
    pos = np.zeros((1, 3))
    # Inner
    pos_in = R1 * inner_vecs
    # Outer
    pos_out = R2 * outer_vecs
    
    positions = np.vstack([pos, pos_in, pos_out])
    
    # Define energy function
    def energy(x):
        # x is flat array of length 3*N
        coords = x.reshape(N, 3)
        # Calculate pair distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=2)
        # Mask diagonal
        np.fill_diagonal(r2, np.inf)
        # Avoid division by zero (though inf handles it)
        r = np.sqrt(r2)
        # LJ potential
        # 4 * (r^-12 - r^-6)
        # Sum over all pairs (i < j)
        # The sum over all i!=j is 2 * sum_{i<j}
        # So we can sum all and divide by 2
        inv_r6 = r**-6
        inv_r12 = inv_r6**2
        total_energy = 4 * np.sum(inv_r12 - inv_r6) / 2
        return total_energy
    
    # Optimize
    x0 = positions.flatten()
    
    # Use L-BFGS-B
    # No bounds, but we need to ensure atoms don't overlap
    # L-BFGS-B might push atoms together if not careful, but LJ repulsion is strong
    result = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    final_positions = result.x.reshape(N, 3)
    
    return final_positions
