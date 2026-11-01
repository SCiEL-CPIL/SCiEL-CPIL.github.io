import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Mackay icosahedron with 55 atoms and relax it.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # 1. Generate Mackay Icosahedron coordinates
    # A Mackay icosahedron consists of:
    # - 1 atom at the center
    # - 12 atoms at the vertices of the inner icosahedron (radius r1)
    # - 12 atoms at the vertices of the outer icosahedron (radius r2)
    # - 30 atoms at the midpoints of the edges of the outer icosahedron (radius r3)
    # Total: 1 + 12 + 12 + 30 = 55 atoms.
    
    # Standard icosahedron vertices (unit radius)
    # Golden ratio phi
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a regular icosahedron with circumradius 1
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    v = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            v.append([0, i, j * phi])
            v.append([i, j * phi, 0])
            v.append([j * phi, 0, i])
    v = np.array(v)
    # Normalize to unit length
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    
    # Mackay radii for the shells
    # The standard Mackay icosahedron has shells at radii R, R*sqrt(5)/2, etc.
    # However, the optimal LJ cluster is not exactly the geometric Mackay icosahedron,
    # but very close. We start with the geometric Mackay radii.
    # Inner shell (12 atoms): radius ~ 1.12 * sqrt(5)/2 ? No, let's use standard Mackay scaling.
    # The Mackay icosahedron is built by shrinking the outer icosahedron.
    # Let R_outer be the radius of the 12 outer vertices.
    # The 30 edge atoms are at distance R_outer * sqrt(1 + 1/phi^2) / 2 ? 
    # Actually, the edge midpoints of a unit icosahedron are at distance 1/2 * sqrt(1 + 1/phi^2) * 2? 
    # Let's calculate the distance from center to edge midpoint for unit vertices.
    # Midpoint of (0,1,phi) and (1,phi,0) is (0.5, (1+phi)/2, phi/2).
    # Distance squared: 0.25 + ((1+phi)/2)^2 + (phi/2)^2
    # (1+phi)^2/4 + phi^2/4 + 1/4 = (1 + 2phi + phi^2 + phi^2 + 1)/4 = (2 + 2phi + 2phi^2)/4 = (1 + phi + phi^2)/2
    # phi^2 = phi + 1, so 1 + phi + phi + 1 = 2 + 2phi.
    # Dist^2 = (2 + 2phi)/2 = 1 + phi.
    # So edge midpoint radius is sqrt(1 + phi) for unit circumradius.
    # sqrt(1 + 1.618) = sqrt(2.618) ≈ 1.618 = phi.
    # So if outer vertices are at R, edge midpoints are at R * phi? No, that's for unit circumradius.
    # If circumradius is 1, edge midpoint radius is sqrt(1+phi) = phi.
    # Wait, phi = 1.618. sqrt(2.618) = 1.618. Yes.
    # So the 30 edge atoms are at the same radius as the 12 outer vertices?
    # No, the 12 outer vertices are at radius 1. The 30 edge atoms are at radius phi?
    # Let's check the Mackay icosahedron definition.
    # The Mackay icosahedron is a subset of the icosahedral quasicrystal.
    # The 55-atom cluster is the first complete shell.
    # It has 12 vertices (outer), 30 edge centers (outer), and 12 inner vertices + 1 center.
    # The "outer" shell consists of 42 atoms (12 vertices + 30 edges).
    # In the Mackay construction, the outer shell is a regular icosahedron with 12 vertices.
    # The 30 edge atoms are NOT at the same radius as the vertices in the optimized LJ cluster,
    # but in the geometric Mackay icosahedron, they are at the midpoints of the edges.
    # If the outer vertices are at radius R, the edge midpoints are at R * (phi / 1) ? No.
    # For a regular icosahedron with circumradius R, the distance from center to edge midpoint is R * sqrt(1 + phi) / 1 ?
    # Let's re-verify.
    # Vertices at distance R.
    # Midpoint distance = R * sqrt(1+phi).
    # Since sqrt(1+phi) = phi, the edge midpoints are at radius R * phi.
    # This means the 30 edge atoms are FURTHER OUT than the 12 vertices?
    # No, that's impossible for a convex hull. The vertices are the extreme points.
    # My calculation:
    # Vertices: (0, 1, phi). Norm = sqrt(1 + phi^2) = sqrt(1 + phi + 1) = sqrt(phi + 2).
    # So if we want circumradius R, we scale by R / sqrt(phi + 2).
    # Midpoint: (0.5, (1+phi)/2, phi/2).
    # Norm^2 = 0.25 + (1+phi)^2/4 + phi^2/4 = (1 + (1+2phi+phi^2) + phi^2)/4 = (2 + 2phi + 2phi^2)/4 = (1 + phi + phi^2)/2.
    # phi^2 = phi + 1.
    # Norm^2 = (1 + phi + phi + 1)/2 = (2 + 2phi)/2 = 1 + phi.
    # Ratio of radii: R_mid / R_vert = sqrt(1+phi) / sqrt(phi+2).
    # Since phi+2 = phi^2+1? No, phi^2 = phi+1. So phi+2 = phi^2+1.
    # So R_mid / R_vert = sqrt(1+phi) / sqrt(phi^2+1).
    # Note that phi^2+1 = phi+2.
    # Is 1+phi smaller than phi+2? Yes.
    # So R_mid < R_vert.
    # Let's compute the ratio:
    # sqrt(2.618) / sqrt(3.618) = 1.618 / 1.902 = 0.85.
    # So the 30 edge atoms are closer to the center than the 12 vertices.
    
    # Let's set the outer vertex radius R_v = 1.0 (arbitrary scale, LJ will optimize).
    # Actually, the optimal LJ distance is 2^(1/6) ≈ 1.122.
    # We should scale the whole structure so that typical distances are around 1.12.
    
    R_v = 1.0
    scale_vert = R_v / np.sqrt(phi + 2)
    vertices = v * scale_vert
    
    # Calculate edge midpoints
    # Edges connect vertices that are at distance 2/sqrt(phi+2) * sqrt(1+phi)? No.
    # Let's just find pairs of vertices that are close to each other.
    # In an icosahedron, each vertex is connected to 5 others.
    # Distance between connected vertices (edge length):
    # d = 2 * R_v * sin(pi/5) ? No, that's for a circle.
    # For icosahedron, edge length a = 2 * R * sin(pi/5) is for the circumscribed sphere?
    # Actually, edge length a = R * 2 / sqrt(phi+2) * sqrt(1+phi)?
    # Let's just compute distances between all pairs of the 12 vertices and pick the 30 shortest.
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # The 30 edges correspond to the 30 smallest distances (which are all equal in a regular icosahedron).
    # We need to identify the pairs.
    edge_pairs = []
    for i in range(12):
        for j in range(i+1, 12):
            if dists[i, j] < 0.1: # This is wrong, we need the actual value.
                pass
    
    # Let's find the unique edge length.
    unique_dists = np.unique(dists[dists > 0])
    # There should be 2 unique distances: edge length and face diagonal?
    # In an icosahedron, all edges are equal. The next closest are face diagonals? No, triangles.
    # So there are 30 edges.
    min_dist = np.min(unique_dists)
    
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edge_pairs.append((i, j))
    
    # There should be 30 pairs.
    # Calculate midpoints
    edge_atoms = []
    for i, j in edge_pairs:
        mid = (vertices[i] + vertices[j]) / 2.0
        edge_atoms.append(mid)
    edge_atoms = np.array(edge_atoms)
    
    # Inner shell: 12 atoms.
    # In the Mackay icosahedron, the inner shell is a smaller icosahedron.
    # The ratio of radii is 1/phi? Or something similar.
    # The Mackay icosahedron is generated by the icosahedral quasicrystal.
    # The first shell is at radius R1, second at R2.
    # For the 55-atom cluster, the inner 12 atoms are at the vertices of a smaller icosahedron.
    # The radius ratio is typically 1/phi for the Mackay construction?
    # Let's check the literature or derive it.
    # The Mackay icosahedron is a subset of the icosahedral lattice.
    # The 55-atom cluster is the "first" Mackay icosahedron.
    # It consists of the origin, the 12 vertices of a small icosahedron, and the 42 vertices of a large icosahedron (12 vertices + 30 edge midpoints).
    # The ratio of the radii of the two icosahedra is 1/phi.
    # Let's assume R_inner = R_outer / phi.
    # But wait, the "outer" shell here is the 42-atom shell.
    # The 12 vertices are at R_v.
    # The 30 edges are at R_e = R_v * sqrt(1+phi) / sqrt(phi+2) = R_v * (phi/sqrt(phi+2)) / sqrt(phi+2)?
    # No, R_e = R_v * sqrt(1+phi)/sqrt(phi+2).
    # Since 1+phi = phi^2, R_e = R_v * phi / sqrt(phi+2).
    # And R_v = R_circum * 1/sqrt(phi+2) ? No, we set R_v = 1.
    # So R_e = phi / (phi+2) * R_v? No.
    # R_e / R_v = sqrt(1+phi) / sqrt(phi+2) = sqrt(phi^2) / sqrt(phi+2) = phi / sqrt(phi+2).
    # This is approx 1.618 / 1.902 = 0.85.
    
    # The inner 12 atoms are at a smaller radius.
    # In the Mackay icosahedron, the inner shell is rotated?
    # The Mackay icosahedron has the same orientation.
    # So inner vertices = v * (R_inner / sqrt(phi+2)).
    # What is R_inner?
    # If the ratio is 1/phi, then R_inner_circum = R_outer_circum / phi.
    # R_outer_circum = R_v * sqrt(phi+2).
    # So R_inner_circum = R_v * sqrt(phi+2) / phi.
    # Inner vertex position = v * (R_inner_circum / sqrt(phi+2)) = v * (R_v / phi).
    # So inner_vertices = vertices / phi.
    
    inner_vertices = vertices / phi
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Combine all atoms
    # 12 outer vertices
    # 30 outer edge atoms
    # 12 inner vertices
    # 1 center
    atoms = np.vstack([vertices, edge_atoms, inner_vertices, center])
    
    # Check count
    assert len(atoms) == 55, f"Expected 55 atoms, got {len(atoms)}"
    
    # Scale the entire cluster.
    # The optimal LJ distance is 2^(1/6) ≈ 1.122.
    # In our construction, the distance between outer vertices is:
    # d_edge = 2 * R_v * sin(pi/5)? No.
    # d_edge = min_dist from earlier.
    # Let's calculate the typical distance.
    # The distance between adjacent outer vertices (edge length) is:
    # a = 2 * R_v * sin(pi/5) is for a regular polygon.
    # For icosahedron, edge length a = R_circum * 2 / sqrt(phi+2) * sqrt(1+phi)?
    # We know R_v = R_circum / sqrt(phi+2) is wrong.
    # We defined v as unit vectors.
    # vertices = v / sqrt(phi+2) * R_v ? No, vertices = v * (R_v / sqrt(phi+2)).
    # So R_circum = R_v.
    # Edge length a = 2 * R_circum * sin(pi/5) ? No.
    # a = 2 * R_circum * sqrt(1/10 * (5 - sqrt(5))) ?
    # Let's just compute the average nearest neighbor distance from the array.
    # And scale so that the nearest neighbor distance is 1.122.
    
    # Compute all pair distances
    dists_all = np.zeros((55, 55))
    for i in range(55):
        for j in range(i+1, 55):
            d = np.linalg.norm(atoms[i] - atoms[j])
            dists_all[i, j] = d
            dists_all[j, i] = d
    
    # Find the minimum distance (nearest neighbors)
    min_nn_dist = np.min(dists_all[dists_all > 0])
    
    # Scale factor to make min_nn_dist = 1.122
    scale_factor = 1.122 / min_nn_dist
    atoms = atoms * scale_factor
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    atoms += np.random.normal(0, 0.01, atoms.shape)
    
    # Relax using science_chemistry if available
    try:
        import science_chemistry
        if hasattr(science_chemistry, 'relax'):
            # Check time
            elapsed = time.time() - start_time
            if elapsed < deadline - 10:
                atoms = science_chemistry.relax(atoms)
    except Exception:
        pass
    
    # Final check: ensure atoms are at least 0.5 apart
    # If relaxation caused overlaps, we might need to handle it, but LJ relaxation usually avoids this.
    # We'll just return the relaxed positions.
    
    return atoms

# EVOLVE-BLOCK-END
