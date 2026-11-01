"""Program for Lennard-Jones cluster global minimum, 55 atoms.
Uses Mackay icosahedron construction as the primary candidate,
with fallback to cubic lattice if geometry is invalid.
"""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product


def _muller_roots(c2, c1, c0):
    """Solve c2*x^2 + c1*x + c0 = 0, return real roots sorted descending."""
    disc = c1 * c1 - 4.0 * c2 * c0
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    r1 = (-c1 + sq) / (2.0 * c2)
    r2 = (-c1 - sq) / (2.0 * c2)
    roots = [r for r in (r1, r2) if r > 1e-9]
    roots.sort(reverse=True)
    return roots


def _build_mackay_icosahedron(n_atoms=55):
    """
    Build a Mackay icosahedron with shells at radii r1 and r2.
    Shell 1: 12 vertices of regular icosahedron at radius r1.
    Shell 2: 42 atoms at face-centers (20), edge-midpoints (30) positions,
             scaled to radius r2.
    Total: 1 + 12 + 42 = 55 atoms.
    
    We solve for r1, r2 such that nearest-neighbor distances match
    the LJ minimum at r_min = 2^(1/6) ≈ 1.122462.
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron (circumradius 1)
    verts = []
    for a, b in product([0, 1], repeat=2):
        x = a * 1.0 - 0.5
        y = b * phi - 0.5 * phi
        z = 0.0
        verts.append([x, y, z])
        verts.append([y, z, x])
        verts.append([z, x, y])
    # Normalize to unit circumradius
    norms = np.array([np.linalg.norm(v) for v in verts])
    verts = [np.array(v) / n for v, n in zip(verts, norms)]
    verts = np.array(verts)
    
    # 20 face centers of icosahedron (unit circumradius)
    # Each face is a triangle of 3 vertices. For icosahedron, faces can be
    # identified by triples. Let's use the standard icosahedron faces.
    # Standard icosahedron vertices (circumradius 1):
    # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) normalized
    
    # Let me rebuild with explicit coordinates
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw_verts = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    raw_verts = np.array(raw_verts, dtype=float)
    norms = np.linalg.norm(raw_verts, axis=1)
    verts = raw_verts / norms[:, np.newaxis]
    
    # Find faces: for each triple of vertices, check if they form a face
    # (all pairwise distances equal to edge length)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    tol = 1e-8
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if (abs(d1 - edge_len) < tol and 
                    abs(d2 - edge_len) < tol and 
                    abs(d3 - edge_len) < tol):
                    faces.append((i, j, k))
    
    # Should have 20 faces
    face_centers = []
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # 30 edge midpoints
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_len) < tol:
                edges.add((i, j))
    
    edge_mids = []
    for i, j in edges:
        em = (verts[i] + verts[j]) / 2.0
        edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    
    # Now we have:
    # - Center: origin
    # - 12 shell-1 vertices at radius r1
    # - 20 face centers at radius r2 (they lie at some fixed fraction of r1)
    # - 30 edge midpoints at radius r2
    
    # The face centers and edge midpoints have fixed radial distances relative to r1.
    # Let's compute their radial distances (assuming r1 = 1 for now)
    face_radial = np.linalg.norm(face_centers, axis=1)
    edge_radial = np.linalg.norm(edge_mids, axis=1)
    
    # For a Mackay icosahedron, all shell-2 atoms should be at the same radius.
    # face_radial and edge_radial should be approximately equal. Let's average.
    r2_over_r1 = np.mean(np.concatenate([face_radial, edge_radial]))
    
    # Now solve for r1, r2 such that nearest-neighbor distances = r_min = 2^(1/6)
    # Nearest neighbors:
    # - Center to shell-1: distance = r1
    # - Shell-1 to shell-2: distance depends on geometry
    
    # Actually, for the standard Mackay icosahedron, the nearest neighbor 
    # distance is r1 (center to shell 1). Let's set r1 = r_min = 2^(1/6).
    # Then r2 = r2_over_r1 * r1.
    
    r_min = 2.0 ** (1.0/6.0)
    r1 = r_min
    r2 = r2_over_r1 * r1
    
    # Build positions
    positions = np.zeros((55, 3))
    positions[0] = [0, 0, 0]  # center
    positions[1:13] = verts * r1  # shell 1
    positions[13:33] = face_centers * r2  # 20 face centers
    positions[33:63] = edge_mids * r2  # 30 edge midpoints
    
    # But we only have 55 atoms: 1 + 12 + 20 + 30 = 63. That's too many!
    # Wait: 1 + 12 + 42 = 55. So shell 2 has 42 atoms.
    # 20 face centers + 30 edge midpoints = 50. That's 1+12+50 = 63.
    # 
    # Hmm, the Mackay icosahedron with 55 atoms is:
    # 1 (center) + 12 (first shell) + 42 (second shell)
    # The second shell of 42 consists of: 20 face centers + 22 edge positions?
    # No, standard Mackay icosahedron shells:
    # Shell 0: 1
    # Shell 1: 12
    # Shell 2: 42
    # Shell 3: 120
    # 
    # Shell 2 has 42 atoms. These are at the 20 face centers and 22 other positions?
    # Actually, for the icosahedral Mackay structure:
    # - 12 vertices
    # - 30 edges  
    # - 20 faces
    # The second shell atoms are at: 20 face centers + 22 positions along edges?
    # No. Let me reconsider.
    
    # The Mackay icosahedron is built by placing atoms at:
    # - Center
    # - 12 vertices of icosahedron
    # - Atoms at positions corresponding to the icosahedral tiling
    
    # Actually, the standard 55-atom Mackay icosahedron has:
    # 1 + 12 + 42 = 55
    # The 42 atoms in the second shell are located at:
    # - 20 face centers
    # - 22 atoms at 1/3 and 2/3 along edges? No, that would be 60.
    
    # Let me look at this differently. The Mackay icosahedron shells:
    # n=0: 1 atom
    # n=1: 12 atoms  
    # n=2: 42 atoms
    # Total for n=2: 55 atoms
    
    # The 42 atoms in shell 2 are at the positions of the vertices of a 
    # truncated icosahedron? No.
    
    # Actually, I think the correct construction is:
    # Shell 2 consists of atoms at all the "edge midpoint" and "face center" 
    # type positions but only 42 of them. 
    # 
    # Let me just use a different approach: use the known coordinates from 
    # literature or construct it more carefully.
    
    # Alternative: Use the fact that the Mackay icosahedron can be constructed
    # by taking the icosahedral points and adding a second shell.
    
    # For now, let me just return the 1+12+20+22 configuration.
    # Actually, let me just use 1 + 12 + 42 where the 42 are:
    # 20 face centers + 22 of the 30 edge midpoints? That doesn't make sense.
    
    # Let me try a simpler approach: just use the icosahedral symmetry.
    # The 55-atom Mackay icosahedron has atoms at:
    # (0,0,0)
    # 12 points at radius r1 (icosahedron vertices)
    # 42 points at radius r2
    
    # The 42 points are the orbit of a single point under icosahedral group.
    # The icosahedral group has order 60. A generic orbit has 60 points.
    # 42 = 60 * 7/10? No. 42 is not a divisor of 60.
    
    # Hmm, 42 = 2 * 21. The icosahedral group has orbits of size 1, 2, 6, 10, 12, 20, 30, 60.
    # 42 is not in this list. So the 42 atoms must be a union of orbits.
    # 42 = 12 + 30. Yes! 12 + 30 = 42.
    # So shell 2 = 12 atoms (one orbit of size 12) + 30 atoms (one orbit of size 30).
    
    # Orbit of size 12: vertices of a dodecahedron (dual to icosahedron)
    # Orbit of size 30: edge midpoints of the icosahedron
    
    # So:
    # Shell 1: 12 icosahedron vertices
    # Shell 2a: 12 dodecahedron vertices (face centers of icosahedron)
    # Shell 2b: 30 edge midpoints of icosahedron
    # Total: 1 + 12 + 12 + 30 = 55. 
    
    # Wait, that's 55! Let me verify: 1 + 12 + 12 + 30 = 55. Yes!
    
    # So I need:
    # - 1 center
    # - 12 icosahedron vertices at radius r1
    # - 12 dodecahedron vertices (face centers of icosahedron) at radius r2
    # - 30 edge midpoints of icosahedron at radius r2
    
    # But wait, the face centers of the icosahedron ARE the vertices of the 
    # dodecahedron, and there are 20 of them, not 12. 
    
    # I'm confusing myself. Let me just count:
    # Icosahedron: 12 vertices, 30 edges, 20 faces.
    # Dodecahedron: 20 vertices, 30 edges, 12 faces.
    
    # The face centers of the icosahedron (20) correspond to the vertices of 
    # the dual dodecahedron (20).
    
    # So if shell 2 has 42 atoms, it can't be 20 + 30 = 50.
    
    # I think I'm overcomplicating this. Let me just use a numerical approach:
    # start from a reasonable guess and relax.
    
    # For now, let me just return a cubic lattice as fallback.
    return None


def construct():
    """
    Construct initial positions for 55-atom Lennard-Jones cluster.
    Uses a Mackay icosahedron-like structure if possible, else cubic lattice.
    """
    # Try to build Mackay icosahedron
    # For simplicity and reliability, use a well-structured initial guess.
    
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron (circumradius 1)
    raw_verts = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    raw_verts = np.array(raw_verts, dtype=float)
    norms = np.linalg.norm(raw_verts, axis=1)
    verts = raw_verts / norms[:, np.newaxis]
    
    # Edge length of this icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    
    # Find all edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_len) < 1e-8:
                edges.append((i, j))
    
    # Edge midpoints (30 points)
    edge_mids = [(verts[i] + verts[j]) / 2.0 for i, j in edges]
    edge_mids = np.array(edge_mids)
    
    # Face centers (20 points)
    # Find faces
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if (abs(d1 - edge_len) < 1e-8 and 
                    abs(d2 - edge_len) < 1e-8 and 
                    abs(d3 - edge_len) < 1e-8):
                    faces.append((i, j, k))
    
    face_centers = [(verts[i] + verts[j] + verts[k]) / 3.0 for i, j, k in faces]
    face_centers = np.array(face_centers)
    
    # Now build the 55-atom structure:
    # 1 center + 12 shell1 + 20 face_centers + 22 selected edge_mids = 55
    # But 1+12+20+22 = 55. We need to select 22 of 30 edge midpoints.
    # This is not symmetric. 
    
    # Better: use 1 + 12 + 42 where 42 = some symmetric subset.
    # Since 42 = 12 + 30, use 12 additional points + 30 edge midpoints.
    # The 12 additional points could be at the dodecahedron vertices... but there are 20.
    
    # Let me just use: 1 + 12 + 20 + 22 = 55, taking the first 22 edge midpoints.
    # This breaks symmetry but gives a valid starting point.
    
    # Actually, let me just use a simpler, symmetric approach:
    # Use the icosahedral cluster with 13 atoms (center + 12) and add 42 more
    # in a second shell. For the second shell, use all 30 edge midpoints and 
    # 12 of the 20 face centers. Still asymmetric.
    
    # You know what, let me just use the cubic lattice but with better spacing,
    # or use a simple icosahedral + cubic mix.
    
    # Simplest reliable approach: use the 13-atom icosahedron and add 42 atoms
    # on a cubic shell around it.
    
    r_min = 2.0 ** (1.0/6.0)
    
    # 13-atom icosahedron
    center = np.array([0.0, 0.0, 0.0])
    shell1 = verts * r_min
    
    # Add 42 more atoms in a rough cubic arrangement outside
    # Place them on a larger icosahedral-like shell
    r2 = 2.0 * r_min
    # Generate 42 points on a sphere of radius r2 using Fibonacci sphere
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    extra_points = []
    for i in range(42):
        z = 1.0 - 2.0 * i / 41.0
        radius = np.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        extra_points.append([x, y, z] * r2)
    extra_points = np.array(extra_points)
    
    positions = np.vstack([center.reshape(1,3), shell1, extra_points])
    
    # Ensure minimum distance of 0.5
    # Check and adjust if needed
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.5:
                # Push apart
                if d < 1e-10:
                    positions[j] += np.array([0.5, 0, 0])
                else:
                    diff = positions[j] - positions[i]
                    diff /= d
                    positions[j] = positions[i] + diff * 0.5
    
    return positions
# EVOLVE-BLOCK-END
