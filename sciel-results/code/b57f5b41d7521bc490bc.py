"""Lennard-Jones cluster N=55: construct two-shell Mackay icosahedron and relax."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

# Icosahedron geometry
PHI = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unnormalized)."""
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for s3 in (-1, 1):
                # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
                pass
    # Standard 12 vertices:
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pass
    # Let's just list them explicitly
    coords = [
        (0, 1, PHI), (0, -1, PHI), (0, 1, -PHI), (0, -1, -PHI),
        (1, PHI, 0), (-1, PHI, 0), (1, -PHI, 0), (-1, -PHI, 0),
        (PHI, 0, 1), (-PHI, 0, 1), (PHI, 0, -1), (-PHI, 0, -1),
    ]
    arr = np.array(coords, dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(arr, axis=1)
    arr = arr / norms[:, None]
    return arr

def _lj_energy(pos):
    """Compute total Lennard-Jones energy 4*sum(r^-12 - r^-6)."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    r6 = d ** (-6)
    return 4.0 * np.sum(r6 * r6 - r6)

def _relax(pos, max_iter=5000, tol=1e-10):
    """L-BFGS local relaxation using scipy."""
    from scipy.optimize import minimize
    
    def fun(x):
        p = x.reshape(-1, 3)
        # Ensure minimum separation constraint softly
        d = cdist(p, p)
        np.fill_diagonal(d, 1.0)
        # Penalty for too-close pairs
        close = np.maximum(0, 0.5 - d)
        penalty = 1e6 * np.sum(close ** 2)
        r6 = d ** (-6)
        e = 4.0 * np.sum(r6 * r6 - r6) + penalty
        return e
    
    def jac(x):
        p = x.reshape(-1, 3)
        d = cdist(p, p)
        np.fill_diagonal(d, 1.0)
        # Gradient of LJ energy
        # dE/dr = 4*(-12*r^-13 + 6*r^-7)
        # Use numerical or analytical
        # For simplicity, use finite difference step
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            x0 = x[i]
            x[i] = x0 + eps
            f1 = fun(x)
            x[i] = x0 - eps
            f2 = fun(x)
            x[i] = x0
            g[i] = (f1 - f2) / (2 * eps)
        return g
    
    x0 = pos.flatten()
    result = minimize(fun, x0, method='L-BFGS-B', jac=None, options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s
    
    # Build two-shell Mackay icosahedron
    # Shell 1: 12 vertices of icosahedron at radius r1
    # Shell 2: 42 vertices (12 face centers + 30 edge midpoints) at radius r2
    
    # For N=55 Mackay icosahedron:
    # Inner shell: 12 atoms at icosahedron vertices
    # Outer shell: 42 atoms
    
    # The Mackay icosahedron of order 2 has:
    # - 12 vertices (icosahedron corners)
    # - 30 edge midpoints  
    # - 20 face centers
    # Total: 12 + 30 + 20 = 62... but we need 55
    
    # Actually for N=55, the structure is:
    # 12 + 42 = 55 where 42 = 12 (face centers of a smaller icosahedron?) 
    # Let me reconsider. The Mackay icosahedron with n=2 has 12(n^3 - 3n^2 + 2n + 1)/2... 
    # Actually the formula for Mackay icosahedron of order n is N = 12(n^3 - 3n^2 + 2n + 1)/2 + 1?
    # No. Let me think differently.
    
    # For N=55, the known structure is a two-shell icosahedron:
    # Inner: 12 atoms (icosahedron vertices) 
    # Outer: 42 atoms
    
    # The 42 outer atoms can be placed at:
    # - 12 positions above each face center direction (but that's only 12)
    # Actually, let me use a different construction.
    
    # Better approach: Use the known geometry.
    # Inner shell: 12 atoms at icosahedron vertices, radius ~1.12
    # Outer shell: 42 atoms at radius ~2.2
    
    # For the outer shell, we can place atoms at:
    # - 12 positions: one above each vertex (along vertex normal) - no, that's not right
    
    # Let me use a simpler but effective approach:
    # Start with icosahedron of 12 atoms, then add 42 more at appropriate positions.
    
    # Actually, the cleanest approach for N=55:
    # The Mackay icosahedron order 2 has shells:
    # Shell 0: 1 atom at center (but we don't have one for 55)
    # For 55 atoms without center: 12 + 42
    
    # Let me construct:
    # Inner 12: icosahedron vertices at radius r1
    # Outer 42: distributed on a sphere of radius r2
    
    # For the 42 outer atoms, a good distribution:
    # 12 at face-center directions, 30 at edge-midpoint directions
    
    # Get icosahedron vertices
    ico_verts = _icosahedron_vertices()  # 12 vertices on unit sphere
    
    # Get face centers of icosahedron (20 faces)
    # Each face is a triangle of 3 vertices. Need to find all 20 faces.
    # For a regular icosahedron, the faces can be found by checking which triples of vertices form equilateral triangles.
    
    # Alternative: use known face centers
    # The 20 face centers of a regular icosahedron with vertices as above:
    # Let me compute them by finding all triangles
    
    # Find all faces (triangles) of the icosahedron
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[k] - ico_verts[i])
                # Check if approximately equilateral
                if abs(d1 - d2) < 1e-6 and abs(d2 - d3) < 1e-6:
                    faces.append((i, j, k))
    
    # Get unique face centers
    face_centers = []
    for f in faces:
        center = (ico_verts[f[0]] + ico_verts[f[1]] + ico_verts[f[2]]) / 3.0
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    face_centers = np.array(face_centers)
    # Should have 20 unique face centers
    
    # Get edge midpoints
    # Edges of icosahedron: 30 edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            # Edge length of unit icosahedron
            if d < 1.1:  # roughly the edge length
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edges.append(mid)
    edges = np.array(edges)
    # Should have 30 unique edge midpoints
    
    # Now construct the 55-atom structure:
    # Inner shell: 12 atoms at icosahedron vertices
    # Outer shell: 42 atoms = 12 (at vertex directions, farther out) + 20 (face centers) + 10 (some edges)?
    # That doesn't add up nicely.
    
    # Let me try: inner 12 at vertices (radius r1), outer 42 at:
    # 20 face centers + 12 vertex directions (extended) + 10 edge midpoints? = 42? No, 20+12+10=42. But why 10 edges?
    
    # Alternative simpler approach: just use a good initial guess and let relaxation do the work.
    # Place 55 atoms on a two-shell icosahedral-like structure.
    
    # Let me try a different, more robust approach:
    # Start with a reasonable random-ish icosahedral distribution and relax.
    
    # Actually, let me just build a solid icosahedral-like cluster:
    # Use the icosahedron vertices for inner shell, and for outer shell use
    # face centers and edge midpoints.
    
    # For N=55: 12 inner + 42 outer
    # Outer 42: 20 face centers + 12 "vertex extensions" + 10 "edge extensions"
    # But this is getting complicated. Let me use a simpler strategy.
    
    # Strategy: Build a two-shell structure with icosahedral symmetry.
    # Inner: 12 atoms at icosahedron vertices, radius r1 = 1.12
    # Outer: 42 atoms distributed on a sphere of radius r2 = 2.2
    
    # For the outer 42, use a Fibonacci sphere or icosahedral distribution.
    # Better: use the 20 face centers + 12 vertex directions + 10 edge midpoint directions
    
    # Actually, let me just use a simple but effective initialization:
    # 12 atoms at icosahedron vertices (inner shell)
    # 42 atoms at: 20 face center directions + 12 vertex directions + 10 random-ish
    
    # Hmm, let me try yet another approach. Use the fact that for N=55,
    # a good initial guess is to place atoms at all positions in a small icosahedral lattice.
    
    # Simplest effective approach: 
    # - 12 atoms at icosahedron vertices (radius ~1.1)
    # - 42 atoms: 12 at vertex directions (radius ~2.2), 20 at face center directions (radius ~2.2), 10 at edge midpoint directions (radius ~2.2)
    
    # Wait, 12 + 20 + 10 = 42. Let me use 10 edge midpoints (out of 30).
    
    # Actually, I think the cleanest approach is:
    # Inner shell: 12 atoms, icosahedron vertices, radius r1
    # Outer shell: 42 atoms
    #   - 12 atoms along vertex directions at radius r2
    #   - 20 atoms along face-center directions at radius r2  
    #   - 10 atoms along 10 of the 30 edge-midpoint directions at radius r2
    
    # But which 10 edges? For symmetry, maybe pick a subset.
    
    # Let me just try a different tactic: use a well-known parameterization.
    
    # For the Mackay icosahedron of order 2 (which has 62 atoms including center),
    # removing the center gives 61, not 55. So N=55 is NOT a full Mackay icosahedron.
    
    # Let me look at this differently. The task says "the best-known putative minimum is the 
    # complete two-shell Mackay icosahedron (energy -279.248470)". So it IS a two-shell Mackay icosahedron.
    
    # Mackay icosahedron of order n has N = 12(n^3 - 3n^2 + 2n + 1)/2 + 1... 
    # For n=1: 12(1-3+2+1)/2 + 1 = 12(1)/2 + 1 = 7. That's a 7-atom cluster (center + 6 octahedron).
    # Hmm, that's not right either.
    
    # Let me just use a practical approach:
    # 1. Create a good initial guess using icosahedral geometry
    # 2. Relax with L-BFGS
    # 3. Return the result
    
    # Initial guess: two shells
    r1 = 1.12  # inner shell radius
    r2 = 2.2   # outer shell radius
    
    # Inner: 12 icosahedron vertices
    inner = ico_verts * r1
    
    # Outer: 42 atoms
    # Use 12 vertex directions + 20 face center directions + 10 edge midpoint directions
    outer_dirs = []
    outer_dirs.extend(ico_verts)  # 12 vertex directions
    outer_dirs.extend(face_centers)  # 20 face center directions
    # Need 10 more from edge midpoints
    if len(edges) >= 10:
        outer_dirs.extend(edges[:10])
    
    outer_dirs = np.array(outer_dirs)
    # Normalize
    norms = np.linalg.norm(outer_dirs, axis=1)
    outer_dirs = outer_dirs / norms[:, None]
    outer = outer_dirs * r2
    
    # Combine
    pos = np.vstack([inner, outer])
    
    # Verify we have 55 atoms
    assert len(pos) == 55, f"Expected 55 atoms, got {len(pos)}"
    
    # Check minimum separation
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    min_d = np.min(d)
    if min_d < 0.5:
        # Scale up if needed
        scale = 0.5 / min_d * 1.01
        pos = pos * scale
    
    # Relax
    t_elapsed = time.time() - t_start
    if t_elapsed < deadline - 5.0:
        try:
            pos_relaxed, e_relaxed = _relax(pos, max_iter=3000)
            # Check validity
            d = cdist(pos_relaxed, pos_relaxed)
            np.fill_diagonal(d, 1.0)
            if np.min(d) >= 0.5 - 1e-6:
                pos = pos_relaxed
        except Exception:
            pass
    
    # Final check
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 0.5:
        # Fallback: scale up
        scale = 0.5 / np.min(d) * 1.01
        pos = pos * scale
    
    return pos
# EVOLVE-BLOCK-END
