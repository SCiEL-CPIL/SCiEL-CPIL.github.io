"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay

def construct():
    """
    Place 130 points on the unit sphere to minimize the covering radius.
    Uses a geometric gradient descent on the maximum circumradius of the
    Delaunay triangulation facets.
    """
    
    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_facet_circumcenters(pts):
        """
        Compute the circumcenters and circumradii for all facets of the convex hull.
        Returns:
        centers: array of shape (n_facets, 3)
        radii: array of shape (n_facets,)
        facet_indices: list of tuples of vertex indices
        """
        tri = Delaunay(pts)
        # Get the simplices (facets)
        # Delaunay in 3D returns tetrahedra, but we only want the surface facets.
        # A facet is on the convex hull if it is part of only one tetrahedron.
        # We can find unique triangles from the tetrahedra.
        
        simplices = tri.simplices
        # Each simplex is 4 points. We need to extract the 4 faces of each tetrahedron.
        # Faces of tetra (0,1,2,3) are (0,1,2), (0,1,3), (0,2,3), (1,2,3).
        # We collect all faces and count how many times each appears.
        # Faces appearing once are on the convex hull.
        
        faces = []
        for s in simplices:
            faces.append((s[0], s[1], s[2]))
            faces.append((s[0], s[1], s[3]))
            faces.append((s[0], s[2], s[3]))
            faces.append((s[1], s[2], s[3]))
            
        # Count occurrences
        face_counts = {}
        for f in faces:
            # Normalize order for counting
            nf = tuple(sorted(f))
            if nf in face_counts:
                face_counts[nf] += 1
            else:
                face_counts[nf] = 1
                
        hull_faces = []
        for f, count in face_counts.items():
            if count == 1:
                # Restore original order? The order doesn't matter for circumcenter, 
                # but for gradient direction consistency, it's good.
                # We just need the 3 points.
                hull_faces.append(list(f))
                
        centers = []
        radii = []
        facet_indices = []
        
        for f in hull_faces:
            v1, v2, v3 = pts[f[0]], pts[f[1]], pts[f[2]]
            
            # Compute circumcenter of spherical triangle
            # The circumcenter c is the normalized vector:
            # c = normalize( v1 * sin(A) + v2 * sin(B) + v3 * sin(C) ) ? No.
            # c is the solution to c.v1 = c.v2 = c.v3 = cos(R)
            # c is in the plane spanned by v1, v2, v3.
            # c = a*v1 + b*v2 + c*v3
            # c.v1 = c.v2 => a + b*dot(v1,v2) + c*dot(v1,v3) = b + a*dot(v1,v2) + c*dot(v2,v3)
            # (a-b) (1 - dot(v1,v2)) + c (dot(v1,v3) - dot(v2,v3)) = 0
            # This is a linear system for a,b,c.
            # Alternatively, use the cross product formula for the circumcenter of a triangle on a sphere.
            # The circumcenter is proportional to:
            # v1 * (1 - dot(v2, v3)) + v2 * (1 - dot(v1, v3)) + v3 * (1 - dot(v1, v2))
            # Let's verify.
            # If v1, v2, v3 are equidistant, dot(v2,v3) etc are equal, so sum is (v1+v2+v3). Correct.
            
            d12 = np.dot(v1, v2)
            d13 = np.dot(v1, v3)
            d23 = np.dot(v2, v3)
            
            # Clamp for stability
            d12 = np.clip(d12, -1.0, 1.0)
            d13 = np.clip(d13, -1.0, 1.0)
            d23 = np.clip(d23, -1.0, 1.0)
            
            w1 = 1.0 - d23
            w2 = 1.0 - d13
            w3 = 1.0 - d12
            
            c = w1 * v1 + w2 * v2 + w3 * v3
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                # Degenerate case, use centroid
                c = (v1 + v2 + v3) / 3.0
                norm_c = np.linalg.norm(c)
                if norm_c < 1e-10:
                    continue
            c /= norm_c
            
            # Circumradius
            # R = arccos(c . v1)
            cos_r = np.clip(np.dot(c, v1), -1.0, 1.0)
            r = np.arccos(cos_r)
            
            centers.append(c)
            radii.append(r)
            facet_indices.append(f)
            
        return np.array(centers), np.array(radii), facet_indices

    # Initial configuration
    pts = fibonacci_sphere(N)
    
    # Optimization loop
    lr = 0.05
    n_iter = 2000
    
    for _ in range(n_iter):
        centers, radii, facet_indices = compute_facet_circumcenters(pts)
        
        if len(radii) == 0:
            break
            
        max_r = np.max(radii)
        
        # Find facets with max radius
        max_idx = np.where(np.abs(radii - max_r) < 1e-8)[0]
        
        # Compute gradient for each point
        # grad[i] is the direction to move point i
        grad = np.zeros_like(pts)
        
        for idx in max_idx:
            f = facet_indices[idx]
            c = centers[idx]
            # For each vertex in the facet, add the vector from vertex to center
            # This moves the vertex towards the center, reducing the distance to the center?
            # Wait. If we move v towards c, does R decrease?
            # R is the distance from c to v. If we move v to c, R becomes 0.
            # But c also moves.
            # The gradient of R w.r.t v is roughly (c - v) / |c - v|?
            # Actually, for small steps, moving v towards c reduces the circumradius.
            for v_idx in f:
                grad[v_idx] += (c - pts[v_idx])
                
        # Normalize gradient
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        grad = grad / norms
        
        # Update points
        pts = pts + lr * grad
        
        # Project back to sphere
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Optional: Check for improvement
        # if _ % 100 == 0:
        #     print(f"Iter {_}: Max R {np.degrees(max_r):.4f}")
            
    return pts
# EVOLVE-BLOCK-END
