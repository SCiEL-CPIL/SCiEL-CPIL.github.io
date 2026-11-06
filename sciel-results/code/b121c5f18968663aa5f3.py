"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initial Configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf

    def compute_circumradius(points):
        """
        Computes the exact covering radius by finding the convex hull,
        identifying spherical triangles (facets), and calculating their
        spherical circumradii.
        """
        try:
            # ConvexHull in 3D gives triangular facets for points in general position on a sphere
            hull = ConvexHull(points)
            simplices = hull.simplices
        except Exception:
            return np.inf, np.zeros(N)
            
        max_r = 0.0
        # We need to accumulate gradients for the optimization step
        # But for the evaluation function, we just need the max radius
        # We'll compute gradients separately in the optimization loop for efficiency
        
        for idx in simplices:
            p1, p2, p3 = points[idx]
            
            # Vectors from p1 to p2 and p3
            v1 = p2 - p1
            v2 = p3 - p1
            
            # Spherical circumcenter logic:
            # The circumcenter C is equidistant from p1, p2, p3 on the sphere.
            # This implies C is in the plane spanned by p1, p2, p3 (if they are not antipodal).
            # Actually, for a spherical triangle, the circumcenter is the intersection of the sphere
            # and the plane defined by the points? No.
            # 
            # Let's use the property: The circumcenter $C$ satisfies $C \cdot p_1 = C \cdot p_2 = C \cdot p_3$.
            # This means $C$ is orthogonal to $(p_2 - p_1)$ and $(p_3 - p_1)$.
            # So $C$ is parallel to the cross product $(p_2 - p_1) \times (p_3 - p_1)$.
            # Wait, that's the normal to the plane containing the chord triangle.
            # The circumcenter of a spherical triangle lies on the great circle perpendicular to the plane of the chord triangle?
            # No.
            
            # Let's derive it properly.
            # We want $C$ such that $|C|^2=1$ and $C \cdot p_i = k$ for i=1,2,3.
            # This implies $C$ is in the span of $p_1, p_2, p_3$.
            # Let $C = a p_1 + b p_2 + c p_3$.
            # $C \cdot p_1 = a + b(p_2 \cdot p_1) + c(p_3 \cdot p_1) = k$
            # $C \cdot p_2 = a(p_1 \cdot p_2) + b + c(p_3 \cdot p_2) = k$
            # $C \cdot p_3 = a(p_1 \cdot p_3) + b(p_2 \cdot p_3) + c = k$
            #
            # Subtracting equations:
            # $(a-1) + b u + c v = 0$ where $u = p_2 \cdot p_1, v = p_3 \cdot p_1$
            # $a u + (b-1) + c w = 0$ where $w = p_3 \cdot p_2$
            # $a v + b w + (c-1) = 0$
            #
            # This linear system for a,b,c can be solved.
            # Let $A = \begin{pmatrix} 1 & u & v \\ u & 1 & w \\ v & w & 1 \end{pmatrix}$
            # We want $A \begin{pmatrix} a \\ b \\ c \end{pmatrix} = \begin{pmatrix} k \\ k \\ k \end{pmatrix}$.
            # So $\mathbf{x} = A^{-1} \mathbf{k}$.
            # Note that $\mathbf{1}^T A \mathbf{1} = 3 + 2(u+v+w)$.
            #
            # Once we have $C$, we normalize it to unit length: $C_{norm} = C / |C|$.
            # The circumradius is the angular distance from $C_{norm}$ to $p_1$.
            # $\cos(R) = C_{norm} \cdot p_1 = (C \cdot p_1) / |C| = k / |C|$.
            # So $R = \arccos(k / |C|)$.
            
            u = np.dot(p1, p2)
            v = np.dot(p1, p3)
            w = np.dot(p2, p3)
            
            A = np.array([
                [1.0, u, v],
                [u, 1.0, w],
                [v, w, 1.0]
            ])
            
            # Solve A x = [k, k, k]. We can set k=1 for direction, then scale.
            # Actually, the ratio $k/|C|$ is invariant to scaling of k.
            # So let's solve $A \mathbf{x} = \mathbf{1}$.
            try:
                coeffs = np.linalg.solve(A, np.ones(3))
            except np.linalg.LinAlgError:
                # Degenerate case (points collinear or antipodal), fallback to max edge
                d12 = np.arccos(np.clip(u, -1, 1))
                d13 = np.arccos(np.clip(v, -1, 1))
                d23 = np.arccos(np.clip(w, -1, 1))
                return max(d12, d13, d23), np.zeros(N)
                
            C = coeffs[0]*p1 + coeffs[1]*p2 + coeffs[2]*p3
            norm_C = np.linalg.norm(C)
            
            if norm_C < 1e-10:
                 # Degenerate
                 d12 = np.arccos(np.clip(u, -1, 1))
                 d13 = np.arccos(np.clip(v, -1, 1))
                 d23 = np.arccos(np.clip(w, -1, 1))
                 return max(d12, d13, d23), np.zeros(N)
                 
            k_val = np.dot(C, p1)
            cos_R = k_val / norm_C
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            if R > max_r:
                max_r = R
                
        return max_r, np.zeros(N)

    def objective_and_grad(x_flat):
        """
        Objective: Max circumradius of the convex hull facets.
        We use a smooth approximation for gradient descent:
        Soft-max of the circumradii of the top K facets.
        """
        points = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
        except Exception:
            return 10.0, np.zeros_like(x_flat)
            
        radii = np.zeros(len(simplices))
        # Precompute gradients for each facet
        # Gradient of arccos(k/|C|) with respect to points is complex.
        # Instead, we use a finite difference or a simpler heuristic gradient.
        # For a robust approach within time limits, let's use a "soft max" of the radii
        # and compute gradients via finite differences or analytical approximation.
        
        # Analytical gradient of spherical circumradius is difficult.
        # Let's use a trick: The circumcenter C is the point on the sphere maximizing distance to the 3 vertices.
        # Actually, for the covering radius, we are interested in the maximum over all y on the sphere of min_i d(y, p_i).
        # This maximum is attained at the circumcenters of the Delaunay triangles.
        
        # To make this differentiable, we use a soft-max over the circumradii.
        # But computing the gradient of the circumradius of a spherical triangle w.r.t. its vertices is non-trivial.
        
        # Alternative Strategy:
        # Use the fact that the covering radius is the max over facets of the circumradius.
        # We can use a "worst-case" gradient: identify the facet with the largest radius,
        # and move the points to reduce that specific radius.
        # How to reduce circumradius of a spherical triangle?
        # Move the vertices closer to the circumcenter? No, that might increase it if they move away from each other.
        # The circumcenter is the center of the minimal enclosing circle on the sphere.
        # To minimize the radius, we should move the vertices such that the circumcenter moves towards them?
        # Actually, if we move a vertex towards the circumcenter, the radius decreases?
        # Let's test: Equilateral triangle. Move one vertex towards center. The triangle becomes isosceles.
        # The circumradius might increase or decrease depending on the shape.
        
        # Let's use a simpler, robust method:
        # 1. Find the worst facet.
        # 2. Calculate the circumcenter C.
        # 3. The gradient of the distance from C to p_i is (p_i - C)/|p_i - C| ? No.
        # The distance is arccos(p_i . C).
        # d/dp_i arccos(p_i . C) = - (C - p_i(p_i . C)) / sqrt(1 - (p_i . C)^2)
        # This is the gradient of the angular distance from p_i to C.
        # If we minimize the angular distance from p_i to C, we are moving p_i towards C.
        # If we move all 3 vertices towards C, the triangle shrinks, and the circumradius decreases.
        
        # So, for the worst facet, we apply a force pushing each vertex towards the circumcenter.
        
        max_r_idx = -1
        max_r_val = 0.0
        worst_C = None
        worst_indices = None
        
        for idx in simplices:
            p1, p2, p3 = points[idx]
            u = np.dot(p1, p2)
            v = np.dot(p1, p3)
            w = np.dot(p2, p3)
            
            A = np.array([
                [1.0, u, v],
                [u, 1.0, w],
                [v, w, 1.0]
            ])
            
            try:
                coeffs = np.linalg.solve(A, np.ones(3))
            except np.linalg.LinAlgError:
                continue
                
            C = coeffs[0]*p1 + coeffs[1]*p2 + coeffs[2]*p3
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-10:
                continue
            k_val = np.dot(C, p1)
            cos_R = k_val / norm_C
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            if R > max_r_val:
                max_r_val = R
                max_r_idx = idx[0] # Just for tracking
                worst_C = C / norm_C
                worst_indices = idx
                
        if worst_indices is None:
            return 10.0, np.zeros_like(x_flat)
            
        # Gradient calculation
        # We want to minimize max_r_val.
        # Approximate gradient: push vertices of the worst facet towards worst_C.
        grad = np.zeros_like(x_flat)
        
        # Force strength
        force = 1.0
        
        for i in worst_indices:
            p = points[i]
            # Gradient of angular distance d(p, C)
            # d = arccos(p . C)
            # grad_p d = - (C - p(p . C)) / sin(d)
            dot_pc = np.dot(p, worst_C)
            sin_d = np.sqrt(1.0 - dot_pc*dot_pc)
            if sin_d < 1e-6:
                continue
            grad_vec = -(worst_C - p*dot_pc) / sin_d
            grad[i*3:(i+1)*3] += force * grad_vec
            
        # To make it smoother and avoid getting stuck in local minima of the "worst facet" selection,
        # we can blend with a global energy minimization (e.g. Tammes) or just use this.
        # This is a subgradient method for the max function.
        
        return max_r_val, grad

    # Optimization Loop
    # We use L-BFGS-B on the flattened points.
    # We need to handle the sphere constraint. L-BFGS-B works in Euclidean space.
    # We can project onto the sphere after each step, or use a penalty.
    # Better: Use a parameterization? No, 50 points x 3 coords is fine.
    # We'll use a simple gradient descent with projection.
    
    x = points.flatten().copy()
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(x)
    
    for step in range(500):
        if time.time() > deadline:
            break
            
        val, grad = objective_and_grad(x)
        
        if val < best_radius:
            best_radius = val
            best_points = x.reshape(N, 3)
            
        # Update with momentum
        v = momentum * v - lr * grad
        x = x + v
        
        # Project onto sphere
        points_curr = x.reshape(N, 3)
        norms = np.linalg.norm(points_curr, axis=1, keepdims=True)
        points_curr = points_curr / norms
        x = points_curr.flatten()
        
        # Adaptive learning rate
        if step % 50 == 0:
            lr *= 0.95
            
    # Final check
    final_points = best_points
    # Normalize
    norms = np.linalg.norm(final_points, axis=1, keepdims=True)
    final_points = final_points / norms
    
    return final_points

# EVOLVE-BLOCK-END
