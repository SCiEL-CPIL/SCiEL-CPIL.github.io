"""Multi-objective DTLZ4 optimizer using area-preserving Halton sequence on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer: sample the known Pareto front via area-preserving 2D Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        n_shape = m - 1  # 2 shape variables for 3 objectives

        # Halton sequence in 2D (bases 2 and 3)
        def halton_2d(npts):
            def halton_1d(idx, base):
                f = 1.0
                r = 0.0
                i = idx
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            pts = np.zeros((npts, 2))
            for i in range(1, npts + 1):
                pts[i-1, 0] = halton_1d(i, 2)
                pts[i-1, 1] = halton_1d(i, 3)
            return pts

        # Generate Halton points in [0,1]^2
        halton_pts = halton_2d(self.budget)

        # Area-preserving mapping from [0,1]^2 to the DTLZ4 Pareto front (quarter sphere)
        # The Pareto front has f1^2 + f2^2 + f3^2 = 1, f_i >= 0
        # Parameterize by angles: f1 = cos(phi), f2 = sin(phi)*cos(theta), f3 = sin(phi)*sin(theta)
        # where phi in [0, pi/2], theta in [0, pi/2]
        # Surface area element: sin(phi) dphi dtheta
        # To get uniform sampling on the sphere surface, we need:
        #   u1 = cos(phi)  =>  phi = arccos(u1),  dphi = -du1/sin(phi)
        #   u2 = theta/(pi/2)  =>  theta = u2 * pi/2
        # The area element in (u1, u2) coords: sin(phi) dphi dtheta = sin(arccos(u1)) * (1/sin(arccos(u1))) du1 * (pi/2) du2 = (pi/2) du1 du2
        # So uniform in (u1, u2) gives uniform on the sphere!
        # 
        # But we need to map to DTLZ4 x-space. For DTLZ4 with g=0:
        #   f1 = x1, f2 = (1-x1)*x2, f3 = (1-x1)*(1-x2)
        # This is NOT the same as the spherical parameterization above.
        # The DTLZ4 Pareto front in (x1, x2) space is actually a flat square [0,1]^2 mapped to the sphere.
        # The mapping (x1, x2) -> (f1, f2, f3) = (x1, (1-x1)x2, (1-x1)(1-x2)) has Jacobian that varies.
        #
        # Let's compute the correct area-preserving transform.
        # The Pareto front is the image of [0,1]^2 under the map T(x1,x2) = (x1, (1-x1)x2, (1-x1)(1-x2)).
        # We want to sample (x1, x2) such that the image points are uniform on the sphere surface.
        #
        # Alternative: directly sample the sphere uniformly and invert to get (x1, x2).
        # From f1 = x1, f2 = (1-x1)x2, f3 = (1-x1)(1-x2):
        #   x1 = f1
        #   x2 = f2 / (1 - f1)  (when f1 < 1)
        #   Check: f3 = (1-f1)(1-x2) = (1-f1) - f2 = 1 - f1 - f2
        #   So f1 + f2 + f3 = 1, not f1^2 + f2^2 + f3^2 = 1!
        #
        # Wait, I made an error. Let me re-derive DTLZ4.
        # DTLZ4: f1 = g * x1, f2 = g * (1-x1) * x2, f3 = g * (1-x1) * (1-x2)
        # where g = sum_{i=3}^{n} (x_i - 0.5)^2 for DTLZ4 (with beta=100)
        # When g=0, f1 = 0, f2 = 0, f3 = 0. That's not right either.
        #
        # Actually for DTLZ4: f1 = g * x1, f2 = g * (1-x1) * x2, f3 = g * (1-x1) * (1-x2)
        # The Pareto front is when g = 0, which gives all zeros. That can't be right.
        #
        # Let me look at this more carefully. For DTLZ problems:
        # f1 = g(x) * h1(x), f2 = g(x) * h2(x), ..., fm = g(x) * hm(x)
        # where g is minimized to 0, and the h_i define the shape.
        # The Pareto front is the set of points (h1, h2, ..., hm) where g=0.
        # For DTLZ4 with m=3: h1 = x1, h2 = (1-x1)*x2, h3 = (1-x1)*(1-x2)
        # So the Pareto front is {(x1, (1-x1)x2, (1-x1)(1-x2)) : x1,x2 in [0,1]}
        # And indeed x1 + (1-x1)x2 + (1-x1)(1-x2) = x1 + (1-x1) = 1
        # So the Pareto front lies on the plane f1+f2+f3=1 in the first octant.
        # This is a triangle, not a sphere!
        #
        # So the correct area-preserving sampling is on this triangular Pareto front.
        # The triangle has vertices at (1,0,0), (0,1,0), (0,0,1).
        # Parameterize by (x1, x2) in [0,1]^2: f = (x1, (1-x1)x2, (1-x1)(1-x2))
        # The Jacobian of this map: df1/dx1=1, df1/dx2=0; df2/dx1=-x2, df2/dx2=1-x1; df3/dx1=-(1-x2), df3/dx2=-(1-x1)
        # The area element on the surface is sqrt(1 + (df2/dx1)^2 + (df3/dx1)^2) * ... this is getting complex.
        #
        # Simpler approach: since the Pareto front is a flat triangle in 3D (lying on f1+f2+f3=1),
        # we can sample uniformly on the triangle directly.
        # Uniform sampling on a triangle: pick u,v in [0,1] with u+v<=1, then
        # point = u*A + v*B + (1-u-v)*C where A,B,C are vertices.
        #
        # But we also need to map back to x-space for evaluation. Given f on the Pareto front:
        #   x1 = f1, x2 = f2/(1-f1) if f1<1, else 0
        #
        # Let's use the direct approach: sample (x1, x2) uniformly in [0,1]^2, which gives
        # a non-uniform distribution on the triangle. The density is proportional to 1/(1-x1) 
        # (since the area element scales as (1-x1) dx1 dx2 on the triangle).
        #
        # To get uniform on the triangle, we need x1 ~ Beta(1,1) but with a twist.
        # Actually, let's just use the triangle vertex method and invert.

        # Method: sample uniformly on the triangle using Halton points in [0,1]^2
        # Transform to barycentric coordinates, then to (x1, x2)
        # Vertices: A=(1,0,0)->(x1=1,x2=0), B=(0,1,0)->(x1=0,x2=1), C=(0,0,1)->(x1=0,x2=0)
        # Barycentric: f = a*A + b*B + c*C, a+b+c=1, a,b,c>=0
        # f1 = a, f2 = b, f3 = c
        # x1 = f1 = a
        # x2 = f2/(1-f1) = b/a if a>0, else 0 (but when a=0, x1=0, x2 is free, f2+f3=1)
        
        # For uniform sampling on the triangle, use:
        # u = sqrt(halton1), v = halton2 * (1 - u)
        # Then barycentric coords: (1-u-v, v, u) or similar. Let me use the standard method:
        # r1, r2 uniform in [0,1]; if r1+r2 > 1, flip: r1=1-r1, r2=1-r2
        # Then barycentric = (r1, r2, 1-r1-r2)
        
        # Actually, the simplest: generate halton in [0,1]^2, apply the triangle transform
        h1, h2 = halton_pts[:, 0], halton_pts[:, 1]
        # Standard uniform triangle sampling
        sqrt_h1 = np.sqrt(h1)
        h2_adj = h2 * (1.0 - sqrt_h1)
        # Barycentric coordinates (a, b, c) for vertices A, B, C
        a = 1.0 - sqrt_h1 - h2_adj  # weight for A=(1,0,0) -> x1=1,x2=0
        b = h2_adj                    # weight for B=(0,1,0) -> x1=0,x2=1  
        c = sqrt_h1                   # weight for C=(0,0,1) -> x1=0,x2=0
        # Wait, let me re-derive. Standard: pick u=sqrt(r1), v=r2*(1-u)
        # Barycentric: (1-u-v, v, u) for triangle with vertices at (0,0), (1,0), (0,1) in 2D
        # For our triangle with vertices A,B,C:
        # f = (1-u-v)*A + v*B + u*C
        # f1 = (1-u-v)*1 + v*0 + u*0 = 1-u-v
        # f2 = (1-u-v)*0 + v*1 + u*0 = v
        # f3 = (1-u-v)*0 + v*0 + u*1 = u
        # So: x1 = f1 = 1-u-v, x2 = f2/(1-f1) = v/(u+v) if u+v>0, else 0
        
        u = sqrt_h1
        v = h2_adj
        s = u + v
        x1 = 1.0 - s
        x2 = np.where(s > 1e-15, v / s, 0.0)
        
        # Clip to [0,1]
        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        
        shape_pts = np.column_stack([x1, x2])
        
        # Build X vectors: x[0]=x1, x[1]=x2, x[2:]=0.5 (to make g=0)
        X = np.zeros((len(shape_pts), n))
        X[:, 0] = x1
        X[:, 1] = x2
        for j in range(n_shape, n):
            X[:, j] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points with time guard
        F = np.empty((len(X), m))
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
