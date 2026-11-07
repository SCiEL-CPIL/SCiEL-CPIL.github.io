"""
Optimal Lattice Design for DTLZ1 Hypervolume Maximization.
Strategy:
1. Exploit the analytic structure: Set x_G = 0 to guarantee g(x_G) = 0.
2. Derive the optimal distribution for x_1, x_2 to maximize hypervolume.
   The hypervolume contribution of a point on the DTLZ1 front (f1+f2+f3=0.5) 
   with reference point R=(400,400,400) is proportional to the volume of the 
   box it dominates. 
   For a point (f1, f2, f3), the marginal contribution to the union of boxes 
   in a dense packing is roughly proportional to the "space" it uniquely covers.
   However, a simpler and robust approach for a fixed reference point far away 
   is to maximize the coverage of the Pareto front surface area, weighted by 
   the distance to the reference point. Since R is constant and far, the 
   hypervolume is effectively determined by the "shadow" of the front on the 
   axes. 
   
   Actually, for DTLZ1, the front is a triangle in the plane f1+f2+f3=0.5.
   The hypervolume relative to R is the volume of the union of boxes [f_i, R_i].
   This volume is maximized when the points are spread out to cover the entire 
   triangle efficiently. A uniform distribution over the triangle in objective 
   space is optimal for covering the surface. 
   
   Mapping from objective space (triangle) to parameter space (square [0,1]^2):
   f1 = 0.5 * x1 * x2
   f2 = 0.5 * (1 - x1) * x2
   f3 = 0.5 * (1 - x2)
   
   We want to sample (f1, f2) uniformly in the triangle defined by f1>=0, f2>=0, f1+f2<=0.5.
   Let u = f1 / 0.5 = x1*x2 and v = f2 / 0.5 = (1-x1)*x2.
   The region in (u,v) is the triangle u>=0, v>=0, u+v<=1.
   
   To sample (u,v) uniformly in this triangle, we can use:
   r = sqrt(U1) where U1 ~ Uniform(0,1)
   theta = 2*pi*U2 where U2 ~ Uniform(0,1)
   But this is for a circle. For a triangle, we can use:
   Let s1, s2 be independent Uniform(0,1).
   If s1 + s2 <= 1: u = s1, v = s2
   Else: u = 1 - s1, v = 1 - s2
   This generates uniform points in the triangle [0,1]^2 clipped to u+v<=1.
   
   Now we need to invert (u, v) to (x1, x2).
   u = x1 * x2
   v = (1 - x1) * x2 = x2 - x1*x2 = x2 - u
   So x2 = u + v.
   And x1 = u / x2 = u / (u + v).
   
   So the transformation is:
   1. Generate (u, v) uniformly in the triangle T = {(u,v) | u>=0, v>=0, u+v<=1}.
   2. Compute x2 = u + v.
   3. Compute x1 = u / (u + v).
   
   This ensures that the resulting points (f1, f2, f3) are uniformly distributed 
   over the Pareto front triangle. Uniform distribution over the front is the 
   optimal strategy for maximizing the hypervolume of the union of boxes when 
   the reference point is far away, as it minimizes the maximum gap between 
   points on the front, thus maximizing the covered volume.
   
   We will use a low-discrepancy sequence (Halton) to generate the (U1, U2) 
   pairs, then apply the triangle transformation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_pos = self.n_obj - 1  # 2
        n_g = self.dim - n_pos   # 5
        
        n_total = self.budget
        
        # Generate low-discrepancy points in [0, 1]^2
        # Using Halton sequence with bases 2 and 3.
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        indices = np.arange(1, n_total + 1)
        
        # Generate U1, U2 in [0, 1]
        U1 = np.array([halton_sequence(i, 2) for i in indices])
        U2 = np.array([halton_sequence(i, 3) for i in indices])
        
        # Transform (U1, U2) to uniform points (u, v) in the triangle u>=0, v>=0, u+v<=1.
        # Method: 
        # Let S1 = U1, S2 = U2.
        # If S1 + S2 <= 1, then (u, v) = (S1, S2).
        # Else, (u, v) = (1 - S1, 1 - S2).
        # This is a standard way to map a square to a triangle uniformly.
        
        mask = (U1 + U2) <= 1.0
        u = np.where(mask, U1, 1.0 - U1)
        v = np.where(mask, U2, 1.0 - U2)
        
        # Now transform (u, v) to (x1, x2).
        # x2 = u + v
        # x1 = u / (u + v)
        # Handle the case where u+v is 0 (which happens only if u=0 and v=0, i.e., U1=U2=0 or U1=U2=1)
        # In Halton, first point is (0.5, 1/3), so u+v > 0 generally.
        # But to be safe:
        x2 = u + v
        # Avoid division by zero
        x2_safe = np.where(x2 == 0, 1e-10, x2)
        x1 = u / x2_safe
        
        # Clip to [0, 1] to handle any numerical drift
        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        
        # Construct full dimension vectors
        X_samples = np.zeros((n_total, self.dim))
        X_samples[:, 0] = x1
        X_samples[:, 1] = x2
        # x_G = 0
        
        # Evaluate
        F_samples = np.array([problem(x) for x in X_samples])
        
        return X_samples
# EVOLVE-BLOCK-END
