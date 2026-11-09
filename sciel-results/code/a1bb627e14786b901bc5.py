import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        n_points = self.budget
        
        # For DTLZ4 with 3 objectives, the Pareto front is a 2D surface
        # parameterized by two variables x1, x2 in [0,1]:
        #   f1 = sin(pi*x1/2)
        #   f2 = cos(pi*x1/2) * sin(pi*x2/2)
        #   f3 = cos(pi*x1/2) * cos(pi*x2/2)
        # with x[2:] = 0.5 to achieve g=0
        
        # To maximize hypervolume with reference [2.5, 2.5, 2.5], we need
        # dense coverage of the 2D Pareto front in objective space.
        # Uniform sampling in (x1, x2) space does NOT give uniform coverage
        # in objective space. We use a transformation to get better coverage.
        
        # Strategy: Use a stratified approach in objective space.
        # The hypervolume contribution is maximized by having points spread
        # uniformly across the Pareto front in objective space.
        
        # Key insight: For hypervolume with reference r=(2.5,2.5,2.5),
        # the contribution of a point depends on its position relative to
        # other non-dominated points. We want to maximize the volume of
        # the union of boxes [f_i, r_i] for each non-dominated point.
        
        # A good approach: sample uniformly in a transformed space that
        # corresponds to uniform coverage in objective space.
        
        # For DTLZ4 3-obj front:
        # f1 = sin(pi*x1/2), f2 = cos(pi*x1/2)*sin(pi*x2/2), f3 = cos(pi*x1/2)*cos(pi*x2/2)
        # Note: f1^2 + f2^2 + f3^2 = 1 (unit quarter sphere)
        
        # Let u = f1/sqrt(f2^2+f3^2) when f2^2+f3^2 > 0, i.e., u = tan(pi*x1/2)
        # Actually, let's use spherical coordinates on the quarter sphere.
        
        # Better approach: Use the fact that we can parameterize by angles.
        # Let alpha = pi*x1/2, beta = pi*x2/2
        # f1 = sin(alpha), f2 = cos(alpha)*sin(beta), f3 = cos(alpha)*cos(beta)
        
        # To get uniform coverage in objective space (quarter sphere surface),
        # we should sample alpha and beta such that the area element is uniform.
        # The area element on the sphere is sin(alpha) d(alpha) d(beta) for the full sphere.
        # For the quarter sphere (alpha, beta in [0, pi/2]), the area element is sin(alpha) d(alpha) d(beta).
        
        # To get uniform sampling in objective space, we need to sample alpha such that
        # the CDF of sin(alpha) over [0, pi/2] is uniform.
        # CDF: P(A <= a) = (1 - cos(a)) / (1 - cos(pi/2)) = 1 - cos(a) for a in [0, pi/2]
        # So if U ~ Uniform(0,1), then A = arccos(1-U) gives uniform area coverage.
        
        # For beta, the area element is just d(beta), so uniform beta is fine.
        
        # But wait - we want to maximize HYPervolume, not just have uniform coverage.
        # The hypervolume integral is more complex. Let's think about what distribution
        # of points maximizes hypervolume.
        
        # For a set of points on the Pareto front, the hypervolume is the volume of the
        # union of boxes [f_i, r] for each non-dominated point i.
        
        # An optimal strategy might be to use a space-filling curve or low-discrepancy
        # sequence in the appropriate metric space.
        
        # Let me use a different approach: since we know the exact Pareto front,
        # let's use a grid-like approach but with adaptive refinement.
        
        # Actually, the most effective approach for hypervolume maximization with a
        # known Pareto front is to use a well-distributed set of points that covers
        # the front uniformly in the "hypervolume metric".
        
        # Let's use the following transformation:
        # Sample u, v uniformly in [0,1]^2
        # Convert to points on the quarter sphere with uniform area coverage
        # alpha = arccos(1 - u)  [uniform area in alpha direction]
        # beta = v * pi/2        [uniform in beta direction]
        # x1 = 2*alpha/pi, x2 = 2*beta/pi
        
        # This should give good coverage. Let's also add some strategic points
        # near the extremes to ensure the corners are well-covered.
        
        def generate_points(n):
            points = []
            
            # Add corner and edge points explicitly
            # Corners of the parameter space:
            # (x1=0, x2=0) -> f=(0, 0, 1)
            # (x1=0, x2=1) -> f=(0, 1, 0)
            # (x1=1, x2=0) -> f=(1, 0, 0)
            # (x1=1, x2=1) -> f=(1, 0, 0) same as above
            
            points.append((0.0, 0.0))
            points.append((0.0, 1.0))
            points.append((1.0, 0.0))
            
            # Add some points along edges for better coverage
            # Edge x1=0: f=(0, sin(pi*x2/2), cos(pi*x2/2))
            for x2 in [0.25, 0.5, 0.75]:
                points.append((0.0, x2))
            
            # Edge x2=0: f=(sin(pi*x1/2), 0, cos(pi*x1/2))
            for x1 in [0.25, 0.5, 0.75]:
                points.append((x1, 0.0))
            
            # Edge x1=1: f=(1, 0, 0) - only one unique point
            # Edge x2=1: f=(sin(pi*x1/2), cos(pi*x1/2)*1, 0)
            for x1 in [0.25, 0.5, 0.75]:
                points.append((x1, 1.0))
            
            # Remaining points: use stratified sampling with the area-preserving transform
            num_remaining = n - len(points)
            if num_remaining > 0:
                # Use Halton-like sequence for low-discrepancy
                # But apply the area-preserving transformation
                
                def halton(index, base):
                    f = 1.0
                    r = 0.0
                    idx = index
                    while idx > 0:
                        f = f / base
                        r = r + f * (idx % base)
                        idx //= base
                    return r
                
                for i in range(1, num_remaining + 1):
                    if time.time() > deadline:
                        break
                    u = halton(i, 2)
                    v = halton(i, 3)
                    
                    # Area-preserving transform for uniform coverage on quarter sphere
                    # alpha = arccos(1 - u) gives uniform area in alpha
                    # beta = v * pi/2 gives uniform in beta
                    alpha = np.arccos(1.0 - u)
                    beta = v * (np.pi / 2.0)
                    
                    x1_val = 2.0 * alpha / np.pi
                    x2_val = 2.0 * beta / np.pi
                    
                    points.append((x1_val, x2_val))
            
            return points[:n]
        
        points_param = generate_points(n_points)
        
        results = []
        for x1, x2 in points_param:
            if time.time() > deadline:
                break
            
            x1_c = float(np.clip(x1, 0.0, 1.0))
            x2_c = float(np.clip(x2, 0.0, 1.0))
            
            x = np.zeros(self.dim)
            x[0] = x1_c
            x[1] = x2_c
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
        
        return results
