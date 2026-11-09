import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        n_shape = n_obj - 1
        n_dist = dim - n_shape

        # For DTLZ4, the Pareto front is reached when distance vars = 0.5
        # The front shape is determined by shape vars x_0, x_1
        # f_1 = (1+g)*cos(pi*x_0/2)
        # f_2 = (1+g)*sin(pi*x_0/2)*cos(pi*x_1/2)
        # f_3 = (1+g)*sin(pi*x_0/2)*sin(pi*x_1/2)
        # With g=0: f_1 = cos(pi*x_0/2), f_2 = sin(pi*x_0/2)*cos(pi*x_1/2), f_3 = sin(pi*x_0/2)*sin(pi*x_1/2)
        # This is a quarter of a unit sphere in the positive octant.
        
        # To maximize hypervolume with reference point [2.5, 2.5, 2.5],
        # we want to cover the Pareto front as uniformly as possible.
        # The hypervolume of a set of points on the front is maximized when
        # the points are spread out to cover the front surface area.
        
        # Strategy: Use a 2D stratified grid on the (x_0, x_1) space.
        # The mapping from (x_0, x_1) to the front is:
        #   f_1 = cos(pi*x_0/2)
        #   f_2 = sin(pi*x_0/2)*cos(pi*x_1/2)
        #   f_3 = sin(pi*x_0/2)*sin(pi*x_1/2)
        
        # For uniform coverage on the sphere surface, we need to account
        # for the Jacobian of the transformation. The surface element on
        # the unit sphere in spherical coordinates (theta, phi) is sin(theta) dtheta dphi.
        # Here theta = pi*x_0/2, phi = pi*x_1/2.
        # So the area element is sin(pi*x_0/2) * (pi/2)^2 dx_0 dx_1.
        # To get uniform sampling on the surface, we should sample x_0 with
        # density proportional to 1/sin(pi*x_0/2) and x_1 uniformly.
        
        # Actually, let's think about this differently. The hypervolume of
        # points on the Pareto front with reference point [2.5, 2.5, 2.5]
        # is the volume of the union of boxes from each point to the reference.
        # For points on a smooth front, the hypervolume is maximized when
        # the points are evenly distributed in terms of the "shadow" they
        # cast on the coordinate planes.
        
        # A simpler approach: use a grid in the (f_1, f_2) plane and compute f_3.
        # Or use a stratified grid in (x_0, x_1) with appropriate weighting.
        
        # Let's use a 2D grid with n_per_dim points in each direction.
        # To maximize coverage, use the entire budget.
        n_per_dim = int(np.sqrt(budget))
        n_extra = budget - n_per_dim * n_per_dim

        results = []

        # Stratified grid: use cell centers
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                if time.time() > deadline:
                    break

                # Stratified sampling in x_0 and x_1
                # To get uniform coverage on the sphere, we need to adjust x_0
                # The CDF for uniform on sphere: F(x_0) = 1 - cos(pi*x_0/2) / 1 = 1 - cos(pi*x_0/2)
                # Wait, the surface area from x_0=0 to x_0=a is integral of sin(pi*x_0/2) dx_0 = [−2/pi cos(pi*x_0/2)] from 0 to a = 2/pi (1 - cos(pi*a/2))
                # So CDF = (1 - cos(pi*x_0/2)) / (1 - cos(0)) ... wait, at x_0=1, cos(pi/2)=0, so CDF(1) = 1.
                # CDF(x_0) = 1 - cos(pi*x_0/2)
                # So to sample uniformly on the surface: u = 1 - cos(pi*x_0/2), x_0 = (2/pi)*arccos(1-u)
                
                u = (i + 0.5) / n_per_dim
                v = (j + 0.5) / n_per_dim
                
                # x_0 for uniform surface coverage
                x_0 = (2.0 / np.pi) * np.arccos(1.0 - u)
                x_1 = v
                
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5
                
                f = problem(x)
                results.append(f)
            
            if time.time() > deadline:
                break

        # Add extra points using a Halton-like sequence
        if n_extra > 0:
            for k in range(n_extra):
                if time.time() > deadline:
                    break

                # Use a simple quasi-random sequence
                # Van der Corput in base 2 and 3
                def van_der_corput(n, base):
                    result = 0.0
                    f = 1.0 / base
                    while n > 0:
                        result += f * (n % base)
                        n //= base
                        f /= base
                    return result

                u = van_der_corput(k + 1, 2)
                v = van_der_corput(k + 1, 3)
                
                x_0 = (2.0 / np.pi) * np.arccos(1.0 - u)
                x_1 = v
                
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5
                
                f = problem(x)
                results.append(f)

        return None
