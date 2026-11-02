import numpy as np
from scipy.optimize import minimize

N = 13

def construct():
    # Initial guess: regular 13-gon
    theta0 = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    def score(theta):
        # Ensure angles are sorted and wrapped
        # We assume the input theta is the angles themselves
        # To maintain convexity and ordering, we can just use them as is,
        # but we must be careful with the hull area calculation if they wrap around.
        # Since we start with sorted angles and do local search, they should stay close.
        # However, to be robust, let's sort them.
        theta_sorted = np.sort(theta)
        
        # Calculate coordinates
        x = np.cos(theta_sorted)
        y = np.sin(theta_sorted)
        pts = np.column_stack((x, y))
        
        # Calculate Convex Hull Area
        # Since points are on a circle and sorted, the hull is the polygon itself.
        # Shoelace formula
        x0 = np.roll(x, -1)
        y0 = np.roll(y, -1)
        area_hull = 0.5 * np.abs(np.sum(x * y0 - x0 * y))
        
        if area_hull == 0:
            return -1.0 # Invalid
            
        # Calculate min triangle area
        min_area = np.inf
        # Iterate over all triples
        # For speed, we can vectorize or just loop. 286 loops is fine.
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    # Area of triangle with vertices pts[i], pts[j], pts[k]
                    # 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
                    ax = x[j] - x[i]
                    ay = y[j] - y[i]
                    bx = x[k] - x[i]
                    by = y[k] - y[i]
                    area = 0.5 * np.abs(ax * by - ay * bx)
                    if area < min_area:
                        min_area = area
        
        return -min_area / area_hull # We minimize negative score

    # Optimize
    res = minimize(score, theta0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Extract best angles
    theta_opt = res.x
    # Sort and map to coordinates
    theta_sorted = np.sort(theta_opt)
    x = np.cos(theta_sorted)
    y = np.sin(theta_sorted)
    pts = np.column_stack((x, y))
    
    return pts
