"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct 200 circles in a unit square [-0.5, 0.5]^2.
    Uses a hexagonal lattice approach, selecting the 200 points closest to the center
    from a lattice that fits within the square.
    """
    
    def hex_lattice_points(n_x, n_y, s, rot_deg=0):
        """
        Generate points on a hexagonal lattice.
        n_x: number of points in x-direction
        n_y: number of points in y-direction
        s: spacing (distance between nearest neighbors)
        rot_deg: rotation of the lattice
        """
        # Generate grid indices
        # Standard hex lattice:
        # x = i * s + (j % 2) * s / 2
        # y = j * s * sqrt(3) / 2
        
        i_vals = np.arange(n_x)
        j_vals = np.arange(n_y)
        
        # Create meshgrid
        I, J = np.meshgrid(i_vals, j_vals)
        
        # Calculate coordinates
        x = I * s + (J % 2) * (s / 2.0)
        y = J * s * (np.sqrt(3) / 2.0)
        
        # Center the lattice
        # Find the center of the bounding box of the generated points
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        
        x = x - cx
        y = y - cy
        
        # Rotate
        theta = np.radians(rot_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        
        points = np.column_stack((x_rot, y_rot))
        
        # Filter points that are strictly inside the square [-0.5, 0.5]
        # We need a small margin to ensure they are "inside" and not on the edge if floating point issues arise,
        # but the problem says "on or outside" is invalid. So we must be strictly inside?
        # "A centre on or outside the square ... makes the candidate invalid."
        # So we must ensure |x| < 0.5 and |y| < 0.5.
        # However, the score formula uses min(1/2 - |c_ik|). If c_ik = 0.5, dist = 0, radius = 0.
        # So being on the edge gives radius 0. We need to be strictly inside to get positive radius.
        # The filter should be |x| < 0.5 and |y| < 0.5.
        
        mask = (np.abs(x_rot) < 0.5) & (np.abs(y_rot) < 0.5)
        points = points[mask]
        
        return points

    best_points = None
    best_score = 0.0

    # We want to approximate 200 points.
    # In a hex lattice, the number of points in an n x m grid is n*m.
    # Let's try dimensions around sqrt(200) ~ 14.
    # Possible (n_x, n_y) pairs: (14, 14), (14, 15), (15, 14), (13, 15), (15, 13), (14, 16), (16, 14)
    
    dims = [(14, 14), (14, 15), (15, 14), (13, 15), (15, 13), (14, 16), (16, 14), (13, 16), (16, 13)]
    
    # Rotations to try
    rotations = [0, 15, 30, 45]
    
    # Spacing multipliers to fine-tune fit
    # The spacing s determines the size of the lattice.
    # We want the lattice to be as large as possible while keeping points inside.
    # Instead of guessing s, we can calculate the max s for a given grid shape.
    # However, rotation makes it complex.
    # Easier: Try a range of s values.
    
    # Estimate s from area:
    # Area per point in hex lattice = s * s * sqrt(3)/2
    # Total area ~ N * s^2 * sqrt(3)/2 <= 1
    # s <= sqrt(2 / (N * sqrt(3)))
    # For N=200: s <= sqrt(2 / (200 * 1.732)) = sqrt(2 / 346.4) = sqrt(0.00577) = 0.0759
    # So s is around 0.07 - 0.08.
    
    s_values = np.linspace(0.06, 0.09, 30)

    for n_x, n_y in dims:
        for rot in rotations:
            for s in s_values:
                points = hex_lattice_points(n_x, n_y, s, rot)
                
                if len(points) < N:
                    continue
                
                # Select N points closest to the center (0,0)
                dists = np.linalg.norm(points, axis=1)
                indices = np.argsort(dists)
                selected = points[indices[:N]]
                
                # Calculate score
                # 1. Min distance to boundary
                # min_i min_k (1/2 - |c_ik|)
                # This is min(0.5 - max(|x|), 0.5 - max(|y|))
                max_x = np.max(np.abs(selected[:, 0]))
                max_y = np.max(np.abs(selected[:, 1]))
                dist_to_boundary = min(0.5 - max_x, 0.5 - max_y)
                
                if dist_to_boundary <= 0:
                    continue

                # 2. Min distance between pairs
                # We can use scipy.spatial.distance or a simple loop if N is small, but N=200 is 19900 pairs.
                # Using scipy is faster.
                from scipy.spatial.distance import pdist
                dists_pairs = pdist(selected)
                min_dist_pairs = np.min(dists_pairs)
                
                radius = min(dist_to_boundary, min_dist_pairs / 2.0)
                
                if radius > best_score:
                    best_score = radius
                    best_points = selected

    if best_points is None:
        # Fallback to square grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_points

# EVOLVE-BLOCK-END
