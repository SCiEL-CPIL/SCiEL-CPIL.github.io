import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field operations
    # alpha^2 = 3
    k = 3
    
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul_f(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def div_f(a, b):
        inv_denom = (b[0]*b[0] - b[1]*b[1]*k) % p
        if inv_denom == 0:
            return None
        inv_denom_inv = pow(inv_denom, p-2, p)
        inv_b = ((b[0] * inv_denom_inv) % p, (-b[1] * inv_denom_inv) % p)
        return mul_f(a, inv_b)

    # Precompute all points
    # Point is ((xa, xb), (ya, yb))
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    n_points = len(points)
    
    # Map point to index
    point_to_idx = {}
    for i, pt in enumerate(points):
        point_to_idx[pt] = i
        
    # Choose direction m = 0, i.e., vector (1, 0)
    # Line through (x, y) with direction (1, 0) is {(x+t, y) | t in F_q}
    # This line is characterized by the constant y.
    # So we group points by their y-coordinate.
    
    # y is (ya, yb)
    # There are 49 distinct y values.
    
    S_indices = set()
    
    # Group points by y
    # y_to_points = { y_val: [indices] }
    y_to_points = {}
    for i, pt in enumerate(points):
        y_val = pt[1]
        if y_val not in y_to_points:
            y_to_points[y_val] = []
        y_to_points[y_val].append(i)
        
    # For each line (group), pick one point to be in S
    # To follow the "greedy" spirit, we can pick the point with the "highest safety margin".
    # In this specific graph (disjoint cliques), all points have the same degree.
    # So we can just pick the first one, or the one with smallest index.
    # Let's pick the one with smallest index for determinism.
    
    for y_val, indices in y_to_points.items():
        # Sort indices to pick the smallest one consistently
        # Or just pick indices[0]
        S_indices.add(indices[0])
        
    # Construct N: all points not in S
    N_indices = [i for i in range(n_points) if i not in S_indices]
    
    # Extract points
    N = [points[i] for i in N_indices]
    
    # Convert to array
    # Shape (k, 2, 2)
    # Each point is ((xa, xb), (ya, yb))
    # We need to flatten to [xa, xb, ya, yb] or keep nested?
    # Task says: "returning an integer array of shape (k, 2, 2)"
    # "A point ... is written [[x_a, x_b], [y_a, y_b]]"
    # So the array should be of shape (k, 2, 2).
    
    arr = np.zeros((len(N), 2, 2), dtype=np.int64)
    for i, pt in enumerate(N):
        arr[i, 0, 0] = pt[0][0]
        arr[i, 0, 1] = pt[0][1]
        arr[i, 1, 0] = pt[1][0]
        arr[i, 1, 1] = pt[1][1]
        
    return arr
