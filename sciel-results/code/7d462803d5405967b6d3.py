import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define 10 nodes in [-2, 2]
    nodes = np.linspace(-2.0, 2.0, 10)
    
    # Create basis matrix
    # T[:, i] is the hat function for node i
    T = np.zeros((x.shape[0], 10))
    
    for i in range(10):
        # Hat function is 0 outside [nodes[i-1], nodes[i+1]]
        # Left edge: nodes[i-1] (or nodes[0] if i=0)
        # Right edge: nodes[i+1] (or nodes[9] if i=9)
        
        # We can compute this vectorized
        # For efficiency, we can just do it in a loop or vectorize the distance
        # Distance to node i
        dist = np.abs(x - nodes[i])
        
        # The hat function is 1 - dist/width
        # Width is distance to neighbors
        # For interior nodes, width = nodes[i+1] - nodes[i-1] / 2? No.
        # Standard hat:
        # 0 if x < nodes[i-1]
        # (x - nodes[i-1]) / (nodes[i] - nodes[i-1]) if nodes[i-1] <= x <= nodes[i]
        # (nodes[i+1] - x) / (nodes[i+1] - nodes[i]) if nodes[i] < x <= nodes[i+1]
        # 0 if x > nodes[i+1]
        
        # Since nodes are uniform, spacing h = 4/9
        h = 4.0 / 9.0
        
        # Vectorized calculation for uniform grid
        # left = nodes[i] - h
        # right = nodes[i] + h
        # But for edges, we only have one side.
        
        # Simpler: use np.interp? No, we need the basis matrix.
        
        # Let's just compute the value for each point x
        # For node i:
        # If i == 0:
        #   0 if x > nodes[1]
        #   (nodes[1] - x) / (nodes[1] - nodes[0]) if x <= nodes[1]  (and x >= nodes[0])
        #   0 if x < nodes[0]
        
        # This is getting complicated to vectorize per column.
        # Let's just loop over x? No, x is 200 points. Looping over 10 nodes is fine.
        
        T[:, i] = np.zeros_like(x)
        
        if i > 0:
            left_node = nodes[i-1]
            left_dist = nodes[i] - left_node
            mask_left = (x >= left_node) & (x <= nodes[i])
            T[mask_left, i] = (x[mask_left] - left_node) / left_dist
            
        if i < 9:
            right_node = nodes[i+1]
            right_dist = right_node - nodes[i]
            mask_right = (x > nodes[i]) & (x <= right_node)
            T[mask_right, i] = (right_node - x[mask_right]) / right_dist
            
        # At the node itself, it should be 1.
        # The masks above cover it.
        # mask_left includes nodes[i], value = (nodes[i]-nodes[i-1])/dist = 1.
        # mask_right excludes nodes[i] (strict >).
        # So it works.
        
    pred = T @ params
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    return pred
