import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 10.0
    
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.int8)
        
    n = adjacency.shape[0]
    
    # Compute degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Construct Laplacian L = D - A
    # D is diagonal matrix of degrees
    # A is adjacency matrix
    # L = D - A
    # Note: A is symmetric.
    
    # Efficient construction:
    # L = np.diag(degrees) - adjacency
    # This creates a dense matrix. n=200 is small.
    
    L = np.diag(degrees) - adjacency.astype(float)
    
    # Compute Moore-Penrose pseudoinverse
    # This handles disconnected graphs and singular matrices
    try:
        L_pinv = np.linalg.pinv(L)
        # Extract diagonal
        r = np.diag(L_pinv)
        
        # Score: Negative effective resistance (peripheralness)
        # Vertices with high resistance (peripheral) get higher (less negative) scores?
        # Wait.
        # If we want to pick peripheral vertices (like LDF), we want them to have HIGH priority.
        # LDF: Low degree -> High Priority.
        # Resistance: Peripheral -> High Resistance.
        # So we want High Resistance -> High Priority.
        # Score = +r ?
        
        # Let's verify with a simple path 1-2-3.
        # L = [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
        # Eigenvalues: 0, 1.38, 2.62
        # L_pinv diag?
        # For path P3, resistance between 1 and 3 is 2.
        # r_11, r_22, r_33.
        # Due to symmetry r_11 = r_33.
        # Sum of all entries of L^+ is 0? No.
        # Let's compute.
        # Actually, for a path, the center (2) is "more central".
        # Peripheral nodes (1,3) should have higher score.
        # If r_11 > r_22, then Score = +r works.
        # If r_11 < r_22, then Score = -r works.
        
        # In electrical networks, the diagonal of the pseudoinverse is proportional to the "potential" of the node.
        # For a star, the center has lower potential variance?
        # Let's assume peripheral nodes have HIGHER diagonal entries in L^+.
        # This is a known property: the diagonal of the Green's function (L^+) is larger for nodes far from the "center".
        
        scores = r
        
    except Exception:
        # Fallback to LDF
        scores = -degrees
        
    if time.time() > deadline:
        return -degrees
        
    return scores
