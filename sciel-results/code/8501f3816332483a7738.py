# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed problem data
N = 10
DEMANDS = [
    (0, 5, 3),
    (1, 7, 2),
    (2, 9, 4),
    (3, 6, 1),
    (0, 4, 5),
    (2, 8, 2),
    (5, 9, 3),
    (1, 4, 2),
    (6, 0, 4),
    (3, 8, 1),
    (7, 2, 3),
    (4, 9, 2),
    (8, 1, 5),
    (0, 6, 1),
    (5, 2, 2),
]
D = len(DEMANDS)

# Precompute route incidence vectors for clockwise routing
# For demand (s, t, w), clockwise uses edges s, s+1, ..., t-1 (mod N)
CW_ROUTES = np.zeros((D, N), dtype=np.int8)
for i, (s, t, w) in enumerate(DEMANDS):
    dist = (t - s) % N
    for k in range(dist):
        CW_ROUTES[i, (s + k) % N] = 1

# Precompute weights vector
WEIGHTS = np.array([w for _, _, w in DEMANDS], dtype=np.float64)

def _compute_max_load(routing):
    """Compute the maximum edge load for a given routing vector."""
    # routing: shape (D,) of 0s and 1s
    # loads = base_loads + routing @ Delta.T
    base_loads = CW_ROUTES.T @ WEIGHTS
    Delta = WEIGHTS[None, :] * (1 - 2 * CW_ROUTES.T)
    loads = base_loads + routing @ Delta.T
    return np.max(loads)

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    # Since 2^15 = 32768 is small, we can exhaustively search all possibilities
    # using vectorized operations to find the global optimum.
    
    # Generate all 2^D binary routing vectors
    masks = np.arange(1 << D, dtype=np.int32)
    
    # Create the routing matrix: shape (2^D, D)
    routing_matrix = np.zeros((1 << D, D), dtype=np.float64)
    for j in range(D):
        routing_matrix[:, j] = (masks >> j) & 1
        
    # Base load (all CW)
    base_loads = CW_ROUTES.T @ WEIGHTS  # shape (N,)
    
    # Delta contribution: Delta[e, i] = w_i * (1 - 2 * CW_ROUTES[i, e])
    Delta = WEIGHTS[None, :] * (1 - 2 * CW_ROUTES.T)
    
    # Total loads for all assignments: shape (2^D, N)
    loads = base_loads[None, :] + routing_matrix @ Delta.T
    
    # Find the maximum load for each assignment
    max_loads = loads.max(axis=1)
    
    # Find the assignment with the minimum maximum load
    best_idx = np.argmin(max_loads)
    best_routing = routing_matrix[best_idx].copy()
    
    # Ensure the output is a valid array of floats
    return best_routing.astype(np.float64)

# EVOLVE-BLOCK-END
