# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Precompute the ring structure
N_NODES = 10
N_EDGES = 10
N_DEMANDS = 15

# Demands: (source, target, weight)
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


def _build_edge_masks():
    """
    For each demand, precompute which edges are used if routed clockwise
    and which if routed counter-clockwise.
    
    Clockwise from s to t: edges s, s+1, ..., t-1 (mod N_NODES)
    Counter-clockwise: all other edges.
    
    Returns:
        cw_masks: list of np.ndarray of shape (N_EDGES,) with 1s for edges used CW
        ccw_masks: list of np.ndarray of shape (N_EDGES,) with 1s for edges used CCW
    """
    cw_masks = []
    ccw_masks = []
    for s, t, w in DEMANDS:
        cw_mask = np.zeros(N_EDGES)
        # Clockwise: edges s, s+1, ..., t-1 mod N
        if s <= t:
            for e in range(s, t):
                cw_mask[e] = 1
        else:
            for e in range(s, N_NODES):
                cw_mask[e] = 1
            for e in range(0, t):
                cw_mask[e] = 1
        
        ccw_mask = 1 - cw_mask
        cw_masks.append(cw_mask)
        ccw_masks.append(ccw_mask)
    
    return np.array(cw_masks), np.array(ccw_masks)


# Precompute masks
CW_MASKS, CCW_MASKS = _build_edge_masks()

# Weights array
WEIGHTS = np.array([w for s, t, w in DEMANDS], dtype=np.float64)


def _compute_loads(routing):
    """
    Given a routing vector (0=cw, 1=ccw) of shape (N_DEMANDS,),
    compute the load on each edge.
    
    Returns:
        loads: np.ndarray of shape (N_EDGES,)
    """
    # routing: 0 means CW, 1 means CCW
    # For each demand i, if routing[i] == 0, use CW_MASKS[i], else CCW_MASKS[i]
    # We can do this with matrix operations:
    # loads = sum_i routing_weighted[i] * mask[i]
    
    # Efficient: 
    # cw_contribution = (1 - routing) * WEIGHTS  -> weight if CW
    # ccw_contribution = routing * WEIGHTS       -> weight if CCW
    # loads = sum_i (cw_contribution[i] * CW_MASKS[i] + ccw_contribution[i] * CCW_MASKS[i])
    
    cw_weights = (1.0 - routing) * WEIGHTS  # shape (N_DEMANDS,)
    ccw_weights = routing * WEIGHTS          # shape (N_DEMANDS,)
    
    # loads[e] = sum_i cw_weights[i] * CW_MASKS[i,e] + ccw_weights[i] * CCW_MASKS[i,e]
    loads = CW_MASKS.T @ cw_weights + CCW_MASKS.T @ ccw_weights
    return loads


def _max_load(routing):
    """Compute the maximum edge load for a given routing."""
    loads = _compute_loads(routing)
    return np.max(loads)


def _hill_climb(routing, max_iters=100000):
    """
    Simple hill climbing: try flipping each bit, keep if it improves.
    Repeat until no improvement.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(N_DEMANDS):
            new_routing = routing.copy()
            new_routing[i] = 1 - new_routing[i]
            new_max = _max_load(new_routing)
            if new_max < current_max - 1e-12:
                routing = new_routing
                current_max = new_max
                improved = True
    
    return routing, current_max


def _simulated_annealing(routing, start_temp=10.0, end_temp=0.01, max_iters=50000):
    """
    Simulated annealing to escape local optima.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    best_routing = routing.copy()
    best_max = current_max
    
    n_iters = max_iters
    for it in range(n_iters):
        # Temperature schedule
        T = start_temp * (end_temp / start_temp) ** (it / n_iters)
        
        # Random move: flip a random bit
        i = random.randint(0, N_DEMANDS - 1)
        new_routing = routing.copy()
        new_routing[i] = 1 - new_routing[i]
        new_max = _max_load(new_routing)
        
        delta = new_max - current_max
        if delta < 0 or random.random() < np.exp(-delta / (T + 1e-12)):
            routing = new_routing
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_routing = routing.copy()
    
    return best_routing, best_max


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        Entry >= 0.5 routes counter-clockwise, < 0.5 routes clockwise.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    # Start with all clockwise
    best_routing = np.zeros(N_DEMANDS)
    best_max = _max_load(best_routing)
    
    # Try several initializations and optimization methods
    rng = np.random.default_rng(42)
    
    # Method 1: Hill climbing from all-zero
    routing, max_load = _hill_climb(np.zeros(N_DEMANDS))
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 2: Hill climbing from all-one
    routing, max_load = _hill_climb(np.ones(N_DEMANDS))
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 3: Hill climbing from random starts
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        routing, max_load = _hill_climb(random_routing)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 4: Simulated annealing from best found so far
    if time.time() < deadline - 120:
        routing, max_load = _simulated_annealing(best_routing, start_temp=5.0, end_temp=0.01, max_iters=20000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 5: More random restarts with SA
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        # First hill climb
        hc_routing, hc_max = _hill_climb(random_routing)
        # Then SA from there
        sa_routing, sa_max = _simulated_annealing(hc_routing, start_temp=3.0, end_temp=0.01, max_iters=10000)
        if sa_max < best_max:
            best_max = sa_max
            best_routing = sa_routing.copy()
    
    # Final local search polish
    if time.time() < deadline - 30:
        routing, max_load = _hill_climb(best_routing, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Ensure the output is valid: 0 for CW, 1 for CCW
    # The contract says: >= 0.5 routes CCW, < 0.5 routes CW
    # Our routing uses 0 for CW, 1 for CCW, which satisfies this
    return best_routing


# EVOLVE-BLOCK-END
