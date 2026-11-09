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
    """
    cw_masks = []
    ccw_masks = []
    for s, t, w in DEMANDS:
        cw_mask = np.zeros(N_EDGES)
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
    """
    cw_weights = (1.0 - routing) * WEIGHTS
    ccw_weights = routing * WEIGHTS
    loads = CW_MASKS.T @ cw_weights + CCW_MASKS.T @ ccw_weights
    return loads


def _max_load(routing):
    """Compute the maximum edge load for a given routing."""
    loads = _compute_loads(routing)
    return np.max(loads)


def _local_search_2bit(routing, max_iters=50000):
    """
    Local search that tries flipping 1 or 2 bits at a time.
    This helps escape 1-bit local optima.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try 1-bit flips
        for i in range(N_DEMANDS):
            new_routing = routing.copy()
            new_routing[i] = 1 - new_routing[i]
            new_max = _max_load(new_routing)
            if new_max < current_max - 1e-12:
                routing = new_routing
                current_max = new_max
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-bit flips
        for i in range(N_DEMANDS):
            for j in range(i + 1, N_DEMANDS):
                new_routing = routing.copy()
                new_routing[i] = 1 - new_routing[i]
                new_routing[j] = 1 - new_routing[j]
                new_max = _max_load(new_routing)
                if new_max < current_max - 1e-12:
                    routing = new_routing
                    current_max = new_max
                    improved = True
                    break
            if improved:
                break
    
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
        T = start_temp * (end_temp / start_temp) ** (it / n_iters)
        
        # Random move: flip a random bit (or two bits occasionally)
        if random.random() < 0.2:
            i = random.randint(0, N_DEMANDS - 1)
            j = random.randint(0, N_DEMANDS - 1)
            if i == j:
                j = (j + 1) % N_DEMANDS
            new_routing = routing.copy()
            new_routing[i] = 1 - new_routing[i]
            new_routing[j] = 1 - new_routing[j]
        else:
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


def _tabu_search(routing, tabu_len=20, max_iters=30000):
    """
    Tabu search to escape local optima.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    best_routing = routing.copy()
    best_max = current_max
    
    tabu_list = {}  # demand -> iteration when it was last flipped
    current_iter = 0
    
    for it in range(max_iters):
        # Find best non-tabu move
        best_move = None
        best_move_max = current_max
        
        for i in range(N_DEMANDS):
            # Check if i is tabu
            if i in tabu_list and (current_iter - tabu_list[i]) < tabu_len:
                continue
            
            new_routing = routing.copy()
            new_routing[i] = 1 - new_routing[i]
            new_max = _max_load(new_routing)
            
            if new_max < best_move_max:
                best_move_max = new_max
                best_move = i
        
        if best_move is None:
            # All moves are tabu, pick the best one anyway
            for i in range(N_DEMANDS):
                new_routing = routing.copy()
                new_routing[i] = 1 - new_routing[i]
                new_max = _max_load(new_routing)
                if new_max < best_move_max:
                    best_move_max = new_max
                    best_move = i
        
        if best_move is None:
            break
        
        # Apply move
        routing[best_move] = 1 - routing[best_move]
        tabu_list[best_move] = current_iter
        current_max = best_move_max
        current_iter += 1
        
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
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Start with all clockwise
    best_routing = np.zeros(N_DEMANDS)
    best_max = _max_load(best_routing)
    
    rng = np.random.default_rng(42)
    
    # Method 1: 2-bit local search from all-zero
    routing, max_load = _local_search_2bit(np.zeros(N_DEMANDS))
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 2: 2-bit local search from all-one
    routing, max_load = _local_search_2bit(np.ones(N_DEMANDS))
    if max_load < best_max:
        best_max = max_load
        best_routing = routing.copy()
    
    # Method 3: Many random restarts with 2-bit local search
    for trial in range(200):
        if time.time() > deadline - 120:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        routing, max_load = _local_search_2bit(random_routing, max_iters=20000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 4: Simulated annealing from best found so far
    if time.time() < deadline - 120:
        routing, max_load = _simulated_annealing(best_routing, start_temp=5.0, end_temp=0.01, max_iters=30000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 5: Tabu search from best found so far
    if time.time() < deadline - 90:
        routing, max_load = _tabu_search(best_routing, tabu_len=20, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 6: More random restarts with SA + tabu
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        # First 2-bit local search
        hc_routing, hc_max = _local_search_2bit(random_routing, max_iters=10000)
        # Then SA from there
        sa_routing, sa_max = _simulated_annealing(hc_routing, start_temp=3.0, end_temp=0.01, max_iters=15000)
        if sa_max < best_max:
            best_max = sa_max
            best_routing = sa_routing.copy()
        
        # Then tabu from there
        if time.time() < deadline - 30:
            ts_routing, ts_max = _tabu_search(sa_routing, tabu_len=15, max_iters=20000)
            if ts_max < best_max:
                best_max = ts_max
                best_routing = ts_routing.copy()
    
    # Final local search polish
    if time.time() < deadline - 10:
        routing, max_load = _local_search_2bit(best_routing, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    return best_routing


# EVOLVE-BLOCK-END
