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


def _soft_max_load(routing, beta=100.0):
    """
    Smooth surrogate for max load using log-sum-exp.
    As beta -> inf, this approaches the true max.
    """
    loads = _compute_loads(routing)
    m = np.max(loads)
    return m + np.log(np.sum(np.exp(beta * (loads - m)))) / beta


def _hill_climb(routing, max_iters=100000):
    """
    Hill climbing with 1-bit and 2-bit flips.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # 1-bit flips
        for i in range(N_DEMANDS):
            new_routing = routing.copy()
            new_routing[i] = 1 - new_routing[i]
            new_max = _max_load(new_routing)
            if new_max < current_max - 1e-12:
                routing = new_routing
                current_max = new_max
                improved = True
        # 2-bit flips
        if not improved:
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
        # 3-bit flips (for deeper escape)
        if not improved:
            for i in range(N_DEMANDS):
                for j in range(i + 1, N_DEMANDS):
                    for k in range(j + 1, N_DEMANDS):
                        new_routing = routing.copy()
                        new_routing[i] = 1 - new_routing[i]
                        new_routing[j] = 1 - new_routing[j]
                        new_routing[k] = 1 - new_routing[k]
                        new_max = _max_load(new_routing)
                        if new_max < current_max - 1e-12:
                            routing = new_routing
                            current_max = new_max
                            improved = True
                            break
                    if improved:
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


def _smooth_annealing(routing, start_temp=10.0, end_temp=0.01, max_iters=20000, beta_start=10.0, beta_end=100.0):
    """
    Annealing using the smooth surrogate (log-sum-exp) for the objective.
    This helps navigate the landscape more smoothly.
    """
    routing = routing.copy()
    current_obj = _soft_max_load(routing, beta_start)
    best_routing = routing.copy()
    best_obj = current_obj
    best_max = _max_load(routing)
    
    n_iters = max_iters
    for it in range(n_iters):
        frac = it / n_iters
        T = start_temp * (end_temp / start_temp) ** frac
        beta = beta_start * (beta_end / beta_start) ** frac
        
        i = random.randint(0, N_DEMANDS - 1)
        new_routing = routing.copy()
        new_routing[i] = 1 - new_routing[i]
        new_obj = _soft_max_load(new_routing, beta)
        
        delta = new_obj - current_obj
        if delta < 0 or random.random() < np.exp(-delta / (T + 1e-12)):
            routing = new_routing
            current_obj = new_obj
            new_max = _max_load(routing)
            if new_max < best_max:
                best_max = new_max
                best_routing = routing.copy()
    
    return best_routing, best_max


def _exact_local_search(routing):
    """
    Exhaustive local search: try all possible flips of up to 3 bits.
    Since N_DEMANDS=15, the number of 3-bit flips is C(15,3)=455, which is small.
    This guarantees finding the best within 3-bit neighborhood.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    
    improved = True
    while improved:
        improved = False
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
        if improved:
            continue
        
        # Try 3-bit flips
        for i in range(N_DEMANDS):
            for j in range(i + 1, N_DEMANDS):
                for k in range(j + 1, N_DEMANDS):
                    new_routing = routing.copy()
                    new_routing[i] = 1 - new_routing[i]
                    new_routing[j] = 1 - new_routing[j]
                    new_routing[k] = 1 - new_routing[k]
                    new_max = _max_load(new_routing)
                    if new_max < current_max - 1e-12:
                        routing = new_routing
                        current_max = new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return routing, current_max


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_routing = np.zeros(N_DEMANDS)
    best_max = _max_load(best_routing)
    
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
    
    # Method 3: Random restarts with hill climbing
    for trial in range(300):
        if time.time() > deadline - 60:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        routing, max_load = _hill_climb(random_routing)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 4: Smooth annealing from best found so far
    if time.time() < deadline - 120:
        routing, max_load = _smooth_annealing(best_routing, start_temp=5.0, end_temp=0.01, max_iters=15000, beta_start=5.0, beta_end=50.0)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 5: More random restarts with SA
    for trial in range(100):
        if time.time() > deadline - 60:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        hc_routing, hc_max = _hill_climb(random_routing)
        sa_routing, sa_max = _simulated_annealing(hc_routing, start_temp=3.0, end_temp=0.01, max_iters=8000)
        if sa_max < best_max:
            best_max = sa_max
            best_routing = sa_routing.copy()
    
    # Method 6: Exact local search polish on best solution
    if time.time() < deadline - 30:
        routing, max_load = _exact_local_search(best_routing)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Final hill climb polish
    if time.time() < deadline - 15:
        routing, max_load = _hill_climb(best_routing, max_iters=50000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    return best_routing


# EVOLVE-BLOCK-END
