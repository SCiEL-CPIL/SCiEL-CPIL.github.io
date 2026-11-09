# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Problem definition
N = 10  # number of nodes / edges
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

# Precompute edge incidence for clockwise routing
# Clockwise from s to t uses edges s, s+1, ..., t-1 (mod N)
# Counter-clockwise uses all other edges.
# For each demand, precompute which edges are used in CW and CCW routing.

# edge_mask_cw[i] is a bitset or boolean array of length N
edge_mask_cw = []
edge_mask_ccw = []
weights = []

for (s, t, w) in DEMANDS:
    cw_edges = set()
    ccw_edges = set()
    # Clockwise: s, s+1, ..., t-1 mod N
    e = s
    while True:
        cw_edges.add(e)
        if e == t - 1:
            break
        e = (e + 1) % N
    # Counter-clockwise: all other edges
    all_edges = set(range(N))
    ccw_edges = all_edges - cw_edges
    
    edge_mask_cw.append(cw_edges)
    edge_mask_ccw.append(ccw_edges)
    weights.append(w)


def compute_max_load(routing):
    """
    routing: array of length D, each entry 0 (CW) or 1 (CCW)
    Returns the maximum edge load.
    """
    loads = np.zeros(N)
    for i in range(D):
        if routing[i] < 0.5:  # CW
            for e in edge_mask_cw[i]:
                loads[e] += weights[i]
        else:  # CCW
            for e in edge_mask_ccw[i]:
                loads[e] += weights[i]
    return np.max(loads)


def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem to minimize maximum edge load.
    Returns array of shape (15,) where entry >= 0.5 means CCW, else CW.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds wall-clock limit
    
    # Start with all-CW
    best_routing = np.zeros(D)
    best_load = compute_max_load(best_routing)
    
    # Try all-CCW
    alt = np.ones(D)
    load = compute_max_load(alt)
    if load < best_load:
        best_load = load
        best_routing = alt.copy()
    
    # Local search: hill climbing with random restarts
    # For each step, try flipping one demand's direction
    # If it improves, accept; otherwise, try a random flip (simulated annealing style)
    
    # Use a simple approach: iterate through all single flips and take the best improvement
    # Repeat until no improvement or time limit
    
    # Precompute for speed: for each demand i, the delta in loads if we flip from CW to CCW or vice versa
    # Actually, let's just do direct evaluation since D=15, N=10 is small
    
    def evaluate(routing):
        loads = np.zeros(N)
        for i in range(D):
            w = weights[i]
            if routing[i] < 0.5:
                for e in edge_mask_cw[i]:
                    loads[e] += w
            else:
                for e in edge_mask_ccw[i]:
                    loads[e] += w
        return np.max(loads)
    
    current = best_routing.copy()
    current_load = best_load
    
    # Hill climbing
    improved = True
    while improved:
        improved = False
        best_flip = -1
        best_new_load = current_load
        for i in range(D):
            trial = current.copy()
            trial[i] = 1.0 - trial[i]
            trial_load = evaluate(trial)
            if trial_load < best_new_load:
                best_new_load = trial_load
                best_flip = i
        if best_flip >= 0:
            current[best_flip] = 1.0 - current[best_flip]
            current_load = best_new_load
            if current_load < best_load:
                best_load = current_load
                best_routing = current.copy()
            improved = True
    
    # Try random restarts with local search
    rng = np.random.default_rng(42)
    num_restarts = 0
    while time.time() < deadline - 10:
        # Random initial solution
        current = rng.random(D)
        # Local search from here
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_new_load = evaluate(current)
            for i in range(D):
                trial = current.copy()
                trial[i] = 1.0 - trial[i]
                trial_load = evaluate(trial)
                if trial_load < best_new_load:
                    best_new_load = trial_load
                    best_flip = i
            if best_flip >= 0:
                current[best_flip] = 1.0 - current[best_flip]
                if best_new_load < best_load:
                    best_load = best_new_load
                    best_routing = current.copy()
                improved = True
        num_restarts += 1
    
    return best_routing


# EVOLVE-BLOCK-END
