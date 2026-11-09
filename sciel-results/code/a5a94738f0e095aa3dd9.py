# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute clockwise and counter-clockwise edge sets for each demand
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        # Counter-clockwise: all edges not in clockwise
        all_edges = set(range(n))
        ccw = list(all_edges - set(cw))
        ccw_edges.append(ccw)

    def compute_loads(routing):
        """Compute edge loads given a routing vector (0=cw, 1=ccw)."""
        loads = np.zeros(n)
        for i in range(D):
            s, t, w = demands[i]
            if routing[i] < 0.5:  # clockwise
                for e in cw_edges[i]:
                    loads[e] += w
            else:  # counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += w
        return loads

    def max_load(routing):
        return compute_loads(routing).max()

    # Greedy initialization: start with all clockwise, then greedily flip if it helps
    routing = np.zeros(D)
    current_loads = compute_loads(routing)
    current_max = current_loads.max()

    # Greedy improvement: repeatedly flip the demand that most reduces max load
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_i = -1
        for i in range(D):
            # Try flipping demand i
            old_dir = routing[i]
            routing[i] = 1.0 - old_dir
            new_loads = compute_loads(routing)
            new_max = new_loads.max()
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_i = i
            # Restore
            routing[i] = old_dir
        if best_i >= 0 and best_gain > 0:
            routing[best_i] = 1.0 - routing[best_i]
            current_loads = compute_loads(routing)
            current_max = current_loads.max()
            improved = True

    # Simulated annealing to escape local optima
    rng = np.random.RandomState(42)
    best_routing = routing.copy()
    best_max = current_max
    current_max = best_max
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    iterations = 50000

    for it in range(iterations):
        T = max(T * cooling, T_min)
        # Pick a random demand to flip
        i = rng.randint(D)
        old_dir = routing[i]
        routing[i] = 1.0 - old_dir
        new_max = max_load(routing)
        delta = new_max - current_max

        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_routing = routing.copy()
        else:
            routing[i] = old_dir  # reject

    return best_routing


# EVOLVE-BLOCK-END
