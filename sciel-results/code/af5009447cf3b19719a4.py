"""EV routing via precomputed shortest paths and cluster-based route construction."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def plan(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    n = instance["n_customers"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    depot = 0

    # Precompute all pairwise distances
    n_total = len(coords)
    dist = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            dist[i][j] = _dist(coords[i], coords[j])

    remaining = list(range(1, n + 1))
    # Sort by due date for priority
    remaining.sort(key=lambda v: (due[v], v))

    routes = []

    def check_feasible_sequence(seq):
        """Check if a sequence of customers is feasible from depot back to depot.
        Returns (total_distance, final_time) if feasible, else None."""
        cur = depot
        time = 0.0
        batt = float(battery)
        load = 0
        total_d = 0.0
        
        for v in seq:
            d = dist[cur][v]
            t_arr = time + d / speed
            b_arr = batt - d
            if b_arr < -1e-9:
                return None
            t = max(t_arr, ready[v])
            if t > due[v] + 1e-9:
                return None
            t_serv = t + service[v]
            load += demands[v]
            if load > capacity + 1e-9:
                return None
            time = t_serv
            batt = b_arr
            total_d += d
            cur = v
        
        # Return to depot
        d_ret = dist[cur][depot]
        b_ret = batt - d_ret
        if b_ret < -1e-9:
            return None
        t_ret = time + d_ret / speed
        if t_ret > horizon + 1e-9:
            return None
        total_d += d_ret
        return (total_d, t_ret)

    def try_insert(customer, seq):
        """Try to insert customer into seq at best position. Returns new_seq or None."""
        best_pos = -1
        best_cost = float('inf')
        
        for i in range(len(seq) + 1):
            new_seq = seq[:i] + [customer] + seq[i:]
            result = check_feasible_sequence(new_seq)
            if result is not None:
                # Cost: total distance
                if result[0] < best_cost:
                    best_cost = result[0]
                    best_pos = i
        
        if best_pos >= 0:
            return seq[:best_pos] + [customer] + seq[best_pos:]
        return None

    # Build routes using insertion heuristic
    while remaining:
        route_nodes = []
        
        # Start with the customer with earliest due date
        start = remaining[0]
        route_nodes.append(start)
        remaining.remove(start)
        
        # Try to insert other customers
        changed = True
        while changed:
            changed = False
            for v in list(remaining):
                new_route = try_insert(v, route_nodes)
                if new_route is not None:
                    route_nodes = new_route
                    remaining.remove(v)
                    changed = True
        
        if route_nodes:
            routes.append(route_nodes)

    return routes
# EVOLVE-BLOCK-END
