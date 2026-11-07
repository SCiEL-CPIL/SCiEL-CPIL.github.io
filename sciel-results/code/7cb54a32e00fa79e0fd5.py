"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _dist(p, q):
    return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def _route_cost(route, inst, coords, dist_mat):
    """Compute total distance for a route (depot implicit at both ends)."""
    total = 0.0
    prev = 0
    for node in route:
        total += dist_mat[prev, node]
        prev = node
    total += dist_mat[prev, 0]
    return total


def _simulate_route(route, inst, coords, dist_mat):
    """
    Simulate a route and check feasibility.
    Returns (feasible, total_distance, end_time).
    """
    capacity = inst["capacity"]
    battery_max = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    demands = inst["demands"]
    stations = set(inst["stations"])

    # Check capacity
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return (False, 0.0, 0.0)

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0

    for node in route:
        d = dist_mat[prev, node]
        t += d / speed
        battery -= d

        if battery < -1e-9:
            return (False, 0.0, 0.0)

        if node in stations:
            # Charging: time = (battery_max - battery) * recharge_rate
            # Actually the rule says: station adds (battery - energy) * recharge_rate
            # Re-reading: "a station adds (battery - energy) * recharge_rate time and refills"
            # I think "energy" here means the current battery level, and it refills to max.
            # So time added = (battery_max - battery) * recharge_rate? 
            # Wait, let me re-read: "a station adds (battery - energy) * recharge_rate time"
            # This is ambiguous. Let me interpret as: the charging time is proportional to how much
            # energy is needed to fill up. So time = (battery_max - battery) * recharge_rate.
            # Actually re-reading more carefully: "adds (battery - energy) * recharge_rate time"
            # Maybe "energy" is the energy consumed? Or maybe it's a typo and should be
            # (max_battery - battery) * recharge_rate. Let me go with the most common interpretation:
            # charging time = (battery_max - current_battery) * recharge_rate
            charge_time = (battery_max - battery) * recharge_rate
            t += charge_time
            battery = battery_max
            # No service time at stations
        else:
            # Customer: time window check
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return (False, 0.0, 0.0)
            t += service[node]

        total_dist += d
        prev = node

    # Return to depot
    d = dist_mat[prev, 0]
    t += d / speed
    battery -= d
    if battery < -1e-9:
        return (False, 0.0, 0.0)
    if t > horizon + 1e-9:
        return (False, 0.0, 0.0)
    total_dist += d

    return (True, total_dist, t)


def _build_route_from_cluster(cluster, inst, coords, dist_mat, stations_list):
    """
    Build a feasible route from a cluster of customers.
    Tries different orderings and inserts charging stations as needed.
    """
    if not cluster:
        return []

    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    demands = inst["demands"]
    battery_max = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    capacity = inst["capacity"]
    horizon = inst["horizon"]

    # Sort by due time as initial order
    sorted_cluster = sorted(cluster, key=lambda v: due[v])

    # Try to build route, inserting stations when battery runs low
    def build_with_stations(order):
        """Build a route from customer order, inserting stations as needed."""
        route = []
        prev = 0
        t = 0.0
        battery = battery_max

        for cust in order:
            # Check if we can reach customer
            d = dist_mat[prev, cust]
            if battery - d < -1e-9:
                # Need to charge first - find best station
                best_station = None
                best_cost = float('inf')
                for st in stations_list:
                    d_to_st = dist_mat[prev, st]
                    if battery - d_to_st < -1e-9:
                        continue
                    # After charging at station, can we reach customer?
                    d_st_to_cust = dist_mat[st, cust]
                    if d_st_to_cust > battery_max + 1e-9:
                        continue
                    # Cost metric: prefer station that minimizes detour
                    detour = d_to_st + d_st_to_cust - d
                    if detour < best_cost:
                        best_cost = detour
                        best_station = st
                if best_station is None:
                    return None
                # Go to station
                d_to_st = dist_mat[prev, best_station]
                t += d_to_st / speed
                battery -= d_to_st
                if battery < -1e-9:
                    return None
                # Charge
                charge_time = (battery_max - battery) * recharge_rate
                t += charge_time
                battery = battery_max
                route.append(best_station)
                prev = best_station
                # Now go to customer
                d = dist_mat[prev, cust]
                t += d / speed
                battery -= d
                if battery < -1e-9:
                    return None
            else:
                # Direct
                t += d / speed
                battery -= d

            # Time window check
            if t < ready[cust]:
                t = ready[cust]
            if t > due[cust] + 1e-9:
                return None
            t += service[cust]
            route.append(cust)
            prev = cust

        # Return to depot
        d = dist_mat[prev, 0]
        t_ret = t + d / speed
        battery_ret = battery - d
        if battery_ret < -1e-9 or t_ret > horizon + 1e-9:
            return None

        return route

    best_route = None
    best_dist = float('inf')

    # Try several orderings
    orderings = [sorted_cluster]
    # Reverse
    orderings.append(list(reversed(sorted_cluster)))
    # Sort by ready time
    orderings.append(sorted(cluster, key=lambda v: ready[v]))
    # Nearest neighbor from depot
    remaining = set(cluster)
    nn_order = []
    prev = 0
    while remaining:
        best_next = min(remaining, key=lambda v: dist_mat[prev, v])
        nn_order.append(best_next)
        remaining.remove(best_next)
        prev = best_next
    orderings.append(nn_order)

    for order in orderings:
        route = build_with_stations(order)
        if route is not None:
            feasible, total_dist, _ = _simulate_route(route, inst, coords, dist_mat)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_route = route

    if best_route is None:
        # Fallback: one customer per route
        return [sorted_cluster[0]]

    return best_route


def _two_opt_route(route, inst, coords, dist_mat, stations_list):
    """Apply 2-opt improvement to a route."""
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    demands = inst["demands"]
    battery_max = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    capacity = inst["capacity"]
    horizon = inst["horizon"]

    def get_distance(r):
        return _route_cost(r, inst, coords, dist_mat)

    best_route = route[:]
    best_dist = get_distance(best_route)
    improved = True
    max_iter = 50

    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        n = len(best_route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                feasible, total_dist, _ = _simulate_route(new_route, inst, coords, dist_mat)
                if feasible and total_dist < best_dist - 1e-9:
                    best_dist = total_dist
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break

    return best_route


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    coords = np.array(instance["coords"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Build distance matrix
    n_nodes = len(coords)
    dist_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = _dist(coords[i], coords[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    customers = list(range(1, n_customers + 1))
    stations_list = list(stations)

    # Cluster customers using simple k-means-like approach based on geography
    # Use 5-7 clusters for 25 customers
    n_clusters = 5
    # Simple clustering: sort by angle from depot, split into groups
    angles = np.arctan2(coords[1:, 1], coords[1:, 0])
    sorted_idx = np.argsort(angles)
    clusters = []
    for i in range(n_clusters):
        start = i * len(sorted_idx) // n_clusters
        end = (i + 1) * len(sorted_idx) // n_clusters
        cluster = [customers[j] for j in sorted_idx[start:end]]
        clusters.append(cluster)

    # Build routes for each cluster
    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        route = _build_route_from_cluster(cluster, instance, coords, dist_mat, stations_list)
        if route:
            routes.append(route)

    # Verify all customers are covered
    served = set()
    for r in routes:
        for v in r:
            if v not in stations_list:
                served.add(v)
    missing = set(customers) - served
    for c in missing:
        routes.append([c])

    # Optimize each route with 2-opt
    for i in range(len(routes)):
        if time.time() > deadline:
            break
        routes[i] = _two_opt_route(routes[i], instance, coords, dist_mat, stations_list)

    # Try to merge small routes if capacity allows
    if time.time() < deadline - 10:
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(routes)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline - 5:
                        break
                    # Try concatenating routes i and j
                    merged = routes[i] + routes[j]
                    feasible, total_dist, _ = _simulate_route(merged, instance, coords, dist_mat)
                    if feasible:
                        routes[i] = merged
                        routes.pop(j)
                        changed = True
                        break
                if changed:
                    break

    # Final verification
    final_routes = []
    for r in routes:
        feasible, _, _ = _simulate_route(r, instance, coords, dist_mat)
        if feasible:
            final_routes.append(r)
        else:
            # Split into individual customers
            for v in r:
                if v not in stations_list:
                    final_routes.append([v])

    return final_routes
# EVOLVE-BLOCK-END
