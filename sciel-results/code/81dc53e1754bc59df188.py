# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def plan(instance):
    start_time = time.time()
    DEADLINE = 145.0

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)

    # Vectorized distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    customers = list(range(1, n_customers + 1))
    station_list = list(stations)

    def is_feasible(route):
        """Check if a route is feasible. Returns (valid, total_distance)."""
        if not route:
            return True, 0.0

        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_dist = 0.0
        total_demand = 0.0

        for node in route:
            d = dist[curr_node][node]
            total_dist += d

            if curr_battery < d - 1e-9:
                return False, 0.0

            curr_battery -= d
            curr_time += d / speed

            if node in stations:
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
            else:
                total_demand += demands[node]
                if total_demand > capacity + 1e-9:
                    return False, 0.0

                service_start = max(curr_time, ready[node])
                if service_start > due[node] + 1e-9:
                    return False, 0.0

                curr_time = service_start + service[node]

            curr_node = node

        d = dist[curr_node][0]
        total_dist += d

        if curr_battery < d - 1e-9:
            return False, 0.0

        curr_time += d / speed

        if curr_time > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        curr = 0
        total = 0.0
        for node in route:
            total += dist[curr][node]
            curr = node
        total += dist[curr][0]
        return total

    def construct_route(ordered_customers):
        """Construct a feasible route from ordered customers, inserting stations as needed."""
        route = []
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_demand = 0.0

        for c in ordered_customers:
            # Check if we can reach customer directly
            d_direct = dist[curr_node][c]
            if curr_battery >= d_direct - 1e-9:
                # Try going directly
                new_time = curr_time + d_direct / speed
                service_start = max(new_time, ready[c])
                if service_start <= due[c] + 1e-9:
                    route.append(c)
                    curr_battery -= d_direct
                    curr_time = service_start + service[c]
                    total_demand += demands[c]
                    curr_node = c
                    continue

            # Need to charge - find best station
            best_station = None
            best_total_dist = float('inf')
            for s in station_list:
                d1 = dist[curr_node][s]
                if curr_battery < d1 - 1e-9:
                    continue
                charge_time = (battery_max - (curr_battery - d1)) * recharge_rate
                new_time = curr_time + d1 / speed + charge_time
                d2 = dist[s][c]
                new_time2 = new_time + d2 / speed
                service_start = max(new_time2, ready[c])
                if service_start <= due[c] + 1e-9:
                    total_d = d1 + d2
                    if total_d < best_total_dist:
                        best_total_dist = total_d
                        best_station = s

            if best_station is None:
                # Cannot reach customer, return what we have
                return route

            # Go to station then to customer
            s = best_station
            d1 = dist[curr_node][s]
            curr_battery -= d1
            curr_time += d1 / speed
            charge_time = (battery_max - curr_battery) * recharge_rate
            curr_time += charge_time
            curr_battery = battery_max
            route.append(s)
            curr_node = s

            d2 = dist[curr_node][c]
            curr_battery -= d2
            curr_time += d2 / speed
            service_start = max(curr_time, ready[c])
            curr_time = service_start + service[c]
            total_demand += demands[c]
            curr_node = c

        return route

    def two_opt(route):
        """Improve route using 2-opt with fast feasibility check."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return route

        improved = True
        while improved:
            improved = False
            n = len(best_route)
            for i in range(n - 1):
                if time.time() - start_time > DEADLINE:
                    return best_route
                for j in range(i + 1, n):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    valid, new_dist = is_feasible(new_route)
                    if valid and new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def or_opt(route):
        """Improve route using OR-opt (move 1-2 nodes)."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return route

        improved = True
        while improved:
            improved = False
            n = len(best_route)
            for seg_len in [1, 2]:
                for i in range(n - seg_len + 1):
                    if time.time() - start_time > DEADLINE:
                        return best_route
                    segment = best_route[i:i+seg_len]
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    for j in range(len(remaining) + 1):
                        if j == i or j == i + seg_len:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        valid, new_dist = is_feasible(new_route)
                        if valid and new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_route

    def relocate_improve(routes):
        """Try moving customers between routes."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                break
            for i in range(len(routes)):
                if improved:
                    break
                for j in range(len(routes[i])):
                    if improved:
                        break
                    c = routes[i][j]
                    if c in stations:
                        continue
                    new_route_i = routes[i][:j] + routes[i][j+1:]
                    valid_i, dist_i = is_feasible(new_route_i)
                    if not valid_i:
                        continue
                    # Try inserting into other routes
                    for k in range(len(routes)):
                        if improved:
                            break
                        if k == i:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            new_route_k = routes[k][:pos] + [c] + routes[k][pos:]
                            valid_k, dist_k = is_feasible(new_route_k)
                            if valid_k:
                                old_total = route_distance(routes[i]) + route_distance(routes[k])
                                new_total = dist_i + dist_k
                                if new_total < old_total - 1e-9:
                                    routes[i] = new_route_i
                                    routes[k] = new_route_k
                                    improved = True
                                    break
        return routes

    def merge_routes(routes):
        """Try merging two routes if feasible."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                break
            for i in range(len(routes)):
                if improved:
                    break
                for j in range(i + 1, len(routes)):
                    if improved:
                        break
                    # Try concatenating routes in both orders
                    for order in [0, 1]:
                        if improved:
                            break
                        if order == 0:
                            combined = routes[i] + routes[j]
                        else:
                            combined = routes[j] + routes[i]
                        valid, total_dist = is_feasible(combined)
                        if valid:
                            old_total = route_distance(routes[i]) + route_distance(routes[j])
                            if total_dist < old_total - 1e-9:
                                new_routes = routes[:i] + [combined] + routes[i+1:j] + routes[j+1:]
                                routes = new_routes
                                improved = True
                                break
        return routes

    def greedy_insertion(sort_key="due"):
        """Greedy insertion heuristic."""
        routes = []
        if sort_key == "due":
            sorted_customers = sorted(customers, key=lambda v: due[v])
        elif sort_key == "angle":
            sorted_customers = sorted(customers, key=lambda v: np.arctan2(coords[v][1] - coords[0][1], coords[v][0] - coords[0][0]))
        else:
            sorted_customers = customers[:]

        for c in sorted_customers:
            if time.time() - start_time > DEADLINE:
                break

            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')

            for r_idx, route in enumerate(routes):
                current_demand = sum(demands[node] for node in route if node not in stations)
                if current_demand + demands[c] > capacity + 1e-9:
                    continue

                for p in range(len(route) + 1):
                    new_route = route[:p] + [c] + route[p:]
                    valid, actual_dist = is_feasible(new_route)
                    if valid and actual_dist < best_cost:
                        best_cost = actual_dist
                        best_route_idx = r_idx
                        best_pos = p

            if best_route_idx != -1:
                routes[best_route_idx].insert(best_pos, c)
            else:
                routes.append([c])

        return routes

    def nearest_neighbor():
        """Nearest neighbor construction."""
        routes = []
        unvisited = set(customers)
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_demand = 0.0
        route = []

        while unvisited:
            if time.time() - start_time > DEADLINE:
                break

            # Find nearest feasible customer
            best_c = None
            best_dist = float('inf')
            for c in unvisited:
                if total_demand + demands[c] > capacity + 1e-9:
                    continue
                d = dist[curr_node][c]
                if curr_battery < d - 1e-9:
                    # Check if we can reach via station
                    can_reach = False
                    for s in station_list:
                        d1 = dist[curr_node][s]
                        if curr_battery < d1 - 1e-9:
                            continue
                        charge_time = (battery_max - (curr_battery - d1)) * recharge_rate
                        new_time = curr_time + d1 / speed + charge_time
                        d2 = dist[s][c]
                        new_time2 = new_time + d2 / speed
                        service_start = max(new_time2, ready[c])
                        if service_start <= due[c] + 1e-9:
                            can_reach = True
                            break
                    if not can_reach:
                        continue
                    d = d + dist[curr_node][station_list[0]]  # approximate
                new_time = curr_time + d / speed
                service_start = max(new_time, ready[c])
                if service_start > due[c] + 1e-9:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_c = c

            if best_c is None:
                # Try with station
                best_c = None
                best_total = float('inf')
                for c in unvisited:
                    if total_demand + demands[c] > capacity + 1e-9:
                        continue
                    for s in station_list:
                        d1 = dist[curr_node][s]
                        if curr_battery < d1 - 1e-9:
                            continue
                        charge_time = (battery_max - (curr_battery - d1)) * recharge_rate
                        new_time = curr_time + d1 / speed + charge_time
                        d2 = dist[s][c]
                        new_time2 = new_time + d2 / speed
                        service_start = max(new_time2, ready[c])
                        if service_start <= due[c] + 1e-9:
                            total_d = d1 + d2
                            if total_d < best_total:
                                best_total = total_d
                                best_c = c
                if best_c is None:
                    break

            # Add customer (with station if needed)
            d_direct = dist[curr_node][best_c]
            if curr_battery >= d_direct - 1e-9:
                new_time = curr_time + d_direct / speed
                service_start = max(new_time, ready[best_c])
                if service_start <= due[best_c] + 1e-9:
                    route.append(best_c)
                    curr_battery -= d_direct
                    curr_time = service_start + service[best_c]
                    total_demand += demands[best_c]
                    curr_node = best_c
                    unvisited.remove(best_c)
                    continue

            # Need station
            best_s = None
            best_total_d = float('inf')
            for s in station_list:
                d1 = dist[curr_node][s]
                if curr_battery < d1 - 1e-9:
                    continue
                charge_time = (battery_max - (curr_battery - d1)) * recharge_rate
                new_time = curr_time + d1 / speed + charge_time
                d2 = dist[s][best_c]
                new_time2 = new_time + d2 / speed
                service_start = max(new_time2, ready[best_c])
                if service_start <= due[best_c] + 1e-9:
                    total_d = d1 + d2
                    if total_d < best_total_d:
                        best_total_d = total_d
                        best_s = s

            if best_s is None:
                break

            s = best_s
            d1 = dist[curr_node][s]
            curr_battery -= d1
            curr_time += d1 / speed
            charge_time = (battery_max - curr_battery) * recharge_rate
            curr_time += charge_time
            curr_battery = battery_max
            route.append(s)
            curr_node = s

            d2 = dist[curr_node][best_c]
            curr_battery -= d2
            curr_time += d2 / speed
            service_start = max(curr_time, ready[best_c])
            curr_time = service_start + service[best_c]
            total_demand += demands[best_c]
            curr_node = best_c
            route.append(best_c)
            unvisited.remove(best_c)

        if route:
            routes.append(route)

        # Handle remaining customers with greedy
        if unvisited:
            for c in list(unvisited):
                best_route_idx = -1
                best_pos = -1
                best_cost = float('inf')
                for r_idx, route in enumerate(routes):
                    current_demand = sum(demands[node] for node in route if node not in stations)
                    if current_demand + demands[c] > capacity + 1e-9:
                        continue
                    for p in range(len(route) + 1):
                        new_route = route[:p] + [c] + route[p:]
                        valid, actual_dist = is_feasible(new_route)
                        if valid and actual_dist < best_cost:
                            best_cost = actual_dist
                            best_route_idx = r_idx
                            best_pos = p
                if best_route_idx != -1:
                    routes[best_route_idx].insert(best_pos, c)
                    unvisited.remove(c)
                else:
                    routes.append([c])
                    unvisited.remove(c)

        return routes

    def cluster_based():
        """Cluster customers by angle and build routes per cluster."""
        # Sort by angle from depot
        angles = [np.arctan2(coords[c][1] - coords[0][1], coords[c][0] - coords[0][0]) for c in customers]
        angle_order = [customers[i] for i in np.argsort(angles)]

        routes = []
        cluster = []
        cluster_demand = 0.0

        for c in angle_order:
            if time.time() - start_time > DEADLINE:
                break
            if cluster_demand + demands[c] > capacity:
                if cluster:
                    route = construct_route(cluster)
                    if route:
                        routes.append(route)
                    cluster = []
                    cluster_demand = 0.0
            cluster.append(c)
            cluster_demand += demands[c]

        if cluster:
            route = construct_route(cluster)
            if route:
                routes.append(route)

        # Handle any unvisited
        visited = set()
        for r in routes:
            for node in r:
                if node not in stations:
                    visited.add(node)
        for c in customers:
            if c not in visited:
                best_route_idx = -1
                best_pos = -1
                best_cost = float('inf')
                for r_idx, route in enumerate(routes):
                    current_demand = sum(demands[node] for node in route if node not in stations)
                    if current_demand + demands[c] > capacity + 1e-9:
                        continue
                    for p in range(len(route) + 1):
                        new_route = route[:p] + [c] + route[p:]
                        valid, actual_dist = is_feasible(new_route)
                        if valid and actual_dist < best_cost:
                            best_cost = actual_dist
                            best_route_idx = r_idx
                            best_pos = p
                if best_route_idx != -1:
                    routes[best_route_idx].insert(best_pos, c)
                else:
                    routes.append([c])

        return routes

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def validate_solution(routes):
        """Check all constraints."""
        visited = set()
        for route in routes:
            valid, _ = is_feasible(route)
            if not valid:
                return False
            for node in route:
                if node not in stations:
                    if node in visited:
                        return False
                    visited.add(node)
        return visited == set(customers)

    # Generate multiple initial solutions
    initial_solutions = []

    # Greedy by due date
    r1 = greedy_insertion("due")
    if validate_solution(r1):
        initial_solutions.append(r1)

    # Greedy by angle
    r2 = greedy_insertion("angle")
    if validate_solution(r2):
        initial_solutions.append(r2)

    # Nearest neighbor
    r3 = nearest_neighbor()
    if validate_solution(r3):
        initial_solutions.append(r3)

    # Cluster based
    r4 = cluster_based()
    if validate_solution(r4):
        initial_solutions.append(r4)

    if not initial_solutions:
        # Fallback: simple greedy
        routes = greedy_insertion("due")
        return routes

    # Select best initial solution
    best_routes = min(initial_solutions, key=total_distance)

    # Local search: 2-opt and OR-opt on each route
    for r_idx in range(len(best_routes)):
        if time.time() - start_time > DEADLINE:
            break
        best_routes[r_idx] = two_opt(best_routes[r_idx])
        if time.time() - start_time > DEADLINE:
            break
        best_routes[r_idx] = or_opt(best_routes[r_idx])

    # Inter-route improvements
    if time.time() - start_time < DEADLINE:
        best_routes = relocate_improve(best_routes)

    if time.time() - start_time < DEADLINE:
        best_routes = merge_routes(best_routes)

    # Second pass of local search
    if time.time() - start_time < DEADLINE:
        for r_idx in range(len(best_routes)):
            if time.time() - start_time > DEADLINE:
                break
            best_routes[r_idx] = two_opt(best_routes[r_idx])
            if time.time() - start_time > DEADLINE:
                break
            best_routes[r_idx] = or_opt(best_routes[r_idx])

    # Final validation
    if not validate_solution(best_routes):
        # Fallback
        routes = greedy_insertion("due")
        return routes

    return best_routes

# EVOLVE-BLOCK-END
