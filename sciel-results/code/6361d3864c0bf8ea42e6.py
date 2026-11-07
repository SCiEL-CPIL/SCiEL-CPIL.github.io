import numpy as np
import time
import math

def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate(route):
        """Simulate route, return (valid, total_dist, total_time)."""
        t = 0.0
        b = battery_cap
        load = 0.0
        prev = 0
        total_d = 0.0

        for node in route:
            d = dist[prev, node]
            total_d += d
            t += d / speed
            b -= d

            if b < -1e-9:
                return False, 0.0, 0.0

            if node in station_set:
                energy_needed = battery_cap - b
                if energy_needed > 0:
                    t += energy_needed * recharge_rate
                b = battery_cap
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, 0.0, 0.0
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False, 0.0, 0.0
                t += service[node]

            prev = node

        d = dist[prev, 0]
        total_d += d
        t += d / speed
        b -= d

        if b < -1e-9 or t > horizon + 1e-9:
            return False, 0.0, 0.0

        return True, total_d, t

    def repair_route(route):
        """Try to make route feasible by inserting stations. Returns (feasible_route, dist) or (None, 0)."""
        current = list(route)
        for _ in range(20):
            valid, total_d, _ = simulate(current)
            if valid:
                return current, total_d

            # Find first battery failure
            t = 0.0
            b = battery_cap
            prev = 0
            fail_idx = -1
            for i, node in enumerate(current):
                d = dist[prev, node]
                t += d / speed
                b -= d
                if b < -1e-9:
                    fail_idx = i
                    break
                if node in station_set:
                    energy_needed = battery_cap - b
                    if energy_needed > 0:
                        t += energy_needed * recharge_rate
                    b = battery_cap
                else:
                    if t < ready[node]:
                        t = ready[node]
                    if t > due[node] + 1e-9:
                        return None, 0.0
                    t += service[node]
                prev = node

            if fail_idx == -1:
                return None, 0.0

            prev_node = current[fail_idx - 1] if fail_idx > 0 else 0
            next_node = current[fail_idx]

            best_s = -1
            best_cost = float('inf')
            for s in stations:
                new_route = current[:fail_idx] + [s] + current[fail_idx:]
                valid2, d2, _ = simulate(new_route)
                if valid2:
                    cost = dist[prev_node, s] + dist[s, next_node] - dist[prev_node, next_node]
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s

            if best_s == -1:
                return None, 0.0

            current = current[:fail_idx] + [best_s] + current[fail_idx:]

        return None, 0.0

    # --- Savings-based construction ---
    routes = [[c] for c in customers]

    def build_map(routes):
        m = {}
        for i, r in enumerate(routes):
            for c in r:
                if c not in station_set:
                    m[c] = i
        return m

    route_map = build_map(routes)

    # Precompute savings for all pairs
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            c1 = customers[i]
            c2 = customers[j]
            s_val = dist[0, c1] + dist[0, c2] - dist[c1, c2]
            savings.append((s_val, c1, c2))
    savings.sort(reverse=True)

    for s_val, c1, c2 in savings:
        if time.time() > deadline - 30:
            break

        r1 = route_map.get(c1)
        r2 = route_map.get(c2)
        if r1 is None or r2 is None or r1 == r2:
            continue

        route1 = routes[r1]
        route2 = routes[r2]

        # Try merging: c1 at end of route1, c2 at start of route2
        if route1[-1] == c1 and route2[0] == c2:
            new_route = route1 + route2
            repaired, _ = repair_route(new_route)
            if repaired is not None:
                routes[r1] = repaired
                del routes[r2]
                route_map = build_map(routes)
                continue

        # Try merging: c2 at end of route1, c1 at start of route2
        if route1[-1] == c2 and route2[0] == c1:
            new_route = route1 + route2
            repaired, _ = repair_route(new_route)
            if repaired is not None:
                routes[r1] = repaired
                del routes[r2]
                route_map = build_map(routes)
                continue

    # Remove empty routes
    routes = [r for r in routes if r]

    # --- Local Search ---
    def total_distance(routes):
        total = 0.0
        for r in routes:
            valid, d, _ = simulate(r)
            if valid:
                total += d
        return total

    def try_2opt_intra(routes):
        """2-opt within each route."""
        improved = False
        for i in range(len(routes)):
            if len(routes[i]) < 3:
                continue
            route = routes[i]
            old_dist = simulate(route)[1]
            for a in range(len(route) - 1):
                for b in range(a + 2, len(route)):
                    new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                    repaired, _ = repair_route(new_route)
                    if repaired is not None:
                        new_dist = simulate(repaired)[1]
                        if new_dist < old_dist - 1e-6:
                            routes[i] = repaired
                            old_dist = new_dist
                            improved = True
        return improved

    def try_move_intra(routes):
        """Move a single customer to a different position in the same route."""
        improved = False
        for i in range(len(routes)):
            route = routes[i]
            if len(route) < 2:
                continue
            old_dist = simulate(route)[1]
            for a in range(len(route)):
                c = route[a]
                rest = route[:a] + route[a + 1:]
                if not rest:
                    continue
                for b in range(len(rest)):
                    if b == a or b == a - 1:
                        continue
                    new_route = rest[:b] + [c] + rest[b:]
                    repaired, _ = repair_route(new_route)
                    if repaired is not None:
                        new_dist = simulate(repaired)[1]
                        if new_dist < old_dist - 1e-6:
                            routes[i] = repaired
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        return improved

    def try_inter_move(routes):
        """Move a customer from one route to another."""
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            route_i = routes[i]
            old_dist_i = simulate(route_i)[1]
            for a in range(len(route_i)):
                c = route_i[a]
                new_i = route_i[:a] + route_i[a + 1:]
                if not new_i:
                    repaired_i, dist_i = None, 0.0
                else:
                    repaired_i, dist_i = repair_route(new_i)
                    if repaired_i is None:
                        continue

                for j in range(len(routes)):
                    if j == i:
                        continue
                    route_j = routes[j]
                    old_dist_j = simulate(route_j)[1]
                    best_pos = -1
                    best_dist_j = float('inf')
                    best_repaired_j = None

                    for b in range(len(route_j) + 1):
                        new_j = route_j[:b] + [c] + route_j[b:]
                        repaired_j, dist_j = repair_route(new_j)
                        if repaired_j is not None and dist_j < best_dist_j:
                            best_dist_j = dist_j
                            best_pos = b
                            best_repaired_j = repaired_j

                    if best_pos != -1:
                        new_total = dist_i + best_dist_j
                        old_total = old_dist_i + old_dist_j
                        if new_total < old_total - 1e-6:
                            if new_i:
                                routes[i] = repaired_i
                            else:
                                routes[i] = []
                            routes[j] = best_repaired_j
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        # Clean up empty routes
        routes[:] = [r for r in routes if r]
        return improved

    def try_inter_swap(routes):
        """Swap two customers between routes."""
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            route_i = routes[i]
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                route_j = routes[j]
                old_dist_i = simulate(route_i)[1]
                old_dist_j = simulate(route_j)[1]
                old_total = old_dist_i + old_dist_j

                for a in range(len(route_i)):
                    ci = route_i[a]
                    for b in range(len(route_j)):
                        cj = route_j[b]
                        if ci == cj:
                            continue
                        new_i = route_i[:a] + [cj] + route_i[a + 1:]
                        new_j = route_j[:b] + [ci] + route_j[b + 1:]
                        repaired_i, dist_i = repair_route(new_i)
                        repaired_j, dist_j = repair_route(new_j)
                        if repaired_i is not None and repaired_j is not None:
                            new_total = dist_i + dist_j
                            if new_total < old_total - 1e-6:
                                routes[i] = repaired_i
                                routes[j] = repaired_j
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    # Iterate local search until no improvement or time runs out
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        any_improved = False

        if time.time() < deadline - 5:
            if try_2opt_intra(routes):
                any_improved = True
                continue

        if time.time() < deadline - 5:
            if try_move_intra(routes):
                any_improved = True
                continue

        if time.time() < deadline - 5:
            if try_inter_move(routes):
                any_improved = True
                continue

        if time.time() < deadline - 5:
            if try_inter_swap(routes):
                any_improved = True
                continue

        if not any_improved:
            break

    # Final cleanup: remove empty routes
    routes = [r for r in routes if r]

    # Verify all customers are served
    served = set()
    for r in routes:
        for node in r:
            if node not in station_set:
                served.add(node)

    # If any customer is missing, add them as separate routes (shouldn't happen)
    for c in customers:
        if c not in served:
            routes.append([c])

    return routes
