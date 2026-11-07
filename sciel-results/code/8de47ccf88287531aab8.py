"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def plan(instance):
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    # Precompute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    def build_route(seq):
        """Return (total_dist, feasible) for a customer sequence (depot implicit)."""
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for c in seq:
            d = dist[prev, c]
            if b < d:
                return d_total, False
            b -= d
            t += d / speed
            if t > due[c]:
                return d_total, False
            if t < ready[c]:
                t = ready[c]
            t += service[c]
            load += demands[c]
            if load > capacity:
                return d_total, False
            d_total += d
            prev = c
        # return to depot
        d = dist[prev, 0]
        if b < d:
            return d_total, False
        b -= d
        t += d / speed
        if t > horizon:
            return d_total, False
        d_total += d
        return d_total, True

    def build_route_with_stations(seq, station_plan):
        """seq: list of customers. station_plan: list of (after_customer_index, [station_ids])
        We'll handle stations inline. Returns (dist, feasible)."""
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        # Build full visit list: customers with optional station visits after each
        for i, c in enumerate(seq):
            d = dist[prev, c]
            if b < d:
                return d_total, False
            b -= d
            t += d / speed
            if t > due[c]:
                return d_total, False
            if t < ready[c]:
                t = ready[c]
            t += service[c]
            load += demands[c]
            if load > capacity:
                return d_total, False
            d_total += d
            prev = c
            # Stations after this customer
            if i in station_plan:
                for s in station_plan[i]:
                    d = dist[prev, s]
                    if b < d:
                        return d_total, False
                    b -= d
                    t += d / speed
                    d_total += d
                    # recharge
                    if b < 0:
                        b = 0.0
                    t += (battery_cap - b) * recharge_rate
                    b = battery_cap
                    prev = s
        # return to depot
        d = dist[prev, 0]
        if b < d:
            return d_total, False
        b -= d
        t += d / speed
        if t > horizon:
            return d_total, False
        d_total += d
        return d_total, True

    def find_station_plan(seq):
        """Find a station plan that makes seq feasible. Returns dict or None."""
        # Greedy: insert stations when battery would go below 0
        # We need to find stations to visit between customers
        # Simple approach: try inserting nearest station when needed
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        plan = {}
        for i, c in enumerate(seq):
            d = dist[prev, c]
            if b < d:
                # Need to visit a station before going to c
                # Find nearest station from prev
                best_s = None
                best_d = float('inf')
                for s in stations:
                    ds = dist[prev, s]
                    if ds < best_d:
                        best_d = ds
                        best_s = s
                if best_s is None:
                    return None
                # Visit station
                d2 = dist[prev, best_s]
                if b < d2:
                    return None
                b -= d2
                t += d2 / speed
                d_total += d2
                if b < 0:
                    b = 0.0
                t += (battery_cap - b) * recharge_rate
                b = battery_cap
                prev = best_s
                d = dist[prev, c]
                if b < d:
                    return None
            b -= d
            t += d / speed
            if t > due[c]:
                return None
            if t < ready[c]:
                t = ready[c]
            t += service[c]
            load += demands[c]
            if load > capacity:
                return None
            d_total += d
            prev = c
        # return to depot
        d = dist[prev, 0]
        if b < d:
            return None
        b -= d
        t += d / speed
        if t > horizon:
            return None
        d_total += d
        return plan  # We don't actually need the plan, just feasibility

    # Initial solution: nearest neighbor insertion
    customers = list(range(1, n_customers + 1))
    routes = []
    unassigned = set(customers)

    # Sort by due time for initial ordering
    for c in sorted(customers, key=lambda x: due[x]):
        if c not in unassigned:
            continue
        # Try to insert into existing route
        best_route = -1
        best_pos = -1
        best_cost = float('inf')
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                d, feas = build_route(new_route)
                if feas and d < best_cost:
                    best_cost = d
                    best_route = ri
                    best_pos = pos
        if best_route >= 0:
            routes[best_route].insert(best_pos, c)
            unassigned.discard(c)
        else:
            # Start new route
            routes.append([c])
            unassigned.discard(c)

    # Local search: relocate
    def try_relocate(routes):
        """Try to move each customer to a better position in any route."""
        improved = False
        n_routes = len(routes)
        for ri in range(n_routes):
            for ci in range(len(routes[ri])):
                c = routes[ri][ci]
                # Remove c
                orig_route = routes[ri][:]
                routes[ri] = orig_route[:ci] + orig_route[ci + 1:]
                best_r = -1
                best_p = -1
                best_d = float('inf')
                for rj in range(len(routes)):
                    for pos in range(len(routes[rj]) + 1):
                        new_route = routes[rj][:pos] + [c] + routes[rj][pos:]
                        d, feas = build_route(new_route)
                        if feas and d < best_d:
                            best_d = d
                            best_r = rj
                            best_p = pos
                if best_r >= 0:
                    # Check if this is actually an improvement
                    # Compute total cost before and after
                    total_before = sum(build_route(r)[0] for r in [orig_route] + routes)
                    routes[best_r] = routes[best_r][:best_p] + [c] + routes[best_r][best_p:]
                    total_after = sum(build_route(r)[0] for r in routes)
                    if total_after < total_before - 1e-9:
                        improved = True
                    else:
                        # Revert
                        routes[ri] = orig_route
                else:
                    routes[ri] = orig_route
        return improved

    # Local search: 2-opt within a route
    def try_two_opt(routes):
        improved = False
        for ri in range(len(routes)):
            route = routes[ri]
            n = len(route)
            if n < 2:
                continue
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    d_new, feas_new = build_route(new_route)
                    d_old, feas_old = build_route(route)
                    if feas_new and d_new < d_old - 1e-9:
                        routes[ri] = new_route
                        improved = True
                        break
                if improved:
                    break
        return improved

    # Run local search with deadline guard
    best_routes = [list(r) for r in routes]
    best_cost = sum(build_route(r)[0] for r in routes)
    no_improve_count = 0

    while time.time() - t0 < DEADLINE:
        if no_improve_count > 20:
            break
        # Try relocate
        improved = try_relocate(routes)
        if improved:
            cost = sum(build_route(r)[0] for r in routes)
            if cost < best_cost:
                best_cost = cost
                best_routes = [list(r) for r in routes]
            no_improve_count = 0
        else:
            # Try 2-opt
            improved = try_two_opt(routes)
            if improved:
                cost = sum(build_route(r)[0] for r in routes)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = [list(r) for r in routes]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 10:
                    break

    # Return best routes found
    return best_routes
# EVOLVE-BLOCK-END
