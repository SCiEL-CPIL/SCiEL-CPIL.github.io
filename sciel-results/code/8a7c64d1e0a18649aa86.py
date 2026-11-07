"""EV routing with time windows: constructive heuristic + local improvement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _check_route(route, inst):
    """Simulate a route (without depot endpoints). Return (ok, dist, end_time, end_batt).
    Route is a list of node ids (customers and/or stations). Depot implicit at ends.
    Returns ok=False if any constraint violated."""
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    capacity = inst["capacity"]
    battery_cap = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    n_customers = inst["n_customers"]

    total_demand = 0
    t = 0.0
    batt = float(battery_cap)
    dist = 0.0
    cur = 0  # depot
    for node in route:
        # travel from cur to node
        dx = coords[node][0] - coords[cur][0]
        dy = coords[node][1] - coords[cur][1]
        d = float(np.hypot(dx, dy))
        batt -= d
        t += d / speed
        dist += d
        if batt < -1e-9:
            return (False, dist, t, batt)
        if node <= n_customers:
            # customer
            total_demand += demands[node]
            if total_demand > capacity:
                return (False, dist, t, batt)
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return (False, dist, t, batt)
            t += service[node]
        else:
            # station: recharge
            # time = (battery - energy_needed_to_full) * recharge_rate? 
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # Interpret: energy = battery_cap (full). time = (battery_cap - batt)*recharge_rate? 
            # Actually (battery - energy): battery is current, energy is what's needed to fill.
            # We'll interpret as: charging time = (battery_cap - batt) * recharge_rate, then batt = battery_cap
            charge_time = (battery_cap - batt) * recharge_rate
            t += charge_time
            batt = float(battery_cap)
        cur = node
    # return to depot
    dx = coords[0][0] - coords[cur][0]
    dy = coords[0][1] - coords[cur][1]
    d = float(np.hypot(dx, dy))
    batt -= d
    t += d / speed
    dist += d
    if batt < -1e-9:
        return (False, dist, t, batt)
    if t > horizon + 1e-9:
        return (False, dist, t, batt)
    return (True, dist, t, batt)


def _route_dist(route, inst):
    """Total Euclidean distance of route including depot legs. Assumes feasible-ish; just distance."""
    coords = inst["coords"]
    dist = 0.0
    cur = 0
    for node in route:
        dx = coords[node][0] - coords[cur][0]
        dy = coords[node][1] - coords[cur][1]
        dist += float(np.hypot(dx, dy))
        cur = node
    dx = coords[0][0] - coords[cur][0]
    dy = coords[0][1] - coords[cur][1]
    dist += float(np.hypot(dx, dy))
    return dist


def _total_dist(routes, inst):
    return sum(_route_dist(r, inst) for r in routes)


def _make_station_insertion(route, node, station, inst):
    """Return new route with station inserted at best position to keep feasibility, or None."""
    coords = inst["coords"]
    # try inserting station right before node, or after node, or at various spots
    candidates = []
    # position just before node
    idx = route.index(node) if node in route else None
    if idx is not None:
        candidates.append((idx, station))
        candidates.append((idx + 1, station))
    # also try at end
    candidates.append((len(route), station))
    # try all positions (small routes)
    for pos in range(len(route) + 1):
        if (pos, station) not in candidates:
            candidates.append((pos, station))
    best = None
    best_d = None
    for pos, st in candidates:
        new_route = route[:pos] + [st] + route[pos:]
        ok, dist, t, b = _check_route(new_route, inst)
        if ok:
            if best_d is None or dist < best_d:
                best = new_route
                best_d = dist
    return best


def _constructive(inst):
    """Greedy nearest-neighbour packing with time-window and battery awareness."""
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    n_customers = inst["n_customers"]
    stations = inst["stations"]
    capacity = inst["capacity"]
    battery_cap = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]

    customers = list(range(1, n_customers + 1))
    # sort by due time (earliest first) as a heuristic
    customers.sort(key=lambda v: (due[v], ready[v]))

    unserved = set(customers)
    routes = []

    def try_build(start_customer):
        route = [start_customer]
        total_demand = demands[start_customer]
        # simulate to know current state
        t = 0.0
        batt = float(battery_cap)
        cur = 0
        dx = coords[start_customer][0] - coords[0][0]
        dy = coords[start_customer][1] - coords[0][1]
        d = float(np.hypot(dx, dy))
        batt -= d
        t += d / speed
        t = max(t, ready[start_customer])
        if t > due[start_customer]:
            return None
        t += inst["service"][start_customer]
        cur = start_customer

        while unserved:
            best_c = None
            best_score = None
            for c in list(unserved):
                if c in route:
                    continue
                if total_demand + demands[c] > capacity:
                    continue
                # travel to c
                dx = coords[c][0] - coords[cur][0]
                dy = coords[c][1] - coords[cur][1]
                d = float(np.hypot(dx, dy))
                nbatt = batt - d
                if nbatt < -1e-9:
                    continue
                nt = t + d / speed
                nt = max(nt, ready[c])
                if nt > due[c]:
                    continue
                nt += inst["service"][c]
                # return feasibility from c
                dx2 = coords[0][0] - coords[c][0]
                dy2 = coords[0][1] - coords[c][1]
                d2 = float(np.hypot(dx2, dy2))
                if nbatt - d2 < -1e-9:
                    continue
                nt2 = nt + d2 / speed
                if nt2 > horizon:
                    continue
                # score: prefer close and early due
                score = d + 0.5 * (due[c] - nt)
                if best_score is None or score < best_score:
                    best_score = score
                    best_c = c
                    best_info = (nbatt, nt, d)
            if best_c is None:
                break
            unserved.discard(best_c)
            route.append(best_c)
            total_demand += demands[best_c]
            batt, t, d = best_info
            t = max(t, ready[best_c])
            t += inst["service"][best_c]
            cur = best_c

        # now check full route feasibility with possible station insertions
        # First, plain check
        ok, dist, t_end, b_end = _check_route(route, inst)
        if ok:
            return route
        # try inserting stations to fix battery issues
        # Insert stations greedily where battery dips lowest
        fixed = _try_fix_battery(route, inst)
        if fixed is not None:
            return fixed
        return None

    def _try_fix_battery(route, inst):
        """Try to insert charging stations so battery never <0."""
        coords = inst["coords"]
        battery_cap = inst["battery"]
        speed = inst["speed"]
        recharge_rate = inst["recharge_rate"]
        horizon = inst["horizon"]
        stations = inst["stations"]
        n_customers = inst["n_customers"]

        # simulate to find where battery goes negative
        def simulate(route):
            t = 0.0
            batt = float(battery_cap)
            cur = 0
            min_batt = batt
            neg_index = None
            for i, node in enumerate(route):
                dx = coords[node][0] - coords[cur][0]
                dy = coords[node][1] - coords[cur][1]
                d = float(np.hypot(dx, dy))
                batt -= d
                t += d / speed
                if batt < -1e-9 and neg_index is None:
                    neg_index = i
                if batt < min_batt:
                    min_batt = batt
                if node <= n_customers:
                    t = max(t, inst["ready"][node])
                    if t > inst["due"][node]:
                        return (False, None, t, batt)
                    t += inst["service"][node]
                else:
                    t += (battery_cap - batt) * recharge_rate
                    batt = float(battery_cap)
                cur = node
            # return leg
            dx = coords[0][0] - coords[cur][0]
            dy = coords[0][1] - coords[cur][1]
            d = float(np.hypot(dx, dy))
            batt -= d
            t += d / speed
            if batt < -1e-9:
                return (False, "return", t, batt)
            if t > horizon:
                return (False, None, t, batt)
            return (True, None, t, batt)

        cur_route = list(route)
        for _ in range(10):
            ok, neg_idx, t_end, b_end = simulate(cur_route)
            if ok:
                # full check including time windows
                ok2, dist, te, be = _check_route(cur_route, inst)
                if ok2:
                    return cur_route
                else:
                    return None
            # insert a station before the negative point
            if isinstance(neg_idx, int):
                insert_pos = neg_idx
            else:
                insert_pos = len(cur_route)
            best_new = None
            best_d = None
            for st in stations:
                new_route = cur_route[:insert_pos] + [st] + cur_route[insert_pos:]
                # check
                ok3, dist, te, be = _check_route(new_route, inst)
                if ok3:
                    if best_d is None or dist < best_d:
                        best_new = new_route
                        best_d = dist
            if best_new is None:
                # try other positions
                for pos in range(len(cur_route) + 1):
                    for st in stations:
                        new_route = cur_route[:pos] + [st] + cur_route[pos:]
                        ok3, dist, te, be = _check_route(new_route, inst)
                        if ok3:
                            if best_d is None or dist < best_d:
                                best_new = new_route
                                best_d = dist
            if best_new is None:
                return None
            cur_route = best_new
        return None

    while unserved and _now() < 120.0:
        # pick the customer with earliest due among unserved as start
        start = min(unserved, key=lambda v: due[v])
        built = try_build(start)
        if built is None:
            # try starting from each unserved
            built = None
            for c in sorted(unserved, key=lambda v: due[v]):
                b = try_build(c)
                if b is not None:
                    built = b
                    break
        if built is None:
            # fallback: single customer route
            built = [start]
            ok, dist, te, be = _check_route(built, inst)
            if not ok:
                # cannot even serve alone; still must serve; add station if needed
                fixed = _try_fix_battery(built, inst)
                if fixed is not None:
                    built = fixed
        unserved.difference_update(set(built))
        routes.append(built)

    return routes


def _local_improve(routes, inst, max_iter=30):
    """2-opt / relocate within and across routes, keeping feasibility."""
    coords = inst["coords"]
    n_customers = inst["n_customers"]
    stations = inst["stations"]

    def total():
        return _total_dist(routes, inst)

    def feasible(routes_list):
        for r in routes_list:
            ok, dist, t, b = _check_route(r, inst)
            if not ok:
                return False
        return True

    best = total()
    for it in range(max_iter):
        if _now() > 115.0:
            break
        improved = False
        # relocate a customer to another route
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                node = routes[i][j]
                if node in stations:
                    continue
                # remove node from route i
                new_i = routes[i][:j] + routes[i][j+1:]
                # try inserting into every other route at every position
                for k in range(len(routes)):
                    if k == i and new_i == []:
                        continue
                    target = routes[k] if k != i else new_i
                    for pos in range(len(target) + 1):
                        new_target = target[:pos] + [node] + target[pos:]
                        new_routes = list(routes)
                        new_routes[i] = new_i if k != i else new_i
                        new_routes[k] = new_target
                        if not new_routes or any(len(r) == 0 for r in new_routes):
                            continue
                        if feasible(new_routes):
                            nd = _total_dist(new_routes, inst)
                            if nd < best - 1e-9:
                                routes[:] = new_routes
                                best = nd
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return routes


def plan(instance):
    global _START
    _START = time.time()
    inst = instance
    routes = _constructive(inst)
    routes = _local_improve(routes, inst)
    # final safety: verify all customers served exactly once and feasible
    all_nodes = [n for r in routes for n in r]
    cust = [n for n in all_nodes if n <= inst["n_customers"]]
    if sorted(cust) != list(range(1, inst["n_customers"] + 1)):
        # fallback to seed
        customers = list(range(1, inst["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
    # ensure feasibility
    for r in routes:
        ok, dist, t, b = _check_route(r, inst)
        if not ok:
            # fallback whole thing to seed
            customers = list(range(1, inst["n_customers"] + 1))
            customers.sort(key=lambda v: instance["due"][v])
            return [[v] for v in customers]
    return routes
# EVOLVE-BLOCK-END
