import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 148.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_ids = set(stations)
    station_id_list = sorted([s for s in range(n_nodes) if s in station_ids])

    cx = coords[:, 0]
    cy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i, j] = d
            dist[j, i] = d

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_ids

    def check_route(route):
        if not route:
            return 0.0, True, 0.0, battery_max
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        for node in route:
            if is_customer(node):
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False, 0.0, 0.0
        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            if b < -_EPS:
                return float('inf'), False, 0.0, 0.0
            if is_station(node):
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return float('inf'), False, 0.0, 0.0
                t += service[node]
            prev = node
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        if b < -_EPS:
            return float('inf'), False, 0.0, 0.0
        if t > horizon + _EPS:
            return float('inf'), False, 0.0, 0.0
        return total_dist, True, t, b

    def try_insert_station(route, max_stations=2):
        total_dist, valid, t, b = check_route(route)
        if valid:
            return route, total_dist
        best_route = None
        best_cost = float('inf')
        for s in station_id_list:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c, v, _, _ = check_route(cand)
                if v and c < best_cost:
                    best_cost = c
                    best_route = cand
        if best_route:
            return best_route, best_cost
        for s1 in station_id_list:
            for s2 in station_id_list:
                for i in range(len(route) + 1):
                    cand1 = route[:i] + [s1] + route[i:]
                    c1, v1, _, _ = check_route(cand1)
                    if not v1:
                        continue
                    for j in range(len(cand1) + 1):
                        cand2 = cand1[:j] + [s2] + cand1[j:]
                        c2, v2, _, _ = check_route(cand2)
                        if v2 and c2 < best_cost:
                            best_cost = c2
                            best_route = cand2
        if best_route:
            return best_route, best_cost
        return None, float('inf')

    # --- MILP Formulation using scipy.optimize.linprog ---
    # Decision variables:
    # x[i][j] = 1 if arc i->j is used (i,j in 0..n_nodes-1)
    # t[i] = arrival time at node i
    # b[i] = battery level upon arrival at node i (before service/recharge)
    # y[i] = 1 if node i is visited (for customers)
    #
    # Objective: minimize sum(dist[i][j] * x[i][j])
    #
    # Constraints:
    # 1. Flow conservation: for each node, inflow = outflow
    #    - Depot (0): outflow - inflow = number of vehicles (we use 1 vehicle for simplicity, or allow multiple)
    #    - Customers: inflow = outflow = 1 (must be visited exactly once)
    #    - Stations: inflow = outflow (can be visited 0 or more times)
    #
    # 2. Capacity: sum(demands[j] * x[i][j]) <= capacity for each vehicle
    #    (simplified: total demand of visited customers <= capacity)
    #
    # 3. Battery: b[j] >= b[i] - dist[i][j] for arc i->j, b[j] >= 0
    #    At station: b[j] = battery_max (recharge)
    #
    # 4. Time windows: t[j] >= t[i] + dist[i][j]/speed - M*(1-x[i][j])
    #    t[j] >= ready[j], t[j] <= due[j] for customers
    #
    # 5. Service time: t[j] += service[j] after arrival for customers
    #
    # This is complex. Let me use a simpler approach:
    # Since n_cust=25 and n_nodes=30, we can try a different MILP formulation.
    #
    # Actually, let me implement a proper MILP with scipy.optimize.milp (available in newer scipy)
    # or fall back to a heuristic if not available.

    def solve_milp():
        """Solve EVRPTW as MILP using scipy.optimize.milp"""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return None

        # Variables:
        # x[i,j] for i in 0..n_nodes-1, j in 0..n_nodes-1 (i != j): binary
        # t[i] for i in 0..n_nodes-1: continuous >= 0
        # b[i] for i in 0..n_nodes-1: continuous >= 0
        #
        # Total variables: n_nodes*(n_nodes-1) + n_nodes + n_nodes
        # For n_nodes=30: 30*29 + 30 + 30 = 870 + 60 = 930 variables

        n_arc = n_nodes * (n_nodes - 1)
        n_t = n_nodes
        n_b = n_nodes
        n_vars = n_arc + n_t + n_b

        # Variable indices
        def arc_idx(i, j):
            if i == j:
                return None
            return i * (n_nodes - 1) + (j if j < i else j - 1)

        def t_idx(i):
            return n_arc + i

        def b_idx(i):
            return n_arc + n_t + i

        # Objective: minimize total distance
        c = np.zeros(n_vars)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                c[arc_idx(i, j)] = dist[i, j]

        # Bounds: x binary [0,1], t >= 0, b >= 0
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        # t and b are unbounded above (use large M)
        M = horizon * 10
        for i in range(n_t):
            ub[n_arc + i] = M
        for i in range(n_b):
            ub[n_arc + n_t + i] = battery_max

        # Integer variables: x are binary (0 or 1)
        integrality = np.zeros(n_vars)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    integrality[arc_idx(i, j)] = 1  # binary

        # Constraints
        A_rows = []
        lb_rows = []
        ub_rows = []

        # 1. Flow conservation for depot: outflow - inflow = 1 (single vehicle)
        row = np.zeros(n_vars)
        for j in range(1, n_nodes):
            row[arc_idx(0, j)] = 1  # outflow from depot
        for i in range(1, n_nodes):
            row[arc_idx(i, 0)] = -1  # inflow to depot
        A_rows.append(row)
        lb_rows.append(1)
        ub_rows.append(1)

        # 2. Flow conservation for customers: inflow = outflow = 1
        for k in cust_ids:
            row = np.zeros(n_vars)
            for j in range(n_nodes):
                if j != k:
                    row[arc_idx(j, k)] = 1  # inflow to k
            for j in range(n_nodes):
                if j != k:
                    row[arc_idx(k, j)] = -1  # outflow from k
            A_rows.append(row)
            lb_rows.append(0)
            ub_rows.append(0)

        # 3. Flow conservation for stations: inflow = outflow
        for k in station_id_list:
            row = np.zeros(n_vars)
            for j in range(n_nodes):
                if j != k:
                    row[arc_idx(j, k)] = 1
            for j in range(n_nodes):
                if j != k:
                    row[arc_idx(k, j)] = -1
            A_rows.append(row)
            lb_rows.append(0)
            ub_rows.append(0)

        # 4. Capacity constraint: total demand of visited customers <= capacity
        #    sum(demands[j] * x[i,j]) <= capacity for each i (simplified)
        #    Actually, we need: sum over all arcs into customers of demands[j] <= capacity
        #    But this is tricky with single vehicle. Let's use:
        #    sum(demands[j] * (sum_i x[i,j])) <= capacity
        row = np.zeros(n_vars)
        for j in cust_ids:
            for i in range(n_nodes):
                if i != j:
                    row[arc_idx(i, j)] = demands[j]
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(capacity)

        # 5. Battery constraints: b[j] >= b[i] - dist[i][j] + M*(1-x[i,j])
        #    For station j: b[j] = battery_max
        #    For customer j: b[j] >= 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                if j in station_ids:
                    # b[j] = battery_max when arriving at station
                    # b[j] - battery_max >= 0 and b[j] - battery_max <= 0
                    # But only if x[i,j] = 1
                    # Use: b[j] >= battery_max - M*(1-x[i,j])
                    row = np.zeros(n_vars)
                    row[b_idx(j)] = 1
                    row[arc_idx(i, j)] = -M
                    A_rows.append(row)
                    lb_rows.append(battery_max - M)
                    ub_rows.append(np.inf)
                    # And b[j] <= battery_max + M*(1-x[i,j])
                    row2 = np.zeros(n_vars)
                    row2[b_idx(j)] = 1
                    row2[arc_idx(i, j)] = M
                    A_rows.append(row2)
                    lb_rows.append(-np.inf)
                    ub_rows.append(battery_max + M)
                else:
                    # b[j] >= b[i] - dist[i][j] + M*(1-x[i,j])
                    row = np.zeros(n_vars)
                    row[b_idx(j)] = 1
                    row[b_idx(i)] = -1
                    row[arc_idx(i, j)] = -M
                    A_rows.append(row)
                    lb_rows.append(-dist[i, j] - M)
                    ub_rows.append(np.inf)

        # 6. Time window constraints: t[j] >= t[i] + dist[i][j]/speed + service[i] - M*(1-x[i,j])
        #    for i not depot, j customer
        #    t[j] >= ready[j], t[j] <= due[j]
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                if is_customer(j):
                    # t[j] >= t[i] + dist[i][j]/speed + service[i] - M*(1-x[i,j])
                    row = np.zeros(n_vars)
                    row[t_idx(j)] = 1
                    row[t_idx(i)] = -1
                    row[arc_idx(i, j)] = -M
                    A_rows.append(row)
                    lb_rows.append(dist[i, j] / speed + service[i] - M)
                    ub_rows.append(np.inf)

        # t[j] >= ready[j] for customers
        for j in cust_ids:
            row = np.zeros(n_vars)
            row[t_idx(j)] = 1
            A_rows.append(row)
            lb_rows.append(ready[j])
            ub_rows.append(np.inf)

        # t[j] <= due[j] for customers
        for j in cust_ids:
            row = np.zeros(n_vars)
            row[t_idx(j)] = 1
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(due[j])

        # 7. Horizon constraint: time at depot (return) <= horizon
        #    This is complex. We'll check after solving.

        if not A_rows:
            return None

        A = np.array(A_rows)
        lb_arr = np.array(lb_rows)
        ub_arr = np.array(ub_rows)

        # Replace -inf with large negative number
        lb_arr = np.where(np.isneginf(lb_arr), -M * 10, lb_arr)
        ub_arr = np.where(np.isposinf(ub_arr), M * 10, ub_arr)

        constraints = LinearConstraint(A, lb_arr, ub_arr)
        bounds = Bounds(lb, ub)

        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if not result.success:
            return None

        x = result.x[:n_arc]

        # Extract routes from x
        routes = []
        visited = set()
        current = 0
        route = []
        while True:
            next_node = None
            for j in range(1, n_nodes):
                if x[arc_idx(current, j)] > 0.5:
                    next_node = j
                    break
            if next_node is None or next_node == 0:
                break
            route.append(next_node)
            if is_customer(next_node):
                visited.add(next_node)
            current = next_node
            if len(route) > 100:  # Safety
                break
        if route:
            routes.append(route)

        # Check if all customers visited
        if visited != set(cust_ids):
            return None

        return routes

    # --- Try MILP first ---
    milp_routes = solve_milp()

    if milp_routes is not None and time.time() < DEADLINE:
        # Validate MILP solution
        valid = True
        total_dist = 0.0
        served = set()
        for r in milp_routes:
            c, v, _, _ = check_route(r)
            if not v:
                valid = False
                break
            total_dist += c
            for node in r:
                if node in served:
                    valid = False
                    break
                served.add(node)
            if not valid:
                break
        if valid and served == set(cust_ids):
            return milp_routes

    # --- Fallback: Heuristic (same as before) ---
    sorted_custs = sorted(cust_ids, key=lambda c: due[c])
    routes = []
    unvisited = sorted_custs[:]

    while unvisited:
        route = []
        c = unvisited.pop(0)
        route.append(c)
        changed = True
        while changed:
            changed = False
            best_cust = None
            best_pos = -1
            best_cost = float('inf')
            current_load = sum(demands[n] for n in route if is_customer(n))
            for c_idx, c in enumerate(unvisited):
                if current_load + demands[c] > capacity + _EPS:
                    continue
                for pos in range(len(route) + 1):
                    cand = route[:pos] + [c] + route[pos:]
                    cost, valid, _, _ = check_route(cand)
                    if valid and cost < best_cost:
                        best_cost = cost
                        best_cust = c
                        best_pos = pos
            if best_cust is not None:
                route = route[:best_pos] + [best_cust] + route[best_pos:]
                unvisited.remove(best_cust)
                current_load += demands[best_cust]
                changed = True
        if not check_route(route)[1]:
            rep, _ = try_insert_station(route)
            if rep:
                route = rep
        routes.append(route)

    # Local search
    def get_route_cost(route):
        return check_route(route)[0]

    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE:
                break
            route = routes[ri]
            n = len(route)
            if n < 2:
                continue
            cur_cost, cur_valid, _, _ = check_route(route)
            if not cur_valid:
                rep, rcost = try_insert_station(route)
                if rep:
                    routes[ri] = rep
                    cur_cost = rcost
                    cur_valid = True
                else:
                    continue
            best_route = list(route)
            best_cost = cur_cost
            route_improved = True
            max_iter = 10
            while route_improved and max_iter > 0:
                route_improved = False
                max_iter -= 1
                for i in range(n - 1):
                    if time.time() > DEADLINE:
                        break
                    for j in range(i + 1, n):
                        cand = best_route[:i] + list(reversed(best_route[i:j + 1])) + best_route[j + 1:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            route_improved = True
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True

        if time.time() > DEADLINE:
            break
        for ri in range(len(routes)):
            if time.time() > DEADLINE:
                break
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE:
                    break
                cust = routes[ri][ci]
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    if time.time() > DEADLINE:
                        break
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE:
                            break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            continue
                        load_to = sum(demands[n] for n in cand_to if is_customer(n))
                        if load_to > capacity + _EPS:
                            continue
                        old_cost = c_from + get_route_cost(routes[rj])
                        new_cost = c_from + c_to
                        if new_cost < old_cost - 1e-7:
                            routes[ri] = temp_from
                            routes[rj] = cand_to
                            improved = True
                            ri = -1
                            break
                    if ri == -1:
                        break
                if ri == -1:
                    break
            if ri == -1:
                break

        routes = [r for r in routes if r]

        if time.time() > DEADLINE:
            break
        swapped = False
        for ri in range(len(routes)):
            if swapped or time.time() > DEADLINE:
                break
            for rj in range(ri + 1, len(routes)):
                if swapped or time.time() > DEADLINE:
                    break
                for ci in range(len(routes[ri])):
                    if swapped or time.time() > DEADLINE:
                        break
                    cust1 = routes[ri][ci]
                    for cj in range(len(routes[rj])):
                        if swapped or time.time() > DEADLINE:
                            break
                        cust2 = routes[rj][cj]
                        load_i = sum(demands[n] for n in routes[ri] if is_customer(n))
                        load_j = sum(demands[n] for n in routes[rj] if is_customer(n))
                        new_load_i = load_i - demands[cust1] + demands[cust2]
                        new_load_j = load_j - demands[cust2] + demands[cust1]
                        if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                            continue
                        new_route_i = routes[ri][:]
                        new_route_i[ci] = cust2
                        new_route_j = routes[rj][:]
                        new_route_j[cj] = cust1
                        c_i, v_i, _, _ = check_route(new_route_i)
                        c_j, v_j, _, _ = check_route(new_route_j)
                        if v_i and v_j:
                            old_c_i = get_route_cost(routes[ri])
                            old_c_j = get_route_cost(routes[rj])
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                improved = True
                                swapped = True
                                break
                if swapped:
                    break
            if swapped:
                break

    # Final validation
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
        c, v, _, _ = check_route(r)
        if not v:
            valid = False
            break
        total_dist += c
        for node in r:
            if node in served:
                valid = False
                break
            served.add(node)
        if not valid:
            break

    if not valid or served != set(cust_ids):
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                rep, _ = try_insert_station(routes[i])
                if rep:
                    routes[i] = rep

    return routes
