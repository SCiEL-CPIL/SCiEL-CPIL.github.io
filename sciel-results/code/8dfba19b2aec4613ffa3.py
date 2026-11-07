import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))

    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, 0.0
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, 0.0
            t += service[node]
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, 0.0
            d_total += d
            prev = node
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, 0.0
        if t > horizon + _EPS:
            return False, 0.0, 0.0
        d_total += d
        return True, d_total, t

    def evaluate_routes(routes):
        total_d = 0.0
        for r in routes:
            if not r:
                continue
            valid, d, _ = check_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    def construct_due_date():
        """Sort by due date, insert greedily."""
        routes = []
        cust_sorted = sorted(customers, key=lambda c: due[c])
        for c in cust_sorted:
            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            for i, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, _ = check_route(trial)
                    if valid:
                        # delta distance
                        old_d = 0.0
                        if r:
                            old_d = dist[0, r[0]]
                            for j in range(len(r) - 1):
                                old_d += dist[r[j], r[j+1]]
                            old_d += dist[r[-1], 0]
                        delta = d - old_d
                        if delta < best_delta:
                            best_delta = delta
                            best_route_idx = i
                            best_pos = pos
            if best_route_idx != -1:
                r = routes[best_route_idx]
                routes[best_route_idx] = r[:best_pos] + [c] + r[best_pos:]
            else:
                routes.append([c])
        return routes

    def construct_sweep():
        """Sort by angle from depot, then greedy insertion."""
        angles = [math.atan2(coords[c, 1] - coords[0, 1], coords[c, 0] - coords[0, 0]) for c in customers]
        cust_sorted = sorted(customers, key=lambda c: math.atan2(coords[c, 1] - coords[0, 1], coords[c, 0] - coords[0, 0]))
        routes = []
        for c in cust_sorted:
            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            for i, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, _ = check_route(trial)
                    if valid:
                        old_d = 0.0
                        if r:
                            old_d = dist[0, r[0]]
                            for j in range(len(r) - 1):
                                old_d += dist[r[j], r[j+1]]
                            old_d += dist[r[-1], 0]
                        delta = d - old_d
                        if delta < best_delta:
                            best_delta = delta
                            best_route_idx = i
                            best_pos = pos
            if best_route_idx != -1:
                r = routes[best_route_idx]
                routes[best_route_idx] = r[:best_pos] + [c] + r[best_pos:]
            else:
                routes.append([c])
        return routes

    def construct_random(rng):
        """Random order greedy insertion."""
        cust_shuffled = list(customers)
        rng.shuffle(cust_shuffled)
        routes = []
        for c in cust_shuffled:
            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            for i, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, _ = check_route(trial)
                    if valid:
                        old_d = 0.0
                        if r:
                            old_d = dist[0, r[0]]
                            for j in range(len(r) - 1):
                                old_d += dist[r[j], r[j+1]]
                            old_d += dist[r[-1], 0]
                        delta = d - old_d
                        if delta < best_delta:
                            best_delta = delta
                            best_route_idx = i
                            best_pos = pos
            if best_route_idx != -1:
                r = routes[best_route_idx]
                routes[best_route_idx] = r[:best_pos] + [c] + r[best_pos:]
            else:
                routes.append([c])
        return routes

    def tabu_search(initial_routes, rng, max_iters, tabu_tenure=15, aspiration=True):
        """Tabu search with relocate and 2-opt moves."""
        current = [list(r) for r in initial_routes]
        cur_val, cur_dist = evaluate_routes(current)
        if not cur_val:
            return None, float('inf')
        cur_cost = cur_dist

        best = [list(r) for r in current]
        best_cost = cur_dist

        tabu_list = {}  # (customer, from_route, to_route, pos) -> expiry iteration
        tabu_count = 0
        no_improve = 0

        for it in range(max_iters):
            tabu_count += 1
            # Remove expired tabu entries
            if it % 5 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > it}

            # Generate a set of candidate moves
            moves = []
            non_empty = [i for i, r in enumerate(current) if len(r) > 0]

            if not non_empty:
                break

            # Relocate moves
            for i in non_empty:
                r_i = current[i]
                for j in range(len(r_i)):
                    c = r_i[j]
                    # Try inserting into same route at different position
                    for k in range(len(r_i)):
                        if k == j:
                            continue
                        # Skip if already tabu
                        key = (c, i, i, k)
                        if key in tabu_list and tabu_list[key] > it:
                            continue
                        trial = r_i[:k] + [c] + r_i[k:]
                        # Remove c from original position
                        if k < j:
                            trial = trial[:k] + trial[k+1:]
                        else:
                            trial = trial[:k] + trial[k+1:]
                        # Rebuild properly
                        new_r_i = r_i[:j] + r_i[j+1:]
                        if k > j:
                            k -= 1
                        new_r_i = new_r_i[:k] + [c] + new_r_i[k:]
                        trial_routes = [list(x) for x in current]
                        trial_routes[i] = new_r_i
                        val, d = evaluate_routes(trial_routes)
                        if val:
                            delta = d - cur_cost
                            moves.append((delta, 'reloc_same', i, j, k, c))
                    # Try inserting into other routes
                    for m in non_empty:
                        if m == i:
                            continue
                        r_m = current[m]
                        for k in range(len(r_m) + 1):
                            key = (c, i, m, k)
                            if key in tabu_list and tabu_list[key] > it:
                                continue
                            trial_routes = [list(x) for x in current]
                            trial_routes[i] = r_i[:j] + r_i[j+1:]
                            trial_routes[m] = r_m[:k] + [c] + r_m[k:]
                            val, d = evaluate_routes(trial_routes)
                            if val:
                                delta = d - cur_cost
                                moves.append((delta, 'reloc_diff', i, j, m, k, c))

            # 2-opt moves (within a route)
            for i in non_empty:
                r_i = current[i]
                if len(r_i) < 2:
                    continue
                for j in range(len(r_i) - 1):
                    for k in range(j + 1, len(r_i)):
                        key = ('2opt', i, j, k)
                        if key in tabu_list and tabu_list[key] > it:
                            continue
                        trial_r = r_i[:j] + r_i[j:k+1][::-1] + r_i[k+1:]
                        trial_routes = [list(x) for x in current]
                        trial_routes[i] = trial_r
                        val, d = evaluate_routes(trial_routes)
                        if val:
                            delta = d - cur_cost
                            moves.append((delta, '2opt', i, j, k, None))

            if not moves:
                no_improve += 1
                if no_improve > 50:
                    break
                continue

            # Sort by delta and pick best non-tabu, or best if aspiration
            moves.sort(key=lambda x: x[0])

            chosen = None
            for mv in moves:
                delta, mtype, a, b, c, d = mv
                if mtype == 'reloc_same':
                    key = (d, a, a, c)
                elif mtype == 'reloc_diff':
                    key = (d, a, c, b)
                else:
                    key = (mtype, a, b, c)
                is_tabu = key in tabu_list and tabu_list[key] > it
                if not is_tabu:
                    chosen = mv
                    break
                # Aspiration: accept if it improves best
                if aspiration and delta < best_cost - _EPS:
                    chosen = mv
                    break

            if chosen is None and moves:
                chosen = moves[0]

            delta, mtype, a, b, c, d = chosen

            # Apply move
            if mtype == 'reloc_same':
                r_i = current[a]
                node = d
                j = b
                k = c
                new_r = r_i[:j] + r_i[j+1:]
                if k > j:
                    k -= 1
                new_r = new_r[:k] + [node] + new_r[k:]
                current[a] = new_r
                tabu_list[(node, a, a, k)] = it + tabu_tenure
            elif mtype == 'reloc_diff':
                r_a = current[a]
                r_c = current[c]
                node = d
                j = b
                k = c
                current[a] = r_a[:j] + r_a[j+1:]
                current[c] = r_c[:k] + [node] + r_c[k:]
                tabu_list[(node, a, c, k)] = it + tabu_tenure
            elif mtype == '2opt':
                r_i = current[a]
                j, k = b, c
                current[a] = r_i[:j] + r_i[j:k+1][::-1] + r_i[k+1:]
                tabu_list[('2opt', a, j, k)] = it + tabu_tenure

            new_val, new_dist = evaluate_routes(current)
            if not new_val:
                # Revert - shouldn't happen since we checked
                continue

            if new_dist < cur_cost:
                cur_cost = new_dist
                no_improve = 0
                if new_dist < best_cost:
                    best_cost = new_dist
                    best = [list(r) for r in current]
            else:
                no_improve += 1

            if no_improve > 100:
                break

        return best, best_cost

    # Multi-start tabu search
    rng = np.random.default_rng(42)

    # Generate diverse initial solutions
    initial_solutions = []
    initial_solutions.append(construct_due_date())
    initial_solutions.append(construct_sweep())
    for _ in range(3):
        initial_solutions.append(construct_random(rng))

    # Also try a simple nearest-neighbor construction
    def construct_nn():
        routes = []
        remaining = set(customers)
        while remaining:
            # Start from depot, go to nearest remaining customer
            route = []
            prev = 0
            t = 0.0
            b = battery_cap
            load = 0.0
            while remaining:
                best_c = None
                best_d = float('inf')
                for c in remaining:
                    d = dist[prev, c]
                    if d < best_d:
                        # Quick feasibility check
                        new_t = t + d / speed
                        new_b = b - d
                        if new_b >= -_EPS and new_t <= due[c] + _EPS and load + demands[c] <= capacity + _EPS:
                            best_d = d
                            best_c = c
                if best_c is None:
                    break
                route.append(best_c)
                remaining.remove(best_c)
                t += dist[prev, best_c] / speed
                b -= dist[prev, best_c]
                t = max(t, ready[best_c])
                t += service[best_c]
                load += demands[best_c]
                prev = best_c
            if route:
                routes.append(route)
        return routes

    initial_solutions.append(construct_nn())

    global_best = None
    global_best_cost = float('inf')

    # Allocate time to each restart
    n_starts = len(initial_solutions)
    time_per_start = (DEADLINE - (time.time() - t0)) / n_starts

    for sol_idx, init in enumerate(initial_solutions):
        if time.time() > DEADLINE - 5:
            break

        val, d = evaluate_routes(init)
        if val and d < global_best_cost:
            global_best_cost = d
            global_best = [list(r) for r in init]

        # Run tabu search on this initial solution
        remaining_time = DEADLINE - (time.time() - t0) - 5
        if remaining_time <= 0:
            break
        # Estimate max iterations based on remaining time
        # Rough: each iteration ~1-5ms, so ~200-1000 iters per second
        max_iters = min(500, int(remaining_time * 200))
        if max_iters < 50:
            max_iters = 50

        best_sol, best_cost = tabu_search(init, rng, max_iters, tabu_tenure=12, aspiration=True)

        if best_sol is not None and best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best = [list(r) for r in best_sol]

    # Final polish: run a short 2-opt + relocate on the best solution
    if global_best is not None and time.time() < DEADLINE - 3:
        polish_iters = int((DEADLINE - 3 - (time.time() - t0)) * 300)
        polish_iters = max(polish_iters, 100)
        polished, polished_cost = tabu_search(global_best, rng, polish_iters, tabu_tenure=8, aspiration=True)
        if polished is not None and polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best = [list(r) for r in polished]

    # Final validation
    if global_best is None:
        global_best = [[c] for c in customers]

    final_val, final_dist = evaluate_routes(global_best)
    if not final_val:
        global_best = [[c] for c in customers]

    return global_best
