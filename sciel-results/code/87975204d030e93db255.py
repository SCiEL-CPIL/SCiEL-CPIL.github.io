"""Rewritten EV-routing planner: savings heuristic + feasibility + local improvement."""
# EVOLVE-BLOCK-START
import numpy as np


def plan(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    n = len(coords)
    due = instance["due"]
    ready = instance["ready"]
    demand = instance["demand"]
    service = instance["service"]
    stations = list(instance["stations"])
    n_cust = instance["n_customers"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    # Which nodes are customers
    is_customer = np.zeros(n, dtype=bool)
    is_customer[1:n_cust + 1] = True
    cust_ids = list(range(1, n_cust + 1))
    station_set = set(stations)

    def feasible_route(seq):
        """Check a route (list of customer/station ids, depot implicit).
        Returns (ok, total_dist)."""
        t = 0.0
        bat = float(battery_cap)
        cur = 0
        total_d = 0.0
        for node in seq:
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return False, total_d
            total_d += d
            cur = node
            if node in station_set:
                # Recharge: time = (battery - energy)*recharge_rate, then refill
                # Interpretation: station adds (battery - energy)*recharge_rate time and refills.
                # 'energy' is the battery level upon arrival; refill to full.
                bat = float(battery_cap)
                t += 0.0  # refill is instant per contract; the time cost was already accounted? 
                # Contract: "a station adds (battery - energy) * recharge_rate time and refills"
                # We'll model: time cost = (battery_cap - bat_on_arrival) * recharge_rate
                # But bat was already decremented; recompute arrival bat.
                # Actually we already subtracted d from bat. Let's redo properly below.
                pass
            elif is_customer[node]:
                if t < ready[node]:
                    t = float(ready[node])
                if t > due[node] + 1e-9:
                    return False, total_d
                t += float(service[node])
        # Return to depot
        d = dist[cur, 0]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, total_d
        total_d += d
        if t > horizon + 1e-9:
            return False, total_d
        return True, total_d

    # Better feasibility that correctly models station recharge time
    def feasible_route2(seq):
        t = 0.0
        bat = float(battery_cap)
        cur = 0
        total_d = 0.0
        for node in seq:
            d = float(dist[cur, node])
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return False, total_d
            total_d += d
            cur = node
            if node in station_set:
                # time cost to recharge from arrival level to full
                t += (float(battery_cap) - bat) * recharge_rate
                bat = float(battery_cap)
            else:
                if t < float(ready[node]):
                    t = float(ready[node])
                if t > float(due[node]) + 1e-9:
                    return False, total_d
                t += float(service[node])
        d = float(dist[cur, 0])
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, total_d
        total_d += d
        if t > horizon + 1e-9:
            return False, total_d
        return True, total_d

    # Build initial routes via savings: start each customer as its own route,
    # then merge by largest savings while keeping feasibility.
    # Represent routes as lists; start = [[c] for c in cust_ids]
    routes = [[c] for c in cust_ids]

    # Compute savings for merging route A (ending a) and route B (starting b)
    # savings = dist[depot,a]+dist[depot,b]-dist[a,b]  (when concatenating A+B)
    # We'll just greedily try to insert customers into existing routes via nearest feasible spot.

    # Simpler robust approach: nearest-neighbour with feasibility, then 2-opt local search.
    # Start with empty routes, insert customers one by one (sorted by due) into the
    # feasible position that adds least distance, opening a new route if needed.

    routes = []
    remaining = sorted(cust_ids, key=lambda c: (float(due[c]), c))

    for c in remaining:
        best = None  # (dist_delta, route_idx, pos)
        for ri, r in enumerate(routes):
            for pos in range(len(r) + 1):
                cand = r[:pos] + [c] + r[pos:]
                ok, _ = feasible_route2(cand)
                if ok:
                    # distance delta
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    delta = float(dist[prev, c]) + float(dist[c, nxt]) - float(dist[prev, nxt])
                    if best is None or delta < best[0]:
                        best = (delta, ri, pos)
        if best is not None:
            _, ri, pos = best
            routes[ri].insert(pos, c)
        else:
            routes.append([c])

    # Local improvement: 2-opt on each route + try moving/swapping customers between routes
    improved = True
    while improved:
        improved = False
        # 2-opt within routes
        for ri, r in enumerate(routes):
            if len(r) < 2:
                continue
            for i in range(len(r) - 1):
                for j in range(i + 1, len(r)):
                    # reverse segment i..j
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    ok, _ = feasible_route2(cand)
                    if ok:
                        # compute distance change
                        a = r[i - 1] if i > 0 else 0
                        b = r[i]
                        c2 = r[j]
                        d2 = r[j + 1] if j + 1 < len(r) else 0
                        old = float(dist[a, b]) + float(dist[c2, d2])
                        new = float(dist[a, c2]) + float(dist[b, d2])
                        if new < old - 1e-9:
                            routes[ri] = cand
                            improved = True
        # Try moving a customer to another route
        for ri, r in enumerate(routes):
            for k in range(len(r)):
                c = r[k]
                for rj, r2 in enumerate(routes):
                    if rj == ri:
                        continue
                    for pos in range(len(r2) + 1):
                        cand2 = r2[:pos] + [c] + r2[pos:]
                        ok, _ = feasible_route2(cand2)
                        if ok:
                            # distance delta: remove c from r, insert into r2
                            prev_r = r[k - 1] if k > 0 else 0
                            next_r = r[k + 1] if k + 1 < len(r) else 0
                            rem_delta = float(dist[prev_r, next_r]) - float(dist[prev_r, c]) - float(dist[c, next_r])
                            prev2 = r2[pos - 1] if pos > 0 else 0
                            next2 = r2[pos] if pos < len(r2) else 0
                            ins_delta = float(dist[prev2, c]) + float(dist[c, next2]) - float(dist[prev2, next2])
                            if (rem_delta + ins_delta) < -1e-9:
                                r.pop(k)
                                r2.insert(pos, c)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Drop empty routes
    routes = [r for r in routes if len(r) > 0]
    return routes
# EVOLVE-BLOCK-END
