"""EV routing: cluster + greedy nearest-insertion with battery/time feasibility."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def plan(instance):
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin vs 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    n = len(coords)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_cust + 1))

    # Precompute distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))

    def time_budget_left(t, node):
        # remaining travel time from node back to depot must fit within horizon
        return horizon - t - D[node, 0] / speed

    def simulate(route):
        """Return (ok, time, battery, dist) for a route list (no depot at ends)."""
        t = 0.0
        b = battery_max
        dist = 0.0
        pos = 0
        for node in route:
            d = D[pos, node]
            if d > b + 1e-9:
                return False, t, b, dist
            b -= d
            t += d / speed
            dist += d
            pos = node
            if node in stations:
                # recharge: time = (battery - energy) * recharge_rate ... refills
                # energy spent to reach station already subtracted; refill to full
                t += (battery_max - b) * recharge_rate
                b = battery_max
            else:
                # customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False, t, b, dist
                t += service[node]
                b -= service[node]
                if b < -1e-9:
                    return False, t, b, dist
        # return to depot
        d = D[pos, 0]
        if d > b + 1e-9:
            return False, t, b, dist
        b -= d
        t += d / speed
        dist += d
        if t > horizon + 1e-9:
            return False, t, b, dist
        return True, t, b, dist

    def feasible(route):
        ok, t, b, dist = simulate(route)
        return ok

    # Greedy build: order customers by due time, insert each into the best route
    # (nearest feasible position), else start a new route.
    order = sorted(customers, key=lambda v: (due[v], v))

    routes = []
    loads = []

    for cust in order:
        if time.time() > deadline:
            break
        dem = demands[cust]
        best = None  # (score, ridx, pos)
        for ridx, route in enumerate(routes):
            if loads[ridx] + dem > capacity:
                continue
            # try inserting at each position
            for pos in range(len(route) + 1):
                cand = route[:pos] + [cust] + route[pos:]
                if feasible(cand):
                    # score by added distance
                    added = (D[route[pos - 1] if pos > 0 else 0, cust] +
                             D[cust, route[pos] if pos < len(route) else 0] -
                             (D[route[pos - 1] if pos > 0 else 0, route[pos] if pos < len(route) else 0]
                              if 0 < pos < len(route) else 0))
                    sc = added
                    if best is None or sc < best[0]:
                        best = (sc, ridx, pos)
        if best is not None:
            _, ridx, pos = best
            routes[ridx] = routes[ridx][:pos] + [cust] + routes[ridx][pos:]
            loads[ridx] += dem
        else:
            # start new route
            if feasible([cust]):
                routes.append([cust])
                loads.append(dem)
            else:
                # cannot serve; leave unserved (should not happen)
                pass

    # Ensure all customers served; fallback to single-customer routes for any missing
    served = set()
    for r in routes:
        served.update(r)
    for cust in customers:
        if cust not in served:
            routes.append([cust])

    return routes
# EVOLVE-BLOCK-END
