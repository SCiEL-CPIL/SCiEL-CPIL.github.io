# EVOLVE-BLOCK-START
import numpy as np
import time

def _total_cost(result, demands, capacities, open_cost, assign_cost):
    used = np.unique(result)
    open_sum = float(open_cost[used].sum())
    assign_sum = float(assign_cost[np.arange(len(demands)), result].sum())
    return open_sum + assign_sum

def _is_valid(result, demands, capacities):
    load = np.zeros(len(capacities), dtype=float)
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    return np.all(load <= capacities + 1e-9)

def _greedy_init(demands, capacities, open_cost, assign_cost):
    n = len(demands); m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m, dtype=float)
    for i in np.argsort(-demands):
        best_f, best_c = -1, np.inf
        for f in range(m):
            if load[f] + demands[i] <= capacities[f]:
                c = assign_cost[i, f] + (open_cost[f] * 0.01 if load[f] == 0 else 0.0)
                if c < best_c:
                    best_c = c; best_f = f
        if best_f < 0:
            best_f = int(np.argmin(load))
        result[i] = best_f
        load[best_f] += demands[i]
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit):
    n = len(demands); m = len(capacities)
    result = result.copy()
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    start = time.time()
    while time.time() - start < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start >= time_limit:
                break
            cur_f = result[i]; d = demands[i]
            feasible = (load + d) <= capacities
            feasible[cur_f] = False
            if not np.any(feasible):
                continue
            feas_idx = np.where(feasible)[0]
            d_assign = assign_cost[i, feas_idx] - assign_cost[i, cur_f]
            d_open_t = np.where(load[feas_idx] == 0, open_cost[feas_idx], 0.0)
            d_open_s = -open_cost[cur_f] if (load[cur_f] - d) == 0 else 0.0
            total_delta = d_assign + d_open_t + d_open_s
            min_idx = np.argmin(total_delta)
            if total_delta[min_idx] < -1e-9:
                best_f = feas_idx[min_idx]
                load[cur_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += total_delta[min_idx]
                improved = True
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_result = result.copy()
        if not improved:
            break
    return best_result, best_cost

def _solve_mcf(demands, capacities, open_cost, assign_cost):
    """Solve CFLP as a min-cost flow problem (exact)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_cost_flow
    n = len(demands); m = len(capacities)
    # Nodes: 0=source, 1..n=customers, n+1..n+m=facilities, n+m+1=sink
    src = 0
    cust_start = 1
    fac_start = 1 + n
    sink = 1 + n + m
    N = sink + 1
    rows, cols, caps, costs = [], [], [], []
    # Source -> customer: demand[i] units, cost 0
    for i in range(n):
        rows.append(src); cols.append(cust_start + i)
        caps.append(int(demands[i])); costs.append(0)
    # Customer -> facility: 1 unit, cost assign_cost[i,f]
    for i in range(n):
        for f in range(m):
            rows.append(cust_start + i); cols.append(fac_start + f)
            caps.append(1); costs.append(int(round(assign_cost[i, f])))
    # Facility -> sink: capacity, cost open_cost (one-time)
    for f in range(m):
        rows.append(fac_start + f); cols.append(sink)
        caps.append(int(capacities[f])); costs.append(int(round(open_cost[f])))
    graph = coo_matrix((costs, (rows, cols)), shape=(N, N))
    try:
        flow, cost = min_cost_flow(graph, capacity=caps, demand=np.zeros(N, dtype=int))
        # Extract assignment: for each customer, find facility with flow 1
        result = np.zeros(n, dtype=int)
        for i in range(n):
            cu = cust_start + i
            found = False
            for f in range(m):
                fu = fac_start + f
                if flow[cu, fu] >= 1:
                    result[i] = f
                    found = True
                    break
            if not found:
                # Fallback: assign to nearest feasible
                for f in range(m):
                    if flow[cu, fac_start + f] >= 0.5:
                        result[i] = f; found = True; break
                if not found:
                    result[i] = int(np.argmin(assign_cost[i, :]))
        return result, cost
    except Exception:
        return None, None

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n = len(demands); m = len(capacities)
    global_start = time.time()
    deadline = global_start + 130.0
    best_result = None
    best_cost = np.inf

    # Phase 1: Exact min-cost flow
    mcf_result, mcf_cost = _solve_mcf(demands, capacities, open_cost, assign_cost)
    if mcf_result is not None and _is_valid(mcf_result, demands, capacities):
        mcf_cost = _total_cost(mcf_result, demands, capacities, open_cost, assign_cost)
        if mcf_cost < best_cost:
            best_cost = mcf_cost
            best_result = mcf_result.copy()

    # Phase 2: Local search on best (or greedy fallback)
    if time.time() < deadline - 2.0:
        start = best_result if best_result is not None else _greedy_init(demands, capacities, open_cost, assign_cost)
        if not _is_valid(start, demands, capacities):
            start = _greedy_init(demands, capacities, open_cost, assign_cost)
        ls_time = min(60.0, deadline - time.time() - 1.0)
        if ls_time > 0.5:
            ls_result, ls_cost = _local_search(start, demands, capacities, open_cost, assign_cost, ls_time)
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_result = ls_result.copy()

    # Fallback
    if best_result is None or not _is_valid(best_result, demands, capacities):
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)

    return best_result.astype(int)
# EVOLVE-BLOCK-END
