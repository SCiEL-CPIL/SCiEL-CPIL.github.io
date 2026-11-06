import time
import random

def _elapsed(start):
    return time.time() - start

def _has_k4(adj, n):
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                c = common
                while c:
                    lsb = c & (-c)
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if (adj[w] & common) & (common >> w):
                        return True
    return False

def _has_ind6(adj, n):
    nonadj = [0]*n
    for v in range(n):
        m = 0
        for u in range(n):
            if u != v and not ((adj[v] >> u) & 1):
                m |= (1 << u)
        nonadj[v] = m
    order = sorted(range(n), key=lambda v: (nonadj[v].bit_count(), v))
    def dfs(start_idx, chosen, forbidden_mask):
        if len(chosen) == 6:
            return True
        rem = len(order) - start_idx
        if rem < 6 - len(chosen):
            return False
        for i in range(start_idx, len(order)):
            v = order[i]
            if (forbidden_mask >> v) & 1:
                continue
            chosen.append(v)
            if dfs(i+1, chosen, forbidden_mask | adj[v]):
                return True
            chosen.pop()
        return False
    return dfs(0, [], 0)

def _is_valid(adj, n):
    return (not _has_k4(adj, n)) and (not _has_ind6(adj, n))

def _k4_count(adj, n):
    count = 0
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                c = common
                while c:
                    lsb = c & (-c)
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if (adj[w] & common) & (common >> w):
                        count += 1
    return count

def _ind6_count(adj, n):
    nonadj = [0]*n
    for v in range(n):
        m = 0
        for u in range(n):
            if u != v and not ((adj[v] >> u) & 1):
                m |= (1 << u)
        nonadj[v] = m
    order = sorted(range(n), key=lambda v: (nonadj[v].bit_count(), v))
    count = 0
    def dfs(start_idx, chosen, forbidden_mask):
        nonlocal count
        if len(chosen) == 6:
            count += 1
            return
        rem = len(order) - start_idx
        if rem < 6 - len(chosen):
            return
        for i in range(start_idx, len(order)):
            v = order[i]
            if (forbidden_mask >> v) & 1:
                continue
            chosen.append(v)
            dfs(i+1, chosen, forbidden_mask | adj[v])
            chosen.pop()
    dfs(0, [], 0)
    return count

def _paley17():
    n = 17
    adj = [0]*n
    squares = {x*x % 17 for x in range(1, 17)}
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, n

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _local_search_random(n, deadline, start_time, p=0.5):
    # Random initialization, simulated annealing with weighted objective
    w_k4 = 100
    w_i6 = 10
    adj = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    def objective(a):
        return w_k4 * _k4_count(a, n) + w_i6 * _ind6_count(a, n)
    cur_obj = objective(adj)
    if cur_obj == 0:
        return adj
    temp = 80.0
    cooling = 0.995
    min_temp = 0.05
    steps = 0
    max_steps = 30000
    best_adj = adj[:]
    best_obj = cur_obj
    while steps < max_steps:
        if _elapsed(start_time) > deadline:
            break
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:
            continue
        had = (adj[u] >> v) & 1
        if had:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        new_obj = objective(adj)
        delta = new_obj - cur_obj
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-9)):
            cur_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_adj = adj[:]
            if new_obj == 0:
                return best_adj
        else:
            if had:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        steps += 1
    return best_adj if best_obj == 0 else None

def _local_search_from_seed(seed_adj, seed_n, n, deadline, start_time):
    # Initialize by embedding seed in first seed_n vertices, random for rest
    w_k4 = 100
    w_i6 = 10
    adj = [0]*n
    # Copy seed
    for i in range(seed_n):
        adj[i] = seed_adj[i] & ((1 << seed_n) - 1)
    # Random edges between seed and new, and among new
    for i in range(seed_n):
        for j in range(seed_n, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    for i in range(seed_n, n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    def objective(a):
        return w_k4 * _k4_count(a, n) + w_i6 * _ind6_count(a, n)
    cur_obj = objective(adj)
    if cur_obj == 0:
        return adj
    temp = 80.0
    cooling = 0.995
    min_temp = 0.05
    steps = 0
    max_steps = 30000
    best_adj = adj[:]
    best_obj = cur_obj
    while steps < max_steps:
        if _elapsed(start_time) > deadline:
            break
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:
            continue
        had = (adj[u] >> v) & 1
        if had:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        new_obj = objective(adj)
        delta = new_obj - cur_obj
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-9)):
            cur_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_adj = adj[:]
            if new_obj == 0:
                return best_adj
        else:
            if had:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        steps += 1
    return best_adj if best_obj == 0 else None

def construct():
    start_time = time.time()
    deadline = 135.0
    seed_adj, seed_n = _paley17()
    if not _is_valid(seed_adj, seed_n):
        return {'n': 1, 'edges': []}
    best = {'n': seed_n, 'edges': _edges_from_adj(seed_adj, seed_n)}
    # Try to find valid graphs for larger n, from 35 down to seed_n+1
    for target in range(35, seed_n, -1):
        if _elapsed(start_time) > deadline - 3:
            break
        found = None
        # Try seed-based first, then random
        for attempt in range(6):
            if _elapsed(start_time) > deadline - 1:
                break
            if attempt < 3:
                result = _local_search_from_seed(seed_adj, seed_n, target, deadline - 1, start_time)
            else:
                result = _local_search_random(target, deadline - 1, start_time)
            if result is not None and _is_valid(result, target):
                found = result
                break
        if found is not None:
            best = {'n': target, 'edges': _edges_from_adj(found, target)}
            break  # Found the largest, stop
    return best
