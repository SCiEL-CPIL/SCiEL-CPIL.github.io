"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _solve_instance(values, seed, deadline):
    """Solve one instance: find max sum-free subset using multi-start local search."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    val_set = set(values)
    
    # Build adjacency: adj[i] = list of j where values[i]+values[j] in val_set
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i].append(j)
                adj[j].append(i)
    
    # O(1) can_add: check if vertex v can be added to current set
    def can_add(v, in_set):
        for nb in adj[v]:
            if in_set[nb]:
                return False
        return True
    
    # Greedy initialization: pick vertices by minimum conflict degree
    def greedy_init():
        in_set = [False] * n
        remaining = list(range(n))
        # Sort by initial degree
        remaining.sort(key=lambda v: len(adj[v]))
        for v in remaining:
            if can_add(v, in_set):
                in_set[v] = True
        return in_set
    
    # 2-swap local search: remove two in-set, add two out-of-set
    def local_search_2swap(cur, time_limit):
        cur = cur[:]
        cur_size = sum(cur)
        in_list = [i for i in range(n) if cur[i]]
        out_list = [i for i in range(n) if not cur[i]]
        best = cur[:]
        best_size = cur_size
        
        start = time.time()
        no_improve = 0
        
        while time.time() - start < time_limit:
            if not in_list or len(out_list) < 2:
                # Try single add
                for v in out_list[:]:
                    if can_add(v, cur):
                        cur[v] = True
                        cur_size += 1
                        in_list.append(v)
                        out_list.remove(v)
                        if cur_size > best_size:
                            best_size = cur_size
                            best = cur[:]
                break
            
            improved = False
            
            # Try 2-swap: remove a, b from in; add c, d from out
            # Sample candidates
            n_in = len(in_list)
            n_out = len(out_list)
            
            # Try single swap first (remove one, add one)
            in_sample = rng.sample(in_list, min(30, n_in))
            out_sample = rng.sample(out_list, min(30, n_out))
            
            for v_in in in_sample:
                if not cur[v_in]:
                    continue
                for v_out in out_sample:
                    if cur[v_out]:
                        continue
                    # Check if v_out can be added when v_in is removed
                    ok = True
                    for nb in adj[v_out]:
                        if cur[nb] and nb != v_in:
                            ok = False
                            break
                    if ok:
                        cur[v_in] = False
                        cur[v_out] = True
                        in_list.remove(v_in)
                        in_list.append(v_out)
                        out_list.remove(v_out)
                        out_list.append(v_in)
                        improved = True
                        if cur_size > best_size:
                            best_size = cur_size
                            best = cur[:]
                        break
                if improved:
                    break
            
            if not improved:
                # Try 2-swap
                in_sample = rng.sample(in_list, min(20, n_in))
                out_sample = rng.sample(out_list, min(20, n_out))
                
                for a in in_sample:
                    if not cur[a]:
                        continue
                    for b in in_sample:
                        if b == a or not cur[b]:
                            continue
                        for c in out_sample:
                            if cur[c]:
                                continue
                            for d in out_sample:
                                if d == c or cur[d]:
                                    continue
                                # Check if c and d can both be added when a, b removed
                                ok = True
                                for nb in adj[c]:
                                    if cur[nb] and nb != a and nb != b:
                                        ok = False
                                        break
                                if ok:
                                    for nb in adj[d]:
                                        if cur[nb] and nb != a and nb != b:
                                            ok = False
                                            break
                                if ok:
                                    cur[a] = False
                                    cur[b] = False
                                    cur[c] = True
                                    cur[d] = True
                                    in_list.remove(a)
                                    in_list.remove(b)
                                    in_list.append(c)
                                    in_list.append(d)
                                    out_list.remove(c)
                                    out_list.remove(d)
                                    out_list.append(a)
                                    out_list.append(b)
                                    improved = True
                                    if cur_size > best_size:
                                        best_size = cur_size
                                        best = cur[:]
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    break
        
        # Final add-only polish
        changed = True
        while changed and time.time() - start < time_limit:
            changed = False
            for v in out_list[:]:
                if cur[v]:
                    continue
                if can_add(v, cur):
                    cur[v] = True
                    cur_size += 1
                    in_list.append(v)
                    out_list.remove(v)
                    changed = True
                    if cur_size > best_size:
                        best_size = cur_size
                        best = cur[:]
        
        return best
    
    # Simulated annealing with proper conflict tracking
    def simulated_annealing(initial, t_start, t_end, time_limit):
        cur = initial[:]
        cur_size = sum(cur)
        
        # Count conflicts
        def count_conflicts(s):
            cnt = 0
            for i in range(n):
                if s[i]:
                    for j in adj[i]:
                        if j > i and s[j]:
                            cnt += 1
            return cnt
        
        cur_conf = count_conflicts(cur)
        best = cur[:]
        best_obj = cur_size - 10000 * cur_conf
        
        start = time.time()
        it = 0
        
        while time.time() - start < time_limit:
            it += 1
            frac = min(it / 50000.0, 1.0)
            t = t_start * (t_end / t_start) ** frac
            t = max(t, 1e-10)
            
            v = rng.randint(0, n - 1)
            old = cur[v]
            cur[v] = not old
            
            if old:
                # Removing v
                removed = sum(1 for nb in adj[v] if cur[nb])
                new_size = cur_size - 1
                new_conf = max(0, cur_conf - removed)
                delta = (new_size - 10000 * new_conf) - (cur_size - 10000 * cur_conf)
            else:
                # Adding v
                added = sum(1 for nb in adj[v] if cur[nb])
                new_size = cur_size + 1
                new_conf = cur_conf + added
                delta = (new_size - 10000 * new_conf) - (cur_size - 10000 * cur_conf)
            
            if delta >= 0 or rng.random() < np.exp(delta / t):
                cur_size += (1 if not old else -1)
                if old:
                    removed = sum(1 for nb in adj[v] if cur[nb])
                    cur_conf = max(0, cur_conf - removed)
                else:
                    added = sum(1 for nb in adj[v] if cur[nb])
                    cur_conf += added
                
                obj = cur_size - 10000 * cur_conf
                if obj > best_obj:
                    best_obj = obj
                    best = cur[:]
            else:
                cur[v] = old
        
        return best
    
    # Multi-start local search
    best_result = None
    best_size = 0
    
    # Start with odds (always sum-free)
    odds = [values[i] % 2 != 0 for i in range(n)]
    odds_size = sum(odds)
    if odds_size > best_size:
        best_size = odds_size
        best_result = odds[:]
    
    # Try multiple greedy initializations with different orderings
    num_starts = 3
    total_time = deadline - time.time()
    time_per_start = total_time / (num_starts + 1)  # +1 for SA
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 2:
            break
        
        # Different greedy orderings
        if start_idx == 0:
            init = greedy_init()
        elif start_idx == 1:
            # Random ordering greedy
            init = [False] * n
            order = list(range(n))
            rng.shuffle(order)
            for v in order:
                if can_add(v, init):
                    init[v] = True
        else:
            # Start from odds and improve
            init = odds[:]
        
        init_size = sum(init)
        
        # Local search
        if time.time() < deadline - 3:
            ls_time = min(time_per_start * 0.6, deadline - time.time() - 3)
            ls_result = local_search_2swap(init, ls_time)
            ls_size = sum(ls_result)
            if ls_size > best_size:
                best_size = ls_size
                best_result = ls_result[:]
        
        # Simulated annealing from best found so far
        if time.time() < deadline - 2:
            sa_time = min(time_per_start * 0.4, deadline - time.time() - 2)
            sa_result = simulated_annealing(best_result, 5.0, 0.01, sa_time)
            sa_size = sum(sa_result)
            if sa_size > best_size:
                best_size = sa_size
                best_result = sa_result[:]
    
    # Final polish
    if time.time() < deadline - 1 and best_result is not None:
        polish_time = min(deadline - time.time() - 0.5, 5.0)
        polish_result = local_search_2swap(best_result, polish_time)
        p_size = sum(polish_result)
        if p_size > best_size:
            best_size = p_size
            best_result = polish_result[:]
    
    if best_result is None:
        return []
    
    return [values[i] for i in range(n) if best_result[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    results = []
    # Allocate time proportionally to instance size
    sizes = [60, 120, 240]
    total_size = sum(sizes)
    
    for i, values in enumerate(instances):
        if time.time() > deadline - 2:
            results.append([values[0]] if values else [])
            continue
        
        # Time proportional to n^1.5 (search space grows)
        weight = (len(values) ** 1.5) / sum(s ** 1.5 for s in sizes)
        time_budget = min(weight * 140.0, deadline - time.time() - 2)
        
        inst_deadline = min(time.time() + time_budget, deadline - 2)
        subset = _solve_instance(values, 801 + i, inst_deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
