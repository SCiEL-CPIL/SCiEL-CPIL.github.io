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

def _build_conflict_graph(values):
    """Build adjacency list for the conflict graph.
    Edge (i,j) exists if values[i]+values[j] is also in values.
    Returns adj as list of lists (sorted) for cache efficiency.
    """
    val_set = set(values)
    n = len(values)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i].append(j)
                adj[j].append(i)
    # Convert to sorted lists for deterministic iteration
    for i in range(n):
        adj[i].sort()
    return adj

def _greedy_mis(adj, n, rng, deadline):
    """Fast greedy MIS: pick min-degree remaining vertex, remove its neighbors."""
    in_set = [False] * n
    remaining = [True] * n
    rem_count = n
    # Maintain current degree counts
    cur_deg = [len(adj[i]) for i in range(n)]
    
    while rem_count > 0:
        if time.time() > deadline:
            break
        # Find vertex with minimum current degree among remaining
        best_v = -1
        best_deg = float('inf')
        for v in range(n):
            if not remaining[v]:
                continue
            d = cur_deg[v]
            if d < best_deg:
                best_deg = d
                best_v = v
                if d == 0:
                    break
        if best_v < 0:
            break
        in_set[best_v] = True
        remaining[best_v] = False
        rem_count -= 1
        for nb in adj[best_v]:
            if remaining[nb]:
                cur_deg[best_v] -= 1
                remaining[nb] = False
                rem_count -= 1
                for nb2 in adj[nb]:
                    if remaining[nb2]:
                        cur_deg[nb2] -= 1
    
    return in_set

def _conflict_count(in_set, adj, n):
    """Count number of conflicting edges within in_set."""
    cnt = 0
    for i in range(n):
        if in_set[i]:
            for j in adj[i]:
                if j > i and in_set[j]:
                    cnt += 1
    return cnt

def _local_search_swap(in_set, adj, n, rng, time_limit):
    """Local search: 1-swap, 2-swap, and add-only moves with O(degree) checks."""
    cur = in_set[:]
    cur_size = sum(cur)
    best = cur[:]
    best_size = cur_size
    
    in_list = [i for i in range(n) if cur[i]]
    out_list = [i for i in range(n) if not cur[i]]
    
    start = time.time()
    no_improve = 0
    
    def try_add(v):
        """Try to add vertex v (must be in out_list). Returns True if added."""
        for nb in adj[v]:
            if cur[nb]:
                return False
        cur[v] = True
        in_list.append(v)
        out_list.remove(v)
        return True
    
    def try_remove(v):
        """Remove vertex v from in_set."""
        cur[v] = False
        in_list.remove(v)
        out_list.append(v)
    
    def try_swap(v_in, v_out):
        """Try to swap: remove v_in, add v_out. Returns True if successful."""
        # Check v_out conflicts with any in-set member except v_in
        for nb in adj[v_out]:
            if cur[nb] and nb != v_in:
                return False
        try_remove(v_in)
        try_add(v_out)
        return True
    
    while time.time() - start < time_limit:
        if not in_list or not out_list:
            break
        
        improved = False
        
        # 1-swap: try to replace one in-set vertex with one out-set vertex
        n_in = len(in_list)
        n_out = len(out_list)
        in_sample = rng.sample(in_list, min(80, n_in))
        out_sample = rng.sample(out_list, min(50, n_out))
        
        for v_in in in_sample:
            if not cur[v_in]:
                continue
            for v_out in out_sample:
                if cur[v_out]:
                    continue
                if try_swap(v_in, v_out):
                    improved = True
                    break
            if improved:
                break
        
        # 2-swap: remove two in-set vertices, add two out-set vertices
        if not improved and len(in_list) >= 2 and len(out_list) >= 2:
            in_sample2 = rng.sample(in_list, min(30, len(in_list)))
            out_sample2 = rng.sample(out_list, min(20, len(out_list)))
            for i1 in range(len(in_sample2)):
                v_in1 = in_sample2[i1]
                if not cur[v_in1]:
                    continue
                for i2 in range(i1 + 1, len(in_sample2)):
                    v_in2 = in_sample2[i2]
                    if not cur[v_in2]:
                        continue
                    for j1 in range(len(out_sample2)):
                        v_out1 = out_sample2[j1]
                        if cur[v_out1]:
                            continue
                        for j2 in range(j1 + 1, len(out_sample2)):
                            v_out2 = out_sample2[j2]
                            if cur[v_out2]:
                                continue
                            # Check both out vertices don't conflict with in-set minus the two removed
                            ok = True
                            for nb in adj[v_out1]:
                                if cur[nb] and nb != v_in1 and nb != v_in2:
                                    ok = False
                                    break
                            if ok:
                                for nb in adj[v_out2]:
                                    if cur[nb] and nb != v_in1 and nb != v_in2:
                                        ok = False
                                        break
                            if ok:
                                try_remove(v_in1)
                                try_remove(v_in2)
                                try_add(v_out1)
                                try_add(v_out2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Add-only: try to add any out vertex with no conflicts
        if not improved:
            added = False
            for _ in range(min(30, len(out_list))):
                v_try = rng.choice(out_list)
                if not cur[v_try] and try_add(v_try):
                    added = True
                    break
            if added:
                improved = True
        
        if improved:
            no_improve = 0
            cur_size = sum(cur)
            if cur_size > best_size:
                best_size = cur_size
                best = cur[:]
        else:
            no_improve += 1
            if no_improve > 300:
                break
    
    # Final add-only polish
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        for v in out_list[:]:
            if not cur[v] and try_add(v):
                changed = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best = cur[:]
    
    return best

def _simulated_annealing(adj, n, rng, t_start, t_end, time_limit, initial=None):
    """Simulated annealing: accept moves based on size change, penalize conflicts."""
    if initial is not None:
        cur = initial[:]
    else:
        cur = [False] * n
        for v in range(n):
            if not any(cur[nb] for nb in adj[v]):
                cur[v] = True
    
    cur_size = sum(cur)
    cur_conf = _conflict_count(cur, adj, n)
    
    best = cur[:]
    best_obj = cur_size - 10000 * cur_conf
    
    start = time.time()
    it = 0
    max_it = 100000
    
    while time.time() - start < time_limit:
        it += 1
        t = t_start * (t_end / t_start) ** (min(it, max_it) / max_it)
        t = max(t, 1e-10)
        
        v = rng.randint(0, n - 1)
        old_state = cur[v]
        
        if old_state:
            # Removing v
            removed_conf = sum(1 for nb in adj[v] if cur[nb])
            new_size = cur_size - 1
            new_conf = max(0, cur_conf - removed_conf)
            delta_obj = (new_size - 10000 * new_conf) - (cur_size - 10000 * cur_conf)
        else:
            # Adding v
            added_conf = sum(1 for nb in adj[v] if cur[nb])
            new_size = cur_size + 1
            new_conf = cur_conf + added_conf
            delta_obj = (new_size - 10000 * new_conf) - (cur_size - 10000 * cur_conf)
        
        if delta_obj >= 0 or rng.random() < np.exp(delta_obj / t):
            cur[v] = not old_state
            cur_size += (1 if not old_state else -1)
            if old_state:
                cur_conf = max(0, cur_conf - removed_conf)
            else:
                cur_conf += added_conf
            
            obj = cur_size - 10000 * cur_conf
            if obj > best_obj:
                best_obj = obj
                best = cur[:]
        else:
            # Revert
            cur[v] = old_state
    
    return best

def _solve_instance(values, seed, deadline):
    """Solve one instance: find max sum-free subset."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    adj = _build_conflict_graph(values)
    
    best_in_set = None
    best_size = 0
    
    # Baseline: odds and evens
    odds_mask = [values[i] % 2 != 0 for i in range(n)]
    evens_mask = [values[i] % 2 == 0 for i in range(n)]
    if sum(odds_mask) > best_size:
        best_size = sum(odds_mask)
        best_in_set = odds_mask[:]
    if sum(evens_mask) > best_size:
        best_size = sum(evens_mask)
        best_in_set = evens_mask[:]
    
    # Multiple greedy MIS with different orderings
    if time.time() < deadline - 5:
        # Standard min-degree greedy
        greedy = _greedy_mis(adj, n, rng, deadline - 2)
        gsize = sum(greedy)
        if gsize > best_size:
            best_size = gsize
            best_in_set = greedy[:]
        
        # Max-degree greedy (pick highest degree first)
        in_set2 = [False] * n
        remaining2 = [True] * n
        rem_count2 = n
        cur_deg2 = [len(adj[i]) for i in range(n)]
        while rem_count2 > 0 and time.time() < deadline - 2:
            best_v = -1
            best_deg = -1
            for v in range(n):
                if not remaining2[v]:
                    continue
                if cur_deg2[v] > best_deg:
                    best_deg = cur_deg2[v]
                    best_v = v
            if best_v < 0:
                break
            in_set2[best_v] = True
            remaining2[best_v] = False
            rem_count2 -= 1
            for nb in adj[best_v]:
                if remaining2[nb]:
                    cur_deg2[best_v] -= 1
                    remaining2[nb] = False
                    rem_count2 -= 1
                    for nb2 in adj[nb]:
                        if remaining2[nb2]:
                            cur_deg2[nb2] -= 1
        gsize2 = sum(in_set2)
        if gsize2 > best_size:
            best_size = gsize2
            best_in_set = in_set2[:]
    
    # Local search with 1-swap and 2-swap
    if time.time() < deadline - 3 and best_in_set is not None:
        ls_time = min(deadline - time.time() - 2, 25.0)
        ls_result = _local_search_swap(best_in_set, adj, n, rng, ls_time)
        ls_size = sum(ls_result)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result[:]
    
    # Simulated annealing
    if time.time() < deadline - 2 and best_in_set is not None:
        sa_time = min(deadline - time.time() - 1, 15.0)
        sa_result = _simulated_annealing(adj, n, rng, 10.0, 0.01, sa_time, initial=best_in_set)
        sa_size = sum(sa_result)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result[:]
    
    # Final polish with local search
    if time.time() < deadline - 1 and best_in_set is not None:
        polish_time = min(deadline - time.time() - 0.5, 5.0)
        polish_result = _local_search_swap(best_in_set, adj, n, rng, polish_time)
        p_size = sum(polish_result)
        if p_size > best_size:
            best_size = p_size
            best_in_set = polish_result[:]
    
    if best_in_set is None:
        return []
    
    return [values[i] for i in range(n) if best_in_set[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline with 5s safety margin
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    results = []
    for i, values in enumerate(instances):
        # Allocate time: give more to larger instances
        # Total budget ~140s, split: 20% for n=60, 30% for n=120, 50% for n=240
        weights = [0.2, 0.3, 0.5]
        total_budget = 140.0
        time_budget = total_budget * weights[i]
        
        inst_deadline = min(start_time + time_budget + 10, deadline - 2)
        subset = _solve_instance(values, 801 + i, inst_deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
