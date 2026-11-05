"""Flow-shop permutation via multi-start heuristics + fast insertion local search + simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """
    Fast makespan calculation using pure Python lists.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0:
        return 0.0
    if n_machines == 0:
        return 0.0

    prev = [0.0] * n_machines

    for job in perm:
        row = pt_list[job]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_list = pt.tolist()
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's dispatching rule."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _spt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _bottleneck_initial(pt):
    """Sort by total processing on the machine with largest total load, descending."""
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _mean_initial(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first_machine_initial(pt):
    return list(np.argsort(-pt[:, 0]))


def _last_machine_initial(pt):
    return list(np.argsort(-pt[:, -1]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _local_search_insertion(pt_list, perm, deadline, max_rounds=5000):
    """
    Local search using insertion neighborhood: remove job at position i, insert at position j.
    Uses first-improvement strategy for speed.
    """
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = current[i]
            rest = current[:i] + current[i+1:]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                if time.time() > deadline:
                    break
                test = rest[:j] + [job] + rest[j:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _local_search_2opt(pt_list, perm, deadline, max_rounds=2000):
    """
    Local search using 2-opt segment reversal: reverse the segment from i to j.
    """
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm)

    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test = current[:i] + current[i:j+1][::-1] + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current_ms = ms
                    current = test
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_ms


def _simulated_annealing(pt_list, perm, deadline, seed=0):
    """Simulated annealing with insertion moves."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    # Temperature schedule
    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    
    # Estimate cooling rate to reach T_min in ~5000 steps
    # T * cooling^steps = T_min => cooling = (T_min/T)^(1/steps)
    target_steps = 5000
    if T > T_min:
        cooling = (T_min / T) ** (1.0 / target_steps)
    else:
        cooling = 0.999
    
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        # Random move: remove job at i, insert at j
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            # Avoid trivial moves
            j = (j + 1) % n
            
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(pt_list, test)
        
        delta = new_ms - current_ms
        if delta < 0:
            current = test
            current_ms = new_ms
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best = list(current)
        else:
            # Accept with probability exp(-delta/T)
            if delta < 100 * T:  # Avoid overflow in exp
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current = test
                    current_ms = new_ms
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best, best_ms


def _ox_crossover(p1, p2):
    """Order Crossover (OX)."""
    n = len(p1)
    if n < 2:
        return list(p1)
    
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    if i > j:
        i, j = j, i
    
    child = [None] * n
    for k in range(i, j + 1):
        child[k] = p1[k]
    
    in_segment = set(child[i:j+1])
    fill_values = [x for x in p2 if x not in in_segment]
    
    pos = 0
    for k in range(n):
        if child[k] is None:
            child[k] = fill_values[pos]
            pos += 1
            
    return child


def _ga_search(pt, pt_list, initial_pops, deadline, pop_size=40, generations=200):
    """
    Genetic Algorithm with Order Crossover (OX) and Insertion Local Search.
    """
    n_jobs = pt.shape[0]
    if n_jobs < 4:
        return initial_pops[0], _makespan_fast(pt_list, initial_pops[0])
    
    pop = [list(p) for p in initial_pops]
    for i in range(len(pop), pop_size):
        pop.append(_random_initial(n_jobs, 1000 + i))
    
    fitness = [_makespan_fast(pt_list, p) for p in pop]
    
    best_idx = int(np.argmin(fitness))
    best_perm = list(pop[best_idx])
    best_ms = fitness[best_idx]
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        new_pop = []
        sorted_indices = np.argsort(fitness)
        # Elitism: keep top 2
        for i in range(2):
            new_pop.append(list(pop[sorted_indices[i]]))
        
        while len(new_pop) < pop_size:
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if fitness[idx1] < fitness[idx2]:
                p1 = pop[idx1]
            else:
                p1 = pop[idx2]
            
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if fitness[idx1] < fitness[idx2]:
                p2 = pop[idx1]
            else:
                p2 = pop[idx2]
            
            child = _ox_crossover(p1, p2)
            
            if random.random() < 0.1:
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                child[i], child[j] = child[j], child[i]
            elif random.random() < 0.1:
                i = random.randint(0, n_jobs - 1)
                job = child.pop(i)
                j = random.randint(0, n_jobs - 1)
                child.insert(j, job)
            
            new_pop.append(child)
        
        pop = new_pop
        fitness = [_makespan_fast(pt_list, p) for p in pop]
        
        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_ms - 1e-9:
            best_ms = fitness[gen_best_idx]
            best_perm = list(pop[gen_best_idx])
            
            # Light polish
            if time.time() < deadline - 5.0:
                polished, polished_ms = _local_search_insertion(pt_list, best_perm, deadline - 5.0, max_rounds=100)
                if polished_ms < best_ms - 1e-9:
                    best_ms = polished_ms
                    best_perm = polished
                    worst_idx = int(np.argmax(fitness))
                    pop[worst_idx] = list(best_perm)
                    fitness[worst_idx] = best_ms

    return best_perm, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 140.0

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()

    # Build diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    for s in range(5):
        candidates.append(_random_initial(n_jobs, 100 + s))

    best_ms = float('inf')
    best_perm = candidates[0]
    candidate_results = []
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        candidate_results.append((ms, cand))
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    candidate_results.sort(key=lambda x: x[0])
    
    # Phase 1: Polish top 5 with Insertion LS
    polish_deadline = deadline - 40.0
    for idx, (ms0, cand) in enumerate(candidate_results[:5]):
        if time.time() > polish_deadline:
            break
        perm, ms = _local_search_insertion(pt_list, cand, polish_deadline, max_rounds=200)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Phase 2: Simulated Annealing on current best
    sa_deadline = deadline - 20.0
    if time.time() < sa_deadline:
        sa_perm, sa_ms = _simulated_annealing(pt_list, best_perm, sa_deadline, seed=42)
        if sa_ms < best_ms - 1e-9:
            best_ms = sa_ms
            best_perm = sa_perm
            # Polish SA result
            if time.time() < sa_deadline - 2.0:
                polished, polished_ms = _local_search_insertion(pt_list, best_perm, sa_deadline - 2.0, max_rounds=100)
                if polished_ms < best_ms - 1e-9:
                    best_ms = polished_ms
                    best_perm = polished

    # Phase 3: GA Search
    ga_deadline = deadline - 10.0
    if time.time() < ga_deadline:
        ga_inits = [c for _, c in candidate_results[:5]]
        if best_perm not in ga_inits:
            ga_inits.append(best_perm)
        
        ga_perm, ga_ms = _ga_search(pt, pt_list, ga_inits, ga_deadline, pop_size=30, generations=100)
        if ga_ms < best_ms - 1e-9:
            best_ms = ga_ms
            best_perm = ga_perm

    # Phase 4: Final Polish
    if time.time() < deadline - 2.0:
        # Use 2-opt for a different perspective
        if time.time() < deadline - 5.0:
            rev_perm, rev_ms = _local_search_2opt(pt_list, best_perm, deadline - 5.0, max_rounds=100)
            if rev_ms < best_ms - 1e-9:
                best_ms = rev_ms
                best_perm = rev_perm
        
        # Final insertion polish
        if time.time() < deadline - 1.0:
            perm, ms = _local_search_insertion(pt_list, best_perm, deadline - 1.0, max_rounds=500)
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_perm = perm

    return best_perm
# EVOLVE-BLOCK-END
