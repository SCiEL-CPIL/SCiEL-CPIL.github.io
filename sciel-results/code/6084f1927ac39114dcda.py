"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_greedy(times, preds, cycle_time, priorities):
    """Simulate greedy station filling. Returns number of stations."""
    n = len(times)
    completed = [False] * n
    eligible = set()
    # Find initially eligible tasks (no predecessors)
    for i in range(n):
        if not preds[i]:
            eligible.add(i)
    
    stations = 0
    remaining = cycle_time
    while len(eligible) < n or any(not c for c in completed):
        # Actually, let's track done count
        pass
    
    # Better approach:
    done = 0
    stations = 0
    remaining = cycle_time
    while done < n:
        # Find best eligible task that fits
        best_task = -1
        best_priority = -1e18
        for t in eligible:
            if times[t] <= remaining and priorities[t] > best_priority:
                best_priority = priorities[t]
                best_task = t
        if best_task == -1:
            # No eligible task fits, start new station
            stations += 1
            remaining = cycle_time
            # Try again with full remaining
            best_task = -1
            best_priority = -1e18
            for t in eligible:
                if times[t] <= remaining and priorities[t] > best_priority:
                    best_priority = priorities[t]
                    best_task = t
            if best_task == -1:
                # Shouldn't happen if all tasks fit in cycle_time
                break
        # Assign best_task to current station
        remaining -= times[best_task]
        completed[best_task] = True
        done += 1
        eligible.discard(best_task)
        # Add newly eligible tasks
        for j in range(n):
            if not completed[j] and j not in eligible:
                # Check if all predecessors are done
                all_done = True
                for p in preds[j]:
                    if not completed[p]:
                        all_done = False
                        break
                if all_done:
                    eligible.add(j)
    stations += 1  # Count the last station
    return stations


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial priority: LPT (longest first)
    p = times.copy()
    
    # Precompute successor lists for efficiency
    succs = [[] for _ in range(n)]
    for j in range(n):
        for pi in preds[j]:
            succs[pi].append(j)
    
    def _count_stations(pri):
        done = 0
        stations = 0
        remaining = cycle_time
        completed = [False] * n
        eligible = []
        for i in range(n):
            if not preds[i]:
                eligible.append(i)
        
        while done < n:
            best_task = -1
            best_priority = -1e18
            for t in eligible:
                if not completed[t] and times[t] <= remaining and pri[t] > best_priority:
                    best_priority = pri[t]
                    best_task = t
            if best_task == -1:
                stations += 1
                remaining = cycle_time
                best_task = -1
                best_priority = -1e18
                for t in eligible:
                    if not completed[t] and times[t] <= remaining and pri[t] > best_priority:
                        best_priority = pri[t]
                        best_task = t
                if best_task == -1:
                    break
            remaining -= times[best_task]
            completed[best_task] = True
            done += 1
            eligible.remove(best_task)
            for j in succs[best_task]:
                if not completed[j]:
                    all_done = True
                    for pi in preds[j]:
                        if not completed[pi]:
                            all_done = False
                            break
                    if all_done:
                        eligible.append(j)
        stations += 1
        return stations
    
    best_stations = _count_stations(p)
    best_p = p.copy()
    
    # Local search: try swapping priorities of pairs
    # Use a more efficient approach: try adjusting individual priorities
    # or use simulated annealing / hill climbing on priority vectors
    
    # Strategy: coordinate descent - for each task, try perturbing its priority
    # and see if station count improves
    import random
    rng = random.Random(42)
    
    # Try swap-based local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs (or a subset for speed)
        if n <= 30:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            rng.shuffle(pairs)
        else:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            rng.shuffle(pairs)
        
        for i, j in pairs:
            if time.time() > deadline:
                break
            # Swap priorities
            p[i], p[j] = p[j], p[i]
            s = _count_stations(p)
            if s < best_stations:
                best_stations = s
                best_p = p.copy()
                improved = True
            else:
                # Revert
                p[i], p[j] = p[j], p[i]
    
    return best_p
# EVOLVE-BLOCK-END
