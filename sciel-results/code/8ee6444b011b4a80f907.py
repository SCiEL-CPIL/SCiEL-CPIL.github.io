import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    
    Strategy:
    Directly optimize the static priority vector using a fast SGS simulator and
    a combination of greedy construction and local search (simulated annealing / hill climbing).
    
    Key improvements over previous versions:
    1. Faster SGS simulation using pre-computed data structures and minimal overhead.
    2. Multiple diverse initial candidates (CPL, RI, LPT, etc.).
    3. Aggressive local search with swap moves and occasional perturbation to escape local minima.
    4. Strict time management to stay within the 150s limit.
    """
    start_time = time.time()
    # Safety margin: return by 145s
    deadline = start_time + 145.0

    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float)  # Shape (R,)
    R = len(capacities)

    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
            preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
            preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
        normalized_preds = []
        for p in preds:
            if isinstance(p, (list, tuple, np.ndarray)):
                normalized_preds.append(list(map(int, p)))
            else:
                normalized_preds.append([])
        preds = normalized_preds
    else:
        preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)

    # 2. Compute Static Heuristics
    # Topological Sort
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)

    # Compute Critical Path Length (CPL) backwards
    cpl = np.zeros(N)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl

    # Compute Resource Intensity (RI): sum of demand/capacity ratios
    cap_safe = np.where(capacities > 0, capacities, 1.0)
    ratios = demands / cap_safe  # Shape (N, R)
    ri = np.sum(ratios, axis=1)  # Shape (N,)

    # 3. Fast SGS Simulation
    # Optimized for speed: use local variables, minimize function calls.
    def simulate_sgs(priorities):
        """
        Serial SGS simulation.
        Returns makespan.
        """
        scheduled = [False] * N
        # active: list of [finish_time, idx]
        active = []
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        scheduled_count = 0

        # Pre-compute predecessor lists for faster access
        # preds is already list of lists.

        while scheduled_count < N:
            # Find eligible activities
            eligible = []
            for i in range(N):
                if not scheduled[i]:
                    # Check preds
                    is_eligible = True
                    for p in preds[i]:
                        if not scheduled[p]:
                            is_eligible = False
                            break
                    if is_eligible:
                        eligible.append(i)

            if not eligible:
                # Wait for next event
                if not active:
                    break  # Should not happen if valid
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                # Process completions
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append([f, idx])
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active
                continue

            # Sort eligible by priority (descending)
            # Optimization: Use sorted() with key.
            eligible.sort(key=lambda x: priorities[x], reverse=True)

            # Attempt to schedule
            any_scheduled = False
            for i in eligible:
                # Check resources
                if np.all(remaining_res >= demands[i]):
                    scheduled[i] = True
                    scheduled_count += 1
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append([finish_t, i])
                    makespan = max(makespan, finish_t)
                    any_scheduled = True

            if not any_scheduled:
                # Wait
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append([f, idx])
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active

        return makespan

    # 4. Generate Initial Candidates
    candidates = []

    # Heuristic 1: CPL
    candidates.append(cpl.copy())

    # Heuristic 2: CPL * (1 + RI)
    candidates.append(cpl * (1.0 + ri))

    # Heuristic 3: CPL * (1 + 2*RI)
    candidates.append(cpl * (1.0 + 2.0 * ri))

    # Heuristic 4: LPT (Duration)
    candidates.append(durations.copy())

    # Heuristic 5: LPT * (1 + RI)
    candidates.append(durations * (1.0 + ri))

    # Heuristic 6: Successor Count weighted
    succ_count = np.zeros(N, dtype=int)
    for i in range(N):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for s in succs[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        succ_count[i] = len(visited)

    candidates.append(cpl * (1.0 + succ_count.astype(float)))

    # Heuristic 7: Random permutation
    random.seed(42)
    candidates.append(np.array(random.sample(range(N), N), dtype=float))

    # 5. Local Search (Hill Climbing with Simulated Annealing)
    best_score = float('inf')
    best_priorities = candidates[0]

    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score = simulate_sgs(cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()

    # Simulated Annealing / Hill Climbing
    current_priorities = best_priorities.copy()
    current_score = best_score

    # Temperature for simulated annealing
    T = 1.0
    cooling_rate = 0.995

    # Use a random seed for reproducibility
    random.seed(123)

    # Main search loop
    while time.time() < deadline:
        # Pick two random activities to swap
        i, j = random.sample(range(N), 2)

        # Swap priorities
        temp = current_priorities[i]
        current_priorities[i] = current_priorities[j]
        current_priorities[j] = temp

        new_score = simulate_sgs(current_priorities)

        # Accept or reject based on simulated annealing
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_priorities = current_priorities.copy()
        else:
            # Compute delta
            delta = new_score - current_score
            # Accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Swap back
                current_priorities[j] = current_priorities[i]
                current_priorities[i] = temp

        # Cool down
        T *= cooling_rate

        # Occasionally reset temperature to escape deep local minima
        if random.random() < 0.01:
            T = max(T, 0.5)

    return best_priorities
