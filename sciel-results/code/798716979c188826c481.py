"""Assembly Line Balancing - Direct structural approach with critical path analysis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque

def priority(instance):
    """
    Score each assembly task for the priority rule.
    Uses a direct structural approach:
    1. Compute critical path length (longest path from task to end of line)
    2. Compute number of successors (positional breadth)
    3. Combine into a priority score that favors tasks that are on the critical path
       and have many successors (to unlock more downstream tasks early)
    4. Per-instance: try several weight combinations and return the one that minimizes stations.
    """
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s limit

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build graph structures
    adj = [[] for _ in range(n)]  # adj[i] = successors of i
    preds_list = []
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        preds_list.append(preds)
        for p in preds:
            adj[p].append(i)

    # Topological sort
    in_deg = np.array([len(p) for p in preds_list], dtype=int)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    in_deg_tmp = in_deg.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in adj[node]:
            in_deg_tmp[succ] -= 1
            if in_deg_tmp[succ] == 0:
                queue.append(succ)

    if len(topo_order) < n:
        # Cycle detected; fall back to task times
        return times.copy()

    # Compute longest path to end (LPE) for each task
    lpe = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in adj[node]:
            val = times[succ] + lpe[succ]
            if val > max_succ:
                max_succ = val
        lpe[node] = times[node] + max_succ

    # Compute longest path from start (LPS) for each task
    lps = np.zeros(n, dtype=float)
    for node in topo_order:
        for succ in adj[node]:
            val = times[node] + lps[node]
            if val > lps[succ]:
                lps[succ] = val

    # Number of successors (transitive)
    num_succ = np.zeros(n, dtype=int)
    for node in reversed(topo_order):
        for succ in adj[node]:
            num_succ[node] += num_succ[succ] + 1

    # Position in topological order (0-based, 0 = earliest)
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx

    # Normalized position (0 = first, 1 = last)
    if n > 1:
        norm_pos = pos_in_topo / (n - 1)
    else:
        norm_pos = np.zeros(n, dtype=float)

    # Candidate priority scores
    candidates = {}

    # Critical path length (most important structural measure)
    candidates["lpe"] = lpe.copy()

    # Critical path + successor count
    candidates["lpe_succ"] = lpe + 0.01 * num_succ.astype(float)

    # Critical path scaled by successor count
    candidates["lpe_times_succ"] = lpe * (1.0 + 0.05 * np.log1p(num_succ.astype(float)))

    # Longest path from start
    candidates["lps"] = lps.copy()

    # Task time
    candidates["time"] = times.copy()

    # Task time + critical path
    candidates["time_lpe"] = times + 0.1 * lpe

    # Inverse topological position (earlier is better)
    candidates["early_pos"] = 1.0 - norm_pos

    # Combined: critical path weighted by how much it constrains the line
    candidates["combo1"] = lpe * (1.0 + 0.02 * num_succ.astype(float))
    candidates["combo2"] = 0.7 * lpe + 0.3 * lps
    candidates["combo3"] = lpe + 0.05 * num_succ.astype(float) - 0.01 * norm_pos
    candidates["combo4"] = (lpe / (lpe.max() + 1e-9)) * (1.0 + 0.1 * num_succ.astype(float) / (num_succ.max() + 1e-9))
    candidates["combo5"] = lpe * (1.0 + 0.03 * np.sqrt(num_succ.astype(float)))
    candidates["combo6"] = lpe + 0.1 * num_succ.astype(float)
    candidates["combo7"] = 0.8 * lpe + 0.2 * times * (1.0 + 0.05 * num_succ.astype(float))
    candidates["combo8"] = lpe * np.exp(0.01 * num_succ.astype(float))

    # Pre-compute for speed
    pred_sets = [set(p) for p in preds_list]

    def simulate(priority_scores):
        """Fast greedy simulation using a heap of eligible tasks sorted by priority."""
        completed = [False] * n
        stations = 0
        current_load = 0.0
        remaining_count = n

        # pending[t] = number of uncompleted predecessors of t
        pending = np.array([len(p) for p in preds_list], dtype=int)

        # Heap entries: (-priority, task_id)
        heap = []

        # Initially eligible: tasks with no predecessors that fit
        for t in range(n):
            if pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                heapq.heappush(heap, (-priority_scores[t], t))

        while remaining_count > 0:
            # Pop highest-priority eligible task that fits remaining capacity
            placed = False
            while heap:
                neg_pri, t = heapq.heappop(heap)
                if completed[t]:
                    continue
                if times[t] <= cycle_time - current_load + 1e-9:
                    # Place it
                    completed[t] = True
                    current_load += times[t]
                    remaining_count -= 1
                    placed = True
                    # Check successors
                    for s in adj[t]:
                        pending[s] -= 1
                        if pending[s] == 0 and times[s] <= cycle_time + 1e-9:
                            heapq.heappush(heap, (-priority_scores[s], s))
                    break

            if not placed:
                # No eligible task fits in current station; start new station
                stations += 1
                current_load = 0.0
                # Re-push all eligible (pending==0, not completed) tasks
                # that were rejected due to capacity
                for t in range(n):
                    if not completed[t] and pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                        heapq.heappush(heap, (-priority_scores[t], t))

            if not heap:
                # Shouldn't happen for feasible instances; break to avoid infinite loop
                stations += 1
                break

        return stations

    # Evaluate candidates and pick the best
    best_priority = None
    best_stations = float('inf')

    for name, p in candidates.items():
        if time.time() > DEADLINE:
            break
        stations = simulate(p)
        if stations < best_stations:
            best_stations = stations
            best_priority = p.copy()

    # Try some additional parameter sweeps if time allows
    if best_priority is not None and time.time() < DEADLINE - 30:
        # Sweep over lpe + alpha * num_succ
        for alpha in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
            if time.time() > DEADLINE:
                break
            p = lpe + alpha * num_succ.astype(float)
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()

        # Sweep over a*lpe + b*lps
        for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            if time.time() > DEADLINE:
                break
            b = 1.0 - a
            p = a * lpe + b * lps
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()

        # Sweep over lpe * (1 + beta * sqrt(num_succ))
        for beta in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
            if time.time() > DEADLINE:
                break
            p = lpe * (1.0 + beta * np.sqrt(num_succ.astype(float)))
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()

    # Local search polish: perturb the priority vector and try to improve
    if best_priority is not None and time.time() < DEADLINE - 5:
        # Use a simple hill-climbing approach: randomly perturb a few tasks' priorities
        # and see if it reduces station count
        rng = np.random.default_rng(42)
        current_priority = best_priority.copy()
        current_stations = best_stations
        improvement_found = False

        # Try a few rounds of local search
        for _ in range(50):
            if time.time() > DEADLINE - 2:
                break

            # Create a perturbed version
            perturbed = current_priority.copy()
            # Perturb 1-3 random tasks
            n_perturb = rng.integers(1, 4)
            indices = rng.choice(n, size=n_perturb, replace=False)
            for idx in indices:
                # Scale the priority of this task by a random factor
                factor = rng.uniform(0.5, 1.5)
                perturbed[idx] *= factor

            stations = simulate(perturbed)
            if stations < current_stations:
                current_priority = perturbed
                current_stations = stations
                improvement_found = True
                # If we improved, continue searching from this new point
            else:
                # No improvement; try a different perturbation
                pass

        if improvement_found:
            best_priority = current_priority
            best_stations = current_stations

    if best_priority is None:
        best_priority = lpe.copy()

    return np.asarray(best_priority, dtype=float)

# EVOLVE-BLOCK-END
