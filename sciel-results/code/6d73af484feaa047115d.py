# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def _simulate_station_count(times, predecessors, cycle_time, priorities):
    """Simulate greedy assignment and return number of stations.
    Uses a max-heap (negated priorities) for the active set so that
    the highest-priority eligible task is found in O(log n) per pop.
    """
    n = len(times)
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i]
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)

    remaining_preds = pred_count[:]
    assigned = [False] * n

    # Max-heap via negated priorities; tie-break by task id for determinism
    heap = []
    for i in range(n):
        if remaining_preds[i] == 0:
            heapq.heappush(heap, (-priorities[i], i))

    station_count = 0
    remaining_time = 0.0
    assigned_count = 0

    while assigned_count < n:
        if remaining_time <= 1e-12:
            station_count += 1
            remaining_time = cycle_time

        # Pop until we find an eligible task that fits
        best_task = -1
        candidates = []
        while heap:
            neg_p, t = heapq.heappop(heap)
            if assigned[t]:
                continue
            if times[t] <= remaining_time + 1e-9:
                best_task = t
                break
            else:
                candidates.append((neg_p, t))

        if best_task == -1:
            # Push back what we popped, open a new station
            for item in candidates:
                heapq.heappush(heap, item)
            station_count += 1
            remaining_time = cycle_time
            # Try again
            while heap:
                neg_p, t = heapq.heappop(heap)
                if assigned[t]:
                    continue
                if times[t] <= remaining_time + 1e-9:
                    best_task = t
                    break
            if best_task == -1:
                # Fallback: pick smallest unassigned
                for t in range(n):
                    if not assigned[t]:
                        best_task = t
                        break

        assigned[best_task] = True
        assigned_count += 1
        remaining_time -= times[best_task]

        for s in successors[best_task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                heapq.heappush(heap, (-priorities[s], s))

    return station_count


def _compute_ranks(times, predecessors):
    """Compute rank for each task: longest path from task to end."""
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    in_deg = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    longest = [0.0] * n
    for u in reversed(topo_order):
        if successors[u]:
            longest[u] = times[u] + max(longest[s] for s in successors[u])
        else:
            longest[u] = times[u]

    return longest


def priority(instance):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Compute ranks
    ranks = _compute_ranks(times, predecessors)

    # Build successors for auxiliary features
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=float)

    # Number of successors (recursive depth from node to sinks)
    depth = np.zeros(n)
    in_deg = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    for u in reversed(topo_order):
        if successors[u]:
            depth[u] = 1.0 + max(depth[s] for s in successors[u])
        else:
            depth[u] = 0.0

    # Successor count (total descendants, not just immediate)
    desc_count = np.zeros(n)
    for u in reversed(topo_order):
        for s in successors[u]:
            desc_count[u] += 1 + desc_count[s]

    # Build candidate priority vectors (richer set)
    candidates = []

    # Scheme 1: rank only
    candidates.append(ranks.copy())

    # Scheme 2: time only (LPT)
    candidates.append(times.copy())

    # Scheme 3: rank + alpha * time
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        candidates.append(ranks + alpha * times)

    # Scheme 4: rank + beta * out_degree
    for beta in [0.5, 1.0, 2.0, 3.0]:
        candidates.append(ranks + beta * out_degree)

    # Scheme 5: rank * (1 + out_degree)
    candidates.append(ranks * (1.0 + out_degree))

    # Scheme 6: rank + depth
    candidates.append(ranks + depth)

    # Scheme 7: rank + desc_count
    candidates.append(ranks + desc_count)

    # Scheme 8: (rank + time) * (1 + out_degree * 0.5)
    candidates.append((ranks + times) * (1.0 + 0.5 * out_degree))

    # Scheme 9: rank * (1 + depth * 0.1) + time
    candidates.append(ranks * (1.0 + 0.1 * depth) + times)

    # Scheme 10: rank + 0.5*time + 0.3*desc_count
    candidates.append(ranks + 0.5 * times + 0.3 * desc_count)

    # Scheme 11: (rank + time) * (1 + depth * 0.2)
    candidates.append((ranks + times) * (1.0 + 0.2 * depth))

    # Scheme 12: rank * (1 + 0.5 * out_degree) + 0.5 * time
    candidates.append(ranks * (1.0 + 0.5 * out_degree) + 0.5 * times)

    # Scheme 13: rank + time + 0.2 * desc_count + 0.1 * depth
    candidates.append(ranks + times + 0.2 * desc_count + 0.1 * depth)

    # Evaluate all candidates
    best_score = float('inf')
    best_priorities = None
    for cand in candidates:
        score = _simulate_station_count(times, predecessors, cycle_time, cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()

    # Local search: multi-neighborhood hill climbing with time budget
    rng = np.random.default_rng(42)
    rank_std = max(1.0, float(np.std(ranks)))
    base_scale = rank_std * 0.3

    improved = True
    stall_count = 0
    eval_count = 0
    max_evals = 20000  # hard cap on evaluations to stay within time

    while improved and (time.time() < deadline - 8) and eval_count < max_evals:
        improved = False
        stall_count += 1
        # Increase perturbation scale as we stall
        scale = base_scale * (1.0 + 0.25 * stall_count)

        # Neighborhood 1: Gaussian perturbation of 1-3 tasks
        for _ in range(60):
            if time.time() > deadline - 8 or eval_count >= max_evals:
                break
            num_perturb = int(rng.integers(1, 4))
            idxs = rng.choice(n, size=num_perturb, replace=False)
            new_priorities = best_priorities.copy()
            for i in idxs:
                new_priorities[i] += rng.normal(0, scale)
            eval_count += 1
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
                improved = True
                stall_count = 0

        # Neighborhood 2: Swap two tasks' priority values
        if time.time() < deadline - 8 and eval_count < max_evals:
            for _ in range(40):
                if time.time() > deadline - 8 or eval_count >= max_evals:
                    break
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                new_priorities = best_priorities.copy()
                new_priorities[i], new_priorities[j] = new_priorities[j], new_priorities[i]
                eval_count += 1
                score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
                if score < best_score:
                    best_score = score
                    best_priorities = new_priorities
                    improved = True
                    stall_count = 0

        # Neighborhood 3: Move a task's priority to another task's value (asymmetric)
        if time.time() < deadline - 8 and eval_count < max_evals:
            for _ in range(30):
                if time.time() > deadline - 8 or eval_count >= max_evals:
                    break
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                new_priorities = best_priorities.copy()
                # Give task i the priority of task j, and give task j a small perturbation
                new_priorities[i] = best_priorities[j] + rng.normal(0, scale * 0.3)
                new_priorities[j] = best_priorities[i] + rng.normal(0, scale * 0.3)
                eval_count += 1
                score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
                if score < best_score:
                    best_score = score
                    best_priorities = new_priorities
                    improved = True
                    stall_count = 0

        # Multi-start: if stalled, try a fresh random restart
        if stall_count >= 3 and time.time() < deadline - 15 and eval_count < max_evals:
            base = candidates[int(rng.integers(0, len(candidates)))]
            new_priorities = base + rng.normal(0, base_scale, size=n)
            eval_count += 1
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
                improved = True
                stall_count = 0

    # Final polish: very small perturbations to squeeze out last improvements
    if time.time() < deadline - 3:
        polish_scale = base_scale * 0.1
        for _ in range(200):
            if time.time() > deadline - 3 or eval_count >= max_evals:
                break
            idx = int(rng.integers(0, n))
            new_priorities = best_priorities.copy()
            new_priorities[idx] += rng.normal(0, polish_scale)
            eval_count += 1
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities

    return best_priorities
# EVOLVE-BLOCK-END
