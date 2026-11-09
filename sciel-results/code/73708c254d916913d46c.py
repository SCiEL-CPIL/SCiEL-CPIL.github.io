import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Build successor list
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)

    out_degree = [len(succ[i]) for i in range(n)]
    od = out_degree[:]

    # Reverse topological order via Kahn on reverse graph (start at sinks)
    q = [i for i in range(n) if od[i] == 0]
    rpv = times.copy()
    # Longest path depth to end (in time units)
    depth = times.copy()
    # Minimum number of stations required to complete downstream subtree
    # This is a lower bound on stations: ceil((total_time) / cycle_time) for a single path,
    # but for a tree, it's more complex. We'll use the sum of times in the downstream subtree
    # divided by cycle_time as a proxy for "stations needed".
    # Actually, the directive says "minimum number of remaining stations required to complete its downstream subtree".
    # A simple lower bound is ceil(sum(times in downstream) / cycle_time).
    # Let's compute downstream_time[u] = times[u] + sum(downstream_time[s] for s in succ[u])
    # And use that.
    downstream_time = times.copy()

    while q:
        if time.time() > deadline:
            # Return best available
            break
        u = q.pop()
        for p in preds[u]:
            rpv[p] += rpv[u]
            depth[p] = max(depth[p], depth[u] + times[u])  # Longest path depth
            downstream_time[p] += downstream_time[u]
            od[p] -= 1
            if od[p] == 0:
                q.append(p)

    cycle_time = instance.get("cycle_time", 40.0)

    # Bottleneck density: times[u] / downstream_time[u]
    # This penalizes long tasks that are part of a large downstream subtree
    # We want to prioritize tasks that are critical bottlenecks.
    # If downstream_time is large, the task is part of a big chunk of work.
    # If times[u] is large relative to that, it's a bottleneck.
    # However, we also want to preserve the RPV signal.
    # Let's combine: priority = rpv + alpha * bottleneck_density
    # But bottleneck_density is small if downstream_time is large.
    # Maybe we should use: priority = rpv * (1 + alpha * bottleneck_density)
    # Or simply: priority = rpv + alpha * (times[u] / (downstream_time[u] + 1))

    # Let's try: priority = rpv + 0.5 * (times / (downstream_time + 1))
    # This adds a small bonus to tasks that are bottlenecks within their subtree.
    # But wait, if downstream_time is large, the bonus is small.
    # If downstream_time is small (leaf), the bonus is times/2.
    # This might favor leaves.
    # The directive says: "penalize long tasks that sit on critical paths where station capacity is scarce".
    # This suggests we should prioritize tasks that consume a lot of the scarce station capacity.
    # Maybe we should use: priority = rpv + alpha * (times / cycle_time)
    # But that's just adding a multiple of times.
    # The "bottleneck density" is times / min_stations.
    # min_stations ~ downstream_time / cycle_time.
    # So density ~ times / (downstream_time / cycle_time) = (times * cycle_time) / downstream_time.
    # This is the fraction of the downstream time that is consumed by the task itself.
    # This makes sense: a task that is 50% of its downstream subtree is a bigger bottleneck than one that is 10%.

    density = np.zeros(n)
    for i in range(n):
        if downstream_time[i] > 0:
            density[i] = (times[i] * cycle_time) / downstream_time[i]
        else:
            density[i] = 0.0

    # Combine RPV and density
    # We want to keep the RPV design but add this correction.
    # Let's try a weighted sum.
    # Since RPV is in time units and density is a ratio (0..1), we need to scale.
    # Let's scale density by the mean RPV or something similar.
    # Or just use a small alpha.
    # Let's try: priority = rpv + 0.1 * density * np.mean(rpv)
    # This ensures the density term is comparable to RPV.

    mean_rpv = np.mean(rpv) if n > 0 else 1.0
    priority = rpv + 0.1 * density * mean_rpv

    return priority
