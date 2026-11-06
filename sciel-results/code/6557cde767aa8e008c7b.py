import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists
    successors = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute immediate successor counts
    imm_succ = np.zeros(n, dtype=int)
    for i in range(n):
        imm_succ[i] = len(successors[i])
    
    # Compute transitive successor counts (unique nodes reachable)
    # For n=30, BFS from each node is cheap
    trans_succ = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set([i])
        q = deque([i])
        cnt = 0
        while q:
            u = q.popleft()
            for v in successors[u]:
                if v not in visited:
                    visited.add(v)
                    cnt += 1
                    q.append(v)
        trans_succ[i] = cnt
    
    # Compute longest path from node i to any sink (in time units)
    # This is the "tail" time: minimum additional time needed after i on the critical path
    # Use reverse topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # tail[i] = max over successors s of (times[s] + tail[s]), or 0 if no successors
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in successors[u]:
            cand = times[v] + tail[v]
            if cand > best:
                best = cand
        tail[u] = best
    
    # head[i] = longest path from any source to i (in time units)
    head = np.zeros(n, dtype=float)
    for u in topo:
        for v in successors[u]:
            cand = times[u] + head[u]
            if cand > head[v]:
                head[v] = cand
    
    # Slack: how much of the cycle time is left for task i
    slack = np.maximum(cycle_time - times, 1e-9)
    
    # Criticality: how much of the critical path does this task sit on?
    # A task with high head + time + tail is on a longer critical path.
    crit_path = head + times + tail
    
    # Bottleneck slack heuristic:
    # Favor tasks with small slack relative to branching complexity.
    # Branching complexity: combine immediate and transitive successors.
    # Use a weighted blend: branching = 0.5 * imm_succ + 0.5 * trans_succ
    # Also incorporate criticality: tasks on longer critical paths are more "bottleneck-y"
    # We want high priority for: small slack, high branching, high criticality.
    
    # Priority score = (branching + 1) * (crit_path + 1) / slack
    # Higher score = higher priority
    branching = 0.5 * imm_succ + 0.5 * trans_succ
    scores = (branching + 1.0) * (crit_path + 1.0) / slack
    
    return scores
