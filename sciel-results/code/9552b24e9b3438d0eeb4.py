"""
Communication-minimizing DAG pebbling.

Strategy:
  Process vertices in topological order. For each vertex, ensure all parents
  are in fast memory (load from slow if necessary, evicting low-priority items
  if fast memory is full). Compute the vertex. At the end, ensure all outputs
  are in fast memory.

  Priority for eviction: vertices with lower future usage (number of times they
  are used as parents in remaining computations) are evicted first.
"""

import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Parse DAG
    n = dag.get('n', None)
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    topo_order = dag.get('topo_order', list(range(n))) if n is not None else list(range(len(parents)))

    if n is None:
        n = len(parents)

    if fast_memory_budget <= 0:
        return []

    # Build children map
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    # Future usage: how many times each vertex is needed as a parent
    future_usage = [0] * n
    for v in range(n):
        for p in parents[v]:
            future_usage[p] += 1

    # State
    fast = set(inputs)
    slow = set()
    ops = []
    transfers = 0

    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break

        if v in inputs:
            continue

        # Ensure all parents are in fast
        for p in parents[v]:
            if p not in fast:
                if p in slow:
                    # Load p
                    if len(fast) < fast_memory_budget:
                        fast.add(p)
                        ops.append(['load', p])
                        transfers += 1
                    else:
                        # Evict a low-priority item
                        evict_candidates = [x for x in fast if x != p]
                        if evict_candidates:
                            evict = min(evict_candidates, key=lambda x: future_usage[x])
                            fast.discard(evict)
                            slow.add(evict)
                            ops.append(['store', evict])
                            transfers += 1
                            fast.add(p)
                            ops.append(['load', p])
                            transfers += 1

        # Compute v
        fast.add(v)
        ops.append(['compute', v])

    # Ensure outputs are in fast
    for o in outputs:
        if o not in fast:
            if o in slow:
                if len(fast) < fast_memory_budget:
                    fast.add(o)
                    ops.append(['load', o])
                    transfers += 1
                else:
                    evict_candidates = [x for x in fast if x not in outputs]
                    if evict_candidates:
                        evict = min(evict_candidates, key=lambda x: future_usage[x])
                        fast.discard(evict)
                        slow.add(evict)
                        ops.append(['store', evict])
                        transfers += 1
                        fast.add(o)
                        ops.append(['load', o])
                        transfers += 1

    return ops
