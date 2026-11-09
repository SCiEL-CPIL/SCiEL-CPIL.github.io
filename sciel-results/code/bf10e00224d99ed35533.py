"""Communication-minimizing DAG pebbling with bounded fast memory."""
import time
import heapq
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def _topo_order(dag):
    n = len(dag)
    children = defaultdict(list)
    in_degree = [0] * n
    for i, v in enumerate(dag):
        for p in v.get('parents', []):
            children[p].append(i)
            in_degree[i] += 1
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for c in children[u]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0

    n = len(dag)
    if n == 0:
        return []

    parent_map = [list(dag[i].get('parents', [])) for i in range(n)]
    children_map = defaultdict(list)
    for i in range(n):
        for p in parent_map[i]:
            children_map[p].append(i)

    inputs = [i for i in range(n) if len(parent_map[i]) == 0]
    outputs = [i for i in range(n) if dag[i].get('outputs', False) or dag[i].get('is_output', False)]
    if not outputs:
        outputs = [i for i in range(n) if not children_map[i]]
        if not outputs:
            outputs = [n - 1]

    order = _topo_order(dag)

    # State
    fast = set()          # vertices in fast memory (red)
    computed = set()      # vertices that have been computed
    ops = []

    # Remaining children count (for eviction priority)
    rem_children = [len(children_map[i]) for i in range(n)]

    def evict_if_needed():
        """Evict a vertex from fast memory to make room if needed."""
        if len(fast) < fast_memory_budget:
            return
        if not fast:
            return
        # Choose victim: min remaining children, prefer non-outputs
        def score(v):
            rc = rem_children[v]
            if v in outputs:
                rc -= 10000  # strongly protect outputs
            return rc
        victim = min(fast, key=score)
        ops.append(['delete', victim])
        fast.discard(victim)

    def load(v):
        """Load vertex v into fast memory from slow memory."""
        if v in fast:
            return
        evict_if_needed()
        fast.add(v)
        ops.append(['load', v])

    for v in order:
        if time.time() > DEADLINE:
            break
        if v in computed:
            continue
        # Ensure all parents are in fast memory
        for p in parent_map[v]:
            if p not in fast:
                load(p)
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        fast.add(v)
        # Update remaining children of parents
        for p in parent_map[v]:
            rem_children[p] -= 1
        # If v is not an output and has no remaining children, delete it
        if v not in outputs and rem_children[v] == 0:
            if v in fast:
                ops.append(['delete', v])
                fast.discard(v)

    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in fast:
            load(o)

    # Final cleanup: delete non-outputs that are no longer needed
    # (optional, doesn't affect correctness)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
