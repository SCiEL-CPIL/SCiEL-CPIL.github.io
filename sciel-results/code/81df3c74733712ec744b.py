"""Pebbling solver with cost-based eviction and lookahead heuristics."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    if not vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Build adjacency
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    # Compute in-degrees and topological order
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1

    topo_order = []
    queue = deque(v for v in vertices if in_degree[v] == 0)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Identify inputs and outputs
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)

    # Compute depth (distance from nearest input)
    depth = {}
    for v in topo_order:
        if not parents.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents[v]) + 1

    # Compute subtree size
    subtree_size = {}
    for v in reversed(topo_order):
        if children[v]:
            subtree_size[v] = 1 + sum(subtree_size.get(c, 0) for c in children[v])
        else:
            subtree_size[v] = 1

    # Compute last-use position in topo order (transitive)
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    last_use = {}
    for v in reversed(topo_order):
        if children[v]:
            last_use[v] = max(last_use.get(c, topo_pos[v]) for c in children[v])
        else:
            last_use[v] = topo_pos[v]

    # Compute recompute_cost: number of operations needed to recompute v from scratch
    # This is an approximation: size of the sub-DAG rooted at v
    recompute_cost = {}
    for v in reversed(topo_order):
        if v in input_set:
            recompute_cost[v] = 0  # inputs don't need recomputation, just reload
        elif not parents.get(v, []):
            recompute_cost[v] = 1
        else:
            # Cost to recompute v = 1 (compute v) + sum of costs to ensure parents are available
            # Simplified: just use subtree size as proxy
            recompute_cost[v] = subtree_size[v]

    # State tracking
    state = {}  # vertex -> 'blue' or 'red'
    for v in vertices:
        state[v] = 'blue' if v in input_set else 'red'

    fast_mem = set(v for v in vertices if state[v] == 'blue')
    operations = []

    # Track remaining uses (how many children still need this vertex)
    remaining_uses = defaultdict(int)
    for v in vertices:
        remaining_uses[v] = len(children[v])

    def eviction_cost(u):
        """
        Estimate the total future cost if we evict u now.
        Lower cost = better to evict.
        
        Cost components:
        1. If u is an input: reload cost = 1 (load operation)
        2. If u is computed: potential recompute cost
        3. Penalty for evicting outputs (should never happen, but high penalty)
        4. Benefit of keeping vertices with many remaining uses
        """
        if u in output_set:
            return float('inf')  # Never evict outputs
        
        # Base cost: what it costs to get u back
        if u in input_set:
            base_cost = 1  # Just a load
        else:
            # Cost to recompute: approximate as recompute_cost[u]
            # But only if we actually need it again
            base_cost = recompute_cost.get(u, 1)
        
        # Multiply by remaining uses (if it's used many more times, evicting is more expensive)
        uses_factor = max(1, remaining_uses[u])
        
        # Total estimated cost
        total_cost = base_cost * uses_factor
        
        # Small penalty for large subtrees (more likely to be needed soon)
        total_cost += 0.1 * subtree_size[u]
        
        # Small bonus for early last-use (less likely to be needed soon)
        total_cost -= 0.01 * (topo_pos.get(u, 0))
        
        return total_cost

    def evict_if_needed(exclude=None):
        """Evict vertices until fast_mem is within budget."""
        nonlocal fast_mem, state, operations
        while len(fast_mem) > fast_memory_budget:
            if time.time() > deadline:
                return
            
            # Find best victim using cost-based eviction
            best_victim = None
            best_cost = float('inf')
            for u in fast_mem:
                if u == exclude:
                    continue
                cost = eviction_cost(u)
                if cost < best_cost:
                    best_cost = cost
                    best_victim = u
            
            if best_victim is None:
                # All in fast_mem are excluded or outputs; try any non-excluded
                for u in fast_mem:
                    if u != exclude:
                        best_victim = u
                        break
                if best_victim is None:
                    break
            
            operations.append(['store', best_victim])
            state[best_victim] = 'red'
            fast_mem.discard(best_victim)

    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break

        parents_list = parents.get(v, [])

        # Load any red parents
        for p in parents_list:
            if state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)

        # Evict if over budget (exclude v since we're about to compute it)
        evict_if_needed(exclude=v)

        # Compute v if not already blue
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)

        # After computing v, decrement remaining_uses of its parents
        for p in parents_list:
            remaining_uses[p] -= 1

        # Evict if over budget after compute
        evict_if_needed(exclude=None)

    # Ensure all outputs are blue at the end
    for v in output_set:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        evict_if_needed(exclude=None)

    # Post-processing: remove redundant load/store pairs
    optimized_ops = []
    i = 0
    while i < len(operations):
        if i + 1 < len(operations):
            op1, v1 = operations[i]
            op2, v2 = operations[i + 1]
            if v1 == v2:
                if (op1 == 'load' and op2 == 'store') or (op1 == 'store' and op2 == 'load'):
                    i += 2
                    continue
        optimized_ops.append(operations[i])
        i += 1
    operations = optimized_ops

    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
# EVOLVE-BLOCK-END
