"""DAG pebbling solver with LRU-based eviction and lookahead optimization."""
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

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

    # Precompute topo position for each vertex
    topo_pos = {v: i for i, v in enumerate(topo_order)}

    # For each vertex, precompute sorted list of child positions
    children_positions = {}
    for v in vertices:
        if v in children:
            children_positions[v] = sorted([topo_pos[c] for c in children[v]])
        else:
            children_positions[v] = []

    # Compute total fan-out
    fan_out = {v: len(children.get(v, [])) for v in vertices}

    # Compute number of parents (recomputation cost proxy)
    num_parents = {v: len(parents.get(v, [])) for v in vertices}

    # State
    state = {}
    for v in vertices:
        state[v] = 'blue' if v in input_set else 'red'

    fast_mem = set(v for v in vertices if state[v] == 'blue')
    operations = []

    # Track remaining uses (how many children still need this vertex)
    remaining_uses = {v: fan_out[v] for v in vertices}

    # For efficient next-use lookup, maintain a pointer index per vertex
    next_use_idx = {v: 0 for v in vertices}

    # LRU cache for eviction order - use a doubly-linked list for O(1) operations
    # Represented as: prev/next dicts and head/tail
    lru_prev = {}
    lru_next = {}
    lru_head = None
    lru_tail = None

    def lru_touch(v):
        """Move v to most recently used position (tail)."""
        nonlocal lru_head, lru_tail
        if v in lru_prev:
            # Remove from current position
            prev = lru_prev.get(v)
            next = lru_next.get(v)
            if prev is not None:
                lru_next[prev] = next
            else:
                lru_head = next
            if next is not None:
                lru_prev[next] = prev
            else:
                lru_tail = prev
            del lru_prev[v]
            del lru_next[v]
        # Add to tail
        lru_prev[v] = lru_tail
        lru_next[v] = None
        if lru_tail is not None:
            lru_next[lru_tail] = v
        else:
            lru_head = v
        lru_tail = v

    def lru_remove(v):
        """Remove v from LRU order."""
        nonlocal lru_head, lru_tail
        if v not in lru_prev:
            return
        prev = lru_prev.get(v)
        next = lru_next.get(v)
        if prev is not None:
            lru_next[prev] = next
        else:
            lru_head = next
        if next is not None:
            lru_prev[next] = prev
        else:
            lru_tail = prev
        del lru_prev[v]
        del lru_next[v]

    def lru_iter():
        """Iterate through LRU order from head to tail."""
        cur = lru_head
        while cur is not None:
            yield cur
            cur = lru_next.get(cur)

    def get_next_use_pos(v, current_idx):
        """Get the topo position of the next use of vertex v after current_idx."""
        cps = children_positions.get(v, [])
        idx = next_use_idx[v]
        while idx < len(cps) and cps[idx] <= current_idx:
            idx += 1
        next_use_idx[v] = idx
        if idx < len(cps):
            return cps[idx]
        return None

    def eviction_score(v, current_idx):
        """
        Score for eviction: lower score = more likely to be evicted.
        We want to keep vertices that are:
        1. Outputs (never evict if possible)
        2. Needed soon (small distance to next use)
        3. Have high fan-out (expensive to reload)
        4. Hard to recompute (many parents)
        
        Uses a log-sum-exp inspired smooth surrogate to combine factors.
        """
        if v in output_set:
            return float('inf')  # Never evict outputs
        
        rem = remaining_uses.get(v, 0)
        if rem <= 0:
            return -1000.0  # No more uses, evict immediately
        
        next_pos = get_next_use_pos(v, current_idx)
        if next_pos is None:
            return -500.0  # No future use, evict
        
        dist = max(1, next_pos - current_idx)
        
        # Factor 1: frequency of use (higher = keep)
        freq_factor = min(rem, 10) / (1.0 + dist)
        
        # Factor 2: recomputation cost (higher = keep)
        recompute_cost = num_parents.get(v, 0)
        cost_factor = min(recompute_cost, 20) / 5.0
        
        # Factor 3: fan-out benefit (higher = keep)
        fanout_factor = min(fan_out.get(v, 1), 10) / 10.0
        
        # Combine using log-sum-exp for smooth max-like behavior
        # This gives more weight to the dominant factor while remaining smooth
        import math
        max_factor = max(freq_factor, cost_factor, fanout_factor)
        if max_factor == 0:
            combined = 0.0
        else:
            # Log-sum-exp: log(exp(a) + exp(b) + exp(c)) ≈ max + log(sum(exp(x - max)))
            log_sum = math.log(
                math.exp(freq_factor - max_factor) +
                math.exp(cost_factor - max_factor) +
                math.exp(fanout_factor - max_factor)
            )
            combined = max_factor + log_sum
        
        # Deterministic tiebreaker based on vertex ID to avoid oscillation
        tiebreaker = (v % 1000) * 0.0001
        
        return combined + tiebreaker

    def evict_if_needed(exclude=None, current_idx=0):
        """Evict vertices until fast_mem is within budget."""
        nonlocal fast_mem, state, operations
        while len(fast_mem) > fast_memory_budget:
            if time.time() > deadline:
                return
            
            # Find best victim using eviction score
            best_victim = None
            best_score = float('inf')
            for u in lru_iter():
                if u not in fast_mem:
                    continue
                if u == exclude:
                    continue
                sc = eviction_score(u, current_idx)
                if sc < best_score:
                    best_score = sc
                    best_victim = u
            
            if best_victim is None:
                # Fallback: pick any non-excluded vertex
                for u in fast_mem:
                    if u != exclude:
                        best_victim = u
                        break
                if best_victim is None:
                    break
            
            operations.append(['store', best_victim])
            state[best_victim] = 'red'
            fast_mem.discard(best_victim)
            lru_remove(best_victim)

    # Initialize LRU order with inputs
    for v in input_set:
        lru_touch(v)
    
    # Precompute a heuristic for each vertex: estimated total cost if evicted
    # This helps prioritize which vertices to keep when budget is tight
    vertex_priority = {}
    for v in vertices:
        if v in output_set:
            vertex_priority[v] = float('inf')
        else:
            # Priority based on: how many times it's used, how expensive to recompute
            uses = fan_out.get(v, 1)
            recompute_cost = num_parents.get(v, 0)
            vertex_priority[v] = uses * (1 + recompute_cost * 0.5)

    # Process in topological order
    for idx, v in enumerate(topo_order):
        if time.time() > deadline:
            break

        parents_list = parents.get(v, [])

        # Load any red parents
        for p in parents_list:
            if state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)
                lru_touch(p)

        # Evict if over budget (exclude v since we're about to compute it)
        evict_if_needed(exclude=v, current_idx=idx)

        # Compute v if not already blue
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
            lru_touch(v)

        # After computing v, decrement remaining_uses of its parents
        for p in parents_list:
            remaining_uses[p] -= 1

        # Evict if over budget after compute
        evict_if_needed(exclude=None, current_idx=idx)
        
        # Update LRU order for parents that were just used
        for p in parents_list:
            if p in fast_mem:
                lru_touch(p)
        
        # Also touch v itself to mark it as recently used
        if v in fast_mem:
            lru_touch(v)

    # Ensure all outputs are blue at the end
    # Load outputs in order of priority (highest first) to minimize eviction pressure
    sorted_outputs = sorted(output_set, key=lambda x: vertex_priority.get(x, 0), reverse=True)
    for v in sorted_outputs:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
            lru_touch(v)
        evict_if_needed(exclude=None, current_idx=len(topo_order) - 1)

    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
