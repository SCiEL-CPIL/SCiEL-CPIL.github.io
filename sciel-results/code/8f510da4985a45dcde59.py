"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """Returns a list of vertices in topological order."""
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return list(vertices)
    return order

def _build_graph_info(dag):
    """Pre-compute graph information for scheduling."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    # Determine input vertices (no parents)
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    return order, children_map, outputs, inputs

def _compute_usage_counts(dag, order, children_map):
    """Compute how many times each vertex's value is needed by children."""
    vertices = dag['vertices']
    parents = dag['parents']
    
    # For each vertex, count how many children need it
    usage = {v: len(children_map[v]) for v in vertices}
    
    return usage

def _min_cost_flow_schedule(dag, fast_memory_budget):
    """
    Use min-cost flow to determine optimal set of vertices to keep in fast memory.
    
    Formulation:
    - Each vertex v is split into v_in and v_out
    - Edge v_in -> v_out has capacity 1, cost 1 (loading the vertex)
    - Edges from parent_out to child_in have infinite capacity, cost 0 (free reuse if both in memory)
    - We need to send flow from source to each output
    - The min-cost flow determines which vertices to load (minimizing total loads)
    
    Then we reconstruct a valid schedule from the flow solution.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order, children_map, outputs_set, inputs = _build_graph_info(dag)
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Index mapping
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build flow network
    # Nodes: for each vertex v, we have v_in (2*i) and v_out (2*i+1)
    # Source: 2*n, Sink: 2*n+1
    source = 2 * n
    sink = 2 * n + 1
    total_nodes = 2 * n + 2
    
    # Adjacency list for the flow network
    # Each edge: (to_node, capacity, cost, reverse_edge_index)
    adj = [[] for _ in range(total_nodes)]
    
    def add_edge(u, v, cap, cost):
        """Add directed edge u->v with capacity and cost."""
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])
    
    # Add edges for each vertex: v_in -> v_out with cap 1, cost 1
    for v in vertices:
        i = v_to_idx[v]
        add_edge(2 * i, 2 * i + 1, 1, 1)
    
    # Add edges from parent_out to child_in with infinite capacity, cost 0
    INF_CAP = 10**6
    for v in vertices:
        i = v_to_idx[v]
        for p in parents.get(v, []):
            pi = v_to_idx[p]
            add_edge(2 * pi + 1, 2 * i, INF_CAP, 0)
    
    # Connect source to input vertices' v_in nodes (infinite cap, cost 0)
    # These are free - inputs start blue
    for v in inputs:
        i = v_to_idx[v]
        add_edge(source, 2 * i, INF_CAP, 0)
    
    # Connect output vertices' v_out to sink (cap 1, cost 0)
    for v in outputs:
        i = v_to_idx[v]
        add_edge(2 * i + 1, sink, 1, 0)
    
    # Now run min-cost max-flow
    # We want to send 1 unit of flow to each output
    target_flow = len(outputs)
    
    # Bellman-Ford based min-cost flow (SPFA for efficiency)
    def min_cost_flow():
        """Compute min-cost max-flow from source to sink."""
        flow = 0
        cost = 0
        dist = [0] * total_nodes
        in_queue = [False] * total_nodes
        parent_node = [-1] * total_nodes
        parent_edge = [-1] * total_nodes
        
        while flow < target_flow:
            # SPFA to find shortest path
            dist = [float('inf')] * total_nodes
            dist[source] = 0
            in_queue = [False] * total_nodes
            parent_node = [-1] * total_nodes
            parent_edge = [-1] * total_nodes
            
            queue = deque([source])
            in_queue[source] = True
            
            while queue:
                u = queue.popleft()
                in_queue[u] = False
                
                for edge_idx, (v, cap, c, rev_idx) in enumerate(adj[u]):
                    if cap > 0 and dist[u] + c < dist[v]:
                        dist[v] = dist[u] + c
                        parent_node[v] = u
                        parent_edge[v] = edge_idx
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
            
            if dist[sink] == float('inf'):
                break  # No more augmenting paths
            
            # Find bottleneck capacity along the path
            path_cap = float('inf')
            v = sink
            while v != source:
                u = parent_node[v]
                edge = adj[u][parent_edge[v]]
                path_cap = min(path_cap, edge[1])
                v = u
            
            if path_cap == 0:
                break
            
            # Augment flow
            v = sink
            while v != source:
                u = parent_node[v]
                edge_idx = parent_edge[v]
                adj[u][edge_idx][1] -= path_cap
                adj[v][adj[u][edge_idx][3]][1] += path_cap
                v = u
            
            flow += path_cap
            cost += path_cap * dist[sink]
        
        return flow, cost
    
    flow, total_cost = min_cost_flow()
    
    # Extract which vertices were loaded (have flow on v_in -> v_out edge)
    loaded_vertices = set()
    for v in vertices:
        i = v_to_idx[v]
        edge = adj[2 * i][0]  # First edge from v_in
        if edge[1] == 0:  # Capacity reduced means flow was sent
            loaded_vertices.add(v)
    
    # Now build a schedule that keeps loaded_vertices in fast memory as long as possible
    # Process in topological order, loading/unloading as needed to stay within budget
    # Strategy: 
    # 1. Load vertices in topological order
    # 2. When memory is full, evict vertices that are NOT in loaded_vertices first
    # 3. For loaded_vertices, try to keep them in memory until all their children are computed
    
    state = {}
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    ops = []
    computed = set()
    
    # Track remaining children for each vertex
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    # Determine which loaded vertices we can keep
    # We need to be smarter: process in order, and when we need to evict,
    # prefer evicting non-loaded vertices
    
    def evict_vertex(v):
        """Evict vertex v from fast memory."""
        fast_memory.discard(v)
        if v in outputs:
            ops.append(['store', v])
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            state[v] = 'void'
    
    def free_space_if_needed():
        """Free space in fast memory if needed."""
        if len(fast_memory) < fast_memory_budget:
            return True
        
        # First try to evict non-loaded vertices with no remaining children
        evictable_non_loaded = [v for v in fast_memory 
                               if v not in loaded_vertices 
                               and remaining_children[v] == 0]
        
        if evictable_non_loaded:
            # Sort by depth (evict deepest first)
            depth = {v: 0 for v in vertices}
            for v in order:
                if parents.get(v, []):
                    depth[v] = max(depth[p] for p in parents[v]) + 1
            
            evictable_non_loaded.sort(key=lambda v: -depth[v])
            for v in evictable_non_loaded:
                evict_vertex(v)
                if len(fast_memory) < fast_memory_budget:
                    return True
        
        # If still full, evict loaded vertices with no remaining children
        evictable_loaded = [v for v in fast_memory 
                           if v in loaded_vertices 
                           and remaining_children[v] == 0]
        
        if evictable_loaded:
            depth = {v: 0 for v in vertices}
            for v in order:
                if parents.get(v, []):
                    depth[v] = max(depth[p] for p in parents[v]) + 1
            
            evictable_loaded.sort(key=lambda v: -depth[v])
            for v in evictable_loaded:
                evict_vertex(v)
                if len(fast_memory) < fast_memory_budget:
                    return True
        
        return len(fast_memory) < fast_memory_budget
    
    def ensure_parents_in_memory(v):
        """Ensure all parents of v are in fast memory."""
        for p in parents.get(v, []):
            if state[p] != 'red':
                if state[p] == 'blue':
                    if not free_space_if_needed():
                        return False
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
                elif state[p] == 'void':
                    return False
        return True
    
    # Process vertices in topological order
    for v in order:
        if v in inputs:
            computed.add(v)
            continue
        
        if not ensure_parents_in_memory(v):
            continue
        
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        computed.add(v)
        
        for p in parents.get(v, []):
            remaining_children[p] -= 1
        
        # If v is an output and no more children, store it
        if v in outputs and remaining_children[v] == 0:
            evict_vertex(v)
        
        if len(fast_memory) >= fast_memory_budget:
            free_space_if_needed()
    
    # Final cleanup
    for v in list(fast_memory):
        evict_vertex(v)
    
    return ops

def _improved_pebbling(dag, fast_memory_budget):
    """
    Improved pebbling strategy with better memory management.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order, children_map, outputs_set, inputs = _build_graph_info(dag)
    
    # Compute depth for priority
    depth = {v: 0 for v in vertices}
    for v in order:
        if parents.get(v, []):
            max_parent_depth = max(depth[p] for p in parents[v])
            depth[v] = max_parent_depth + 1
    
    # State tracking
    state = {}
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    ops = []
    computed = set()
    
    # Track remaining children count for each vertex
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    # Track how many times each vertex is needed (for eviction priority)
    usage_count = {v: 0 for v in vertices}
    for v in vertices:
        for c in children_map[v]:
            usage_count[v] += 1
    
    def get_eviction_priority(v):
        """Get eviction priority for vertex v (lower = evict first)."""
        is_output = 1 if v in outputs else 0
        remaining_uses = remaining_children[v]
        return (is_output, remaining_uses, depth[v])
    
    def evict_vertex(v):
        """Evict vertex v from fast memory."""
        fast_memory.discard(v)
        if v in outputs:
            ops.append(['store', v])
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            state[v] = 'void'
    
    def free_space_if_needed():
        """Free space in fast memory if needed."""
        if len(fast_memory) < fast_memory_budget:
            return True
        
        evictable = [v for v in fast_memory if remaining_children[v] == 0]
        
        if not evictable:
            return len(fast_memory) < fast_memory_budget
        
        evictable.sort(key=get_eviction_priority)
        
        for v in evictable:
            evict_vertex(v)
            if len(fast_memory) < fast_memory_budget:
                return True
        
        return len(fast_memory) < fast_memory_budget
    
    def ensure_parents_in_memory(v):
        """Ensure all parents of v are in fast memory. Returns True if successful."""
        for p in parents.get(v, []):
            if state[p] != 'red':
                if state[p] == 'blue':
                    if not free_space_if_needed():
                        return False
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
                elif state[p] == 'void':
                    return False
        return True
    
    for v in order:
        if v in inputs:
            computed.add(v)
            continue
        
        if not ensure_parents_in_memory(v):
            continue
        
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        computed.add(v)
        
        for p in parents.get(v, []):
            remaining_children[p] -= 1
        
        if v in outputs and remaining_children[v] == 0:
            evict_vertex(v)
        
        if len(fast_memory) >= fast_memory_budget:
            free_space_if_needed()
    
    for v in list(fast_memory):
        evict_vertex(v)
    
    return ops

def _validate_schedule(dag, ops, fast_memory_budget):
    """Validate that the schedule is correct and respects memory budget."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    state = {}
    for v in vertices:
        if not parents.get(v, []):
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    computed = set()
    
    for op, v in ops:
        if op == 'load':
            if state[v] != 'blue':
                return False
            if v in fast_memory:
                return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
        elif op == 'store':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'blue'
        elif op == 'delete':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'void'
        elif op == 'compute':
            if v in computed:
                return False
            for p in parents.get(v, []):
                if state.get(p) != 'red':
                    return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
        else:
            return False
    
    for v in outputs:
        if state.get(v) != 'blue':
            return False
    
    return True

def _count_transfers(ops):
    """Count the number of transfers (loads + stores) in the schedule."""
    transfers = 0
    for op, v in ops:
        if op in ('load', 'store'):
            transfers += 1
    return transfers

def _naive_schedule(dag, fast_memory_budget):
    """Simple naive schedule for comparison/fallback."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order, children_map, outputs_set, inputs = _build_graph_info(dag)
    
    state = {}
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    ops = []
    computed = set()
    
    for v in order:
        if v in inputs:
            computed.add(v)
            continue
        
        for p in parents.get(v, []):
            if state[p] != 'red':
                if state[p] == 'blue':
                    while len(fast_memory) >= fast_memory_budget:
                        evicted = False
                        for fm_v in list(fast_memory):
                            if fm_v not in outputs:
                                ops.append(['delete', fm_v])
                                fast_memory.discard(fm_v)
                                state[fm_v] = 'void'
                                evicted = True
                                break
                            elif fm_v in outputs and len(children_map[fm_v]) == 0:
                                ops.append(['store', fm_v])
                                fast_memory.discard(fm_v)
                                state[fm_v] = 'blue'
                                evicted = True
                                break
                        if not evicted:
                            break
                    
                    if len(fast_memory) < fast_memory_budget:
                        ops.append(['load', p])
                        fast_memory.add(p)
                        state[p] = 'red'
        
        if all(state.get(p) == 'red' for p in parents.get(v, [])):
            ops.append(['compute', v])
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
            
            if v in outputs and len(children_map[v]) == 0:
                ops.append(['store', v])
                fast_memory.discard(v)
                state[v] = 'blue'
    
    for v in list(fast_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
        fast_memory.discard(v)
        if v in outputs:
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    try:
        # Try min-cost flow based schedule first
        schedule = _min_cost_flow_schedule(dag, fast_memory_budget)
        
        if _validate_schedule(dag, schedule, fast_memory_budget):
            if len(schedule) <= 20000:
                return schedule
        
        # Try the improved pebbling schedule
        schedule = _improved_pebbling(dag, fast_memory_budget)
        
        if _validate_schedule(dag, schedule, fast_memory_budget):
            if len(schedule) <= 20000:
                return schedule
        
        # Fallback to naive schedule
        schedule = _naive_schedule(dag, fast_memory_budget)
        
        if _validate_schedule(dag, schedule, fast_memory_budget):
            if len(schedule) <= 20000:
                return schedule
        
        return []
        
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
