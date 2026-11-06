"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Solve communication-minimizing DAG pebbling.
    
    Args:
        dag: dict with 'vertices', 'edges', 'inputs', 'outputs'
        fast_memory_budget: int, max red values in fast memory
    
    Returns:
        list of [operation, vertex] where operation is load/store/compute/delete
    """
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    # Parse DAG structure
    vertices = list(dag.get('vertices', range(len(dag.get('edges', {})))))
    if not vertices:
        # Try to infer from edges
        edges = dag.get('edges', {})
        all_verts = set()
        for v, parents in edges.items():
            all_verts.add(v)
            for p in parents:
                all_verts.add(p)
        vertices = sorted(all_verts) if all_verts else []
    
    if not vertices:
        return []
    
    # Build adjacency
    edges = dag.get('edges', {})
    parents = {v: list(edges.get(v, [])) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Topological order (Kahn's algorithm)
    in_degree = {v: len(parents[v]) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle or disconnected - just do naive
        return _naive_schedule(dag, fast_memory_budget)
    
    # Naive baseline for comparison
    naive_ops = _naive_schedule(dag, fast_memory_budget)
    
    # Try optimized pebbling
    optimized_ops = _optimized_pebbling(
        vertices, parents, children, inputs, outputs, 
        fast_memory_budget, topo_order
    )
    
    # Check deadline
    if time.time() > deadline:
        return naive_ops if len(naive_ops) < len(optimized_ops) else optimized_ops
    
    # Validate and return
    if _is_valid(optimized_ops, dag, fast_memory_budget):
        return optimized_ops
    else:
        return naive_ops


def _naive_schedule(dag, fast_memory_budget):
    """Simple topological order schedule."""
    vertices = list(dag.get('vertices', []))
    if not vertices:
        edges = dag.get('edges', {})
        all_verts = set()
        for v, parents in edges.items():
            all_verts.add(v)
            for p in parents:
                all_verts.add(p)
        vertices = sorted(all_verts)
    
    edges = dag.get('edges', {})
    parents = {v: list(edges.get(v, [])) for v in vertices}
    inputs = set(dag.get('inputs', []))
    
    ops = []
    for v in vertices:
        if v in inputs:
            continue  # inputs start blue
        for p in parents[v]:
            if p not in inputs:
                ops.append(['load', p])
        ops.append(['compute', v])
        for p in parents[v]:
            if p not in inputs:
                ops.append(['store', p])
    
    return ops


def _optimized_pebbling(vertices, parents, children, inputs, outputs, budget, topo_order):
    """
    Optimized pebbling using recursive strategy.
    For each vertex, decide whether to keep its value in fast memory or recompute.
    """
    if budget <= 0:
        return _naive_from_structure(vertices, parents, inputs, topo_order)
    
    # Identify layers for layered DAGs
    # Use a heuristic: process in topological order, but manage memory carefully
    
    ops = []
    fast_memory = set()  # vertices currently in fast memory (red)
    
    # Track which vertices have been computed and are in fast memory
    computed_in_fast = set()
    
    for v in topo_order:
        if v in inputs:
            # Input: load into fast memory if budget allows
            if v not in computed_in_fast and len(fast_memory) < budget:
                ops.append(['load', v])
                fast_memory.add(v)
                computed_in_fast.add(v)
            continue
        
        # Check if all parents are available
        parent_available = all(p in computed_in_fast for p in parents[v])
        
        if not parent_available:
            # Need to load missing parents
            for p in parents[v]:
                if p not in computed_in_fast:
                    if p in inputs:
                        # Load input
                        if len(fast_memory) >= budget:
                            # Evict least useful vertex
                            evict_v = _find_evict_target(fast_memory, children, topo_order, v, outputs)
                            if evict_v is not None and evict_v != v:
                                ops.append(['delete', evict_v])
                                fast_memory.discard(evict_v)
                                computed_in_fast.discard(evict_v)
                        ops.append(['load', p])
                        fast_memory.add(p)
                        computed_in_fast.add(p)
                    else:
                        # Recompute parent
                        recompute_ops = _recompute_vertex(p, parents, children, inputs, fast_memory, budget, ops)
                        ops.extend(recompute_ops)
                        fast_memory.add(p)
                        computed_in_fast.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_memory.add(v)
        computed_in_fast.add(v)
        
        # Optionally delete parents that are no longer needed
        for p in parents[v]:
            if p in computed_in_fast and p not in inputs:
                # Check if p is needed by other children
                remaining_children = [c for c in children[p] if c not in computed_in_fast and c != v]
                if not remaining_children and p not in outputs:
                    if len(fast_memory) > budget:
                        ops.append(['delete', p])
                        fast_memory.discard(p)
                        computed_in_fast.discard(p)
    
    # Ensure outputs are in fast memory at end
    for o in outputs:
        if o not in computed_in_fast:
            if o in inputs:
                ops.append(['load', o])
            else:
                recompute_ops = _recompute_vertex(o, parents, children, inputs, fast_memory, budget, ops)
                ops.extend(recompute_ops)
    
    return ops


def _recompute_vertex(v, parents, children, inputs, fast_memory, budget, existing_ops):
    """Recompute vertex v and its dependencies as needed."""
    ops = []
    # Simple: load all parents, compute v
    for p in parents[v]:
        if p not in fast_memory:
            if p in inputs:
                ops.append(['load', p])
            else:
                # Recursive recompute (simplified)
                sub_ops = _recompute_vertex(p, parents, children, inputs, fast_memory, budget, existing_ops)
                ops.extend(sub_ops)
    ops.append(['compute', v])
    return ops


def _find_evict_target(fast_memory, children, topo_order, current_v, outputs):
    """Find which vertex to evict from fast memory."""
    # Prefer evicting vertices that are inputs or have no remaining children
    for v in fast_memory:
        if v in outputs:
            continue  # Don't evict outputs
        remaining_children = [c for c in children[v] if c not in fast_memory]
        if not remaining_children:
            return v
    # Fallback: evict first non-output
    for v in fast_memory:
        if v not in outputs:
            return v
    return None


def _naive_from_structure(vertices, parents, inputs, topo_order):
    """Naive schedule from structure."""
    ops = []
    for v in topo_order:
        if v in inputs:
            continue
        for p in parents[v]:
            if p not in inputs:
                ops.append(['load', p])
        ops.append(['compute', v])
        for p in parents[v]:
            if p not in inputs:
                ops.append(['store', p])
    return ops


def _is_valid(ops, dag, budget):
    """Validate the operation sequence."""
    vertices = set(dag.get('vertices', []))
    if not vertices:
        edges = dag.get('edges', {})
        for v, ps in edges.items():
            vertices.add(v)
            vertices.update(ps)
    
    fast_memory = set()
    computed = set()
    
    for op, v in ops:
        if v not in vertices:
            return False
        if op == 'load':
            if len(fast_memory) >= budget:
                return False
            fast_memory.add(v)
        elif op == 'store':
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
        elif op == 'compute':
            # All parents must be in fast memory or inputs
            parents = dag.get('edges', {}).get(v, [])
            for p in parents:
                if p not in fast_memory:
                    return False
            if len(fast_memory) >= budget:
                return False
            fast_memory.add(v)
            computed.add(v)
        elif op == 'delete':
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
        else:
            return False
    
    return True
# EVOLVE-BLOCK-END

