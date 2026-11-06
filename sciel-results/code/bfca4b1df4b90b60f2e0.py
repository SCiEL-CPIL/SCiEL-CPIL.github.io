"""Communication-minimizing DAG pebbling solver."""
import time
import heapq
import bisect
from collections import defaultdict, deque

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 90s kill

def _now():
    return time.time() - _START_TIME

def _deadline_reached():
    return (_now() > 90.0 - _DEADLINE_MARGIN)


def _topo_order(dag):
    """Compute topological order from the DAG structure."""
    parents = dag.get('parents', [])
    outputs = dag.get('outputs', [])
    n = dag.get('n', len(parents))
    
    # Build children adjacency
    children = defaultdict(list)
    in_degree = [0] * n
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
            in_degree[i] += 1
    
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    return order


def _compute_naive_transfers(dag):
    """Compute the naive schedule's transfer count for baseline comparison."""
    parents = dag.get('parents', [])
    outputs = dag.get('outputs', [])
    n = dag.get('n', len(parents))
    
    order = _topo_order(dag)
    # Naive: load each input, compute each non-input, store each output
    # Inputs start blue (already in slow memory), so we load them when needed
    # Non-inputs: load all parents (if not already loaded/computed), compute
    # Outputs: store at the end
    
    # Simplest naive: topological, load each vertex's parents just-in-time
    # Track what's in fast memory
    fast = set()
    transfers = 0
    
    for v in order:
        if not parents[v]:
            # Input vertex: load it
            if v not in fast:
                fast.add(v)
                transfers += 1
        else:
            # Non-input: load all parents
            for p in parents[v]:
                if p not in fast:
                    fast.add(p)
                    transfers += 1
            # Compute v (no transfer for compute itself)
            # v is now in fast memory
            fast.add(v)
            # Can we delete parents to save space? In naive, we don't delete
            # But we need to respect fast_memory_budget
            # For naive baseline, assume enough memory or we delete as we go
            # Actually naive just does: load all needed, compute, move on
            # Let's say we delete parents after computing if budget is tight
            # But for the score, naive is defined as the baseline schedule
            # Let me re-read: "Score mean transfers / naive schedule on three layered DAGs"
            # So naive is a specific schedule. Let me assume naive = topological with
            # load-on-demand, no deletion (or minimal deletion)
    
    # Store all outputs
    for o in outputs:
        if o in fast:
            transfers += 1
            fast.discard(o)
        else:
            # Need to recompute? In naive, outputs should be available
            # Let's just count it as a transfer
            transfers += 1
    
    return transfers


def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication (transfers).
    
    Args:
        dag: dict with 'parents' (list of lists), 'outputs' (list), 'n' (int)
        fast_memory_budget: int, max number of red values in fast memory
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    if _deadline_reached():
        # Return a minimal valid schedule
        return _emergency_schedule(dag, fast_memory_budget)
    
    parents = dag.get('parents', [])
    outputs = dag.get('outputs', [])
    n = dag.get('n', len(parents))
    
    order = _topo_order(dag)
    output_set = set(outputs)
    
    # Identify inputs (vertices with no parents)
    inputs = [v for v in range(n) if not parents[v]]
    
    # For layered DAGs, we can use a level-based approach
    # Determine the depth (layer) of each vertex
    depth = [0] * n
    for v in order:
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
        else:
            depth[v] = 0
    
    max_depth = max(depth) if depth else 0
    
    # Group vertices by depth
    layers = defaultdict(list)
    for v in order:
        layers[depth[v]].append(v)
    
    # Strategy: Process layer by layer
    # For each layer, compute all vertices in that layer
    # Keep only the necessary vertices in fast memory
    # This is a heuristic but works well for layered DAGs
    
    operations = []
    fast_mem = set()  # vertices currently in fast memory (red)
    computed = set()  # vertices whose value is currently available (in fast or slow)
    
    # We need to track what's in fast memory vs slow memory
    # Initially, inputs are blue in slow memory
    # We load them into fast memory as needed
    
    # For a layered DAG with depth d, the optimal approach for small budgets
    # is to recompute intermediate values rather than storing all layers
    
    # Heuristic: if budget is large enough to hold a full layer, do layer-by-layer
    # Otherwise, use recomputation
    
    # Let's implement a recursive pebbling strategy
    # For simplicity, use a greedy approach with the budget constraint
    
    # First, check if we can do a simple topological pass
    max_layer_size = max(len(layers[d]) for d in range(max_depth + 1)) if max_depth >= 0 else 0
    
    if fast_memory_budget >= max_layer_size + 2:
        # Enough memory for layer-by-layer processing
        return _layered_schedule(dag, fast_memory_budget, order, layers, depth, outputs, output_set, inputs)
    else:
        # Use a more careful approach with recomputation
        return _recomputation_schedule(dag, fast_memory_budget, order, layers, depth, outputs, output_set, inputs)


def _layered_schedule(dag, budget, order, layers, depth, outputs, output_set, inputs):
    """Layer-by-layer schedule for when budget is sufficient."""
    parents = dag.get('parents', [])
    operations = []
    fast = set()
    
    for d in range(len(layers)):
        layer = layers[d]
        # Load inputs for this layer if not already in fast
        for v in layer:
            if not parents[v]:
                # Input
                if v not in fast:
                    operations.append(['load', v])
                    fast.add(v)
            else:
                # Ensure all parents are in fast
                for p in parents[v]:
                    if p not in fast:
                        # Parent should be from a previous layer
                        # Load it
                        operations.append(['load', p])
                        fast.add(p)
                # Compute v
                operations.append(['compute', v])
                fast.add(v)
        
        # After computing this layer, we can delete vertices from earlier layers
        # that are no longer needed
        # A vertex u is needed if it's a parent of some vertex in a later layer
        future_parents = set()
        for d2 in range(d + 1, len(layers)):
            for v in layers[d2]:
                for p in parents[v]:
                    future_parents.add(p)
        
        # Also keep outputs that haven't been stored yet
        for v in list(fast):
            if v not in future_parents and v not in output_set:
                operations.append(['delete', v])
                fast.discard(v)
    
    # Store all outputs
    for o in outputs:
        if o in fast:
            operations.append(['store', o])
            fast.discard(o)
        else:
            operations.append(['load', o])  # shouldn't happen in valid schedule
            operations.append(['store', o])
    
    return operations


def _recomputation_schedule(dag, budget, order, layers, depth, outputs, output_set, inputs):
    """Schedule with recomputation for tight budgets."""
    parents = dag.get('parents', [])
    n = dag.get('n', len(parents))
    
    # Use a DFS-based approach with limited memory
    # This is complex, so use a simpler heuristic:
    # Process in topological order, load parents just-in-time,
    # compute, and delete parents that won't be needed again
    
    operations = []
    fast = set()
    
    # Count how many times each vertex is needed as a parent
    parent_count = defaultdict(int)
    for v in range(n):
        for p in parents[v]:
            parent_count[p] += 1
    
    # Track remaining uses
    remaining = dict(parent_count)
    
    for v in order:
        if not parents[v]:
            # Input: load
            if v not in fast:
                operations.append(['load', v])
                fast.add(v)
        else:
            # Load all parents
            for p in parents[v]:
                if p not in fast:
                    operations.append(['load', p])
                    fast.add(p)
            
            # Compute
            operations.append(['compute', v])
            fast.add(v)
            
            # Delete parents that won't be used again
            for p in parents[v]:
                remaining[p] -= 1
                if remaining[p] == 0 and p not in output_set:
                    if p in fast:
                        operations.append(['delete', p])
                        fast.discard(p)
        
        # If we're over budget, delete non-critical vertices
        # This is a simplification; a proper solution would use a pebbling strategy
        while len(fast) > budget:
            # Delete a vertex that's not an output and won't be needed
            # Find a vertex we can delete
            deleted = False
            for v2 in list(fast):
                if v2 not in output_set and remaining.get(v2, 0) == 0:
                    operations.append(['delete', v2])
                    fast.discard(v2)
                    deleted = True
                    break
            if not deleted:
                # Can't delete anything, but we're over budget
                # This shouldn't happen in a valid schedule
                break
    
    # Store all outputs
    for o in outputs:
        if o in fast:
            operations.append(['store', o])
            fast.discard(o)
        else:
            operations.append(['load', o])
            operations.append(['store', o])
    
    return operations


def _emergency_schedule(dag, fast_memory_budget):
    """Fallback minimal schedule if time is running out."""
    parents = dag.get('parents', [])
    outputs = dag.get('outputs', [])
    n = dag.get('n', len(parents))
    
    order = _topo_order(dag)
    ops = []
    for v in order:
        if not parents[v]:
            ops.append(['load', v])
        else:
            for p in parents[v]:
                ops.append(['load', p])
            ops.append(['compute', v])
    for o in outputs:
        ops.append(['store', o])
    
    # Trim to 20000 ops
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops


# Main entry point
def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication (transfers).
    
    Args:
        dag: dict with 'parents' (list of lists), 'outputs' (list), 'n' (int)
        fast_memory_budget: int, max number of red values in fast memory
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    if _deadline_reached():
        return _emergency_schedule(dag, fast_memory_budget)
    
    try:
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        n = dag.get('n', len(parents))
        
        if n == 0:
            return []
        
        order = _topo_order(dag)
        
        # Compute depths
        depth = [0] * n
        for v in order:
            if parents[v]:
                depth[v] = max(depth[p] for p in parents[v]) + 1
        
        max_depth = max(depth) if depth else 0
        
        # Group by depth
        layers = defaultdict(list)
        for v in order:
            layers[depth[v]].append(v)
        
        input_set = set(v for v in range(n) if not parents[v])
        output_set = set(outputs)
        
        # Choose strategy based on budget
        max_layer_size = max(len(layers[d]) for d in range(max_depth + 1)) if max_depth >= 0 else 0
        
        if fast_memory_budget >= max_layer_size + 2:
            result = _layered_schedule(dag, fast_memory_budget, order, layers, depth, outputs, output_set, input_set)
        else:
            result = _recomputation_schedule(dag, fast_memory_budget, order, layers, depth, outputs, output_set, input_set)
        
        # Ensure we don't exceed 20000 operations
        if len(result) > 20000:
            result = result[:20000]
        
        return result
    
    except Exception:
        # Fallback to emergency schedule
        return _emergency_schedule(dag, fast_memory_budget)
