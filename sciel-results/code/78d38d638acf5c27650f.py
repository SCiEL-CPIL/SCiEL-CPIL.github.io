"""Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG minimizing the number of transfers (load/store operations).
    
    Parameters:
        dag: dict with keys:
            - 'edges': list of (parent, child) pairs, or
            - 'adj': dict mapping vertex -> list of children
            - 'inputs': list of input vertices (start blue in slow memory)
            - 'outputs': list of output vertices (must finish blue in fast memory)
            - 'vertices': list of all vertices (optional, inferred if not present)
        fast_memory_budget: int, maximum number of red values in fast memory
    
    Returns:
        list of [operation, vertex] where operation is one of 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Parse DAG structure
    edges = dag.get('edges', [])
    if not edges and 'adj' in dag:
        adj = dag['adj']
        edges = [(u, v) for u, vs in adj.items() for v in vs]
    
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    all_vertices = set(dag.get('vertices', []))
    for u, v in edges:
        all_vertices.add(u)
        all_vertices.add(v)
    all_vertices |= inputs | outputs
    
    # Build adjacency: parents and children
    children = {v: [] for v in all_vertices}
    parents = {v: [] for v in all_vertices}
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)
    
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(parents[v]) for v in all_vertices}
    topo_order = []
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(all_vertices):
        remaining = all_vertices - set(topo_order)
        topo_order.extend(sorted(remaining))
        
    # Precompute topological index for eviction heuristics
    topo_index = {v: i for i, v in enumerate(topo_order)}
    
    # State
    in_fast = set()
    in_slow = set(inputs) # Only inputs start in slow memory. Others are unknown until computed/stored.
    computed = set()
    operations = []
    
    # Count how many times each vertex is needed as a parent (for eviction heuristic)
    # We can approximate this by counting children, but better to look ahead in topo order?
    # For simplicity, we'll use a "last use" heuristic based on topo order of children.
    
    def get_last_use(v):
        """Estimate when v is last needed. 
        If v is an output, it's needed at the end.
        Otherwise, it's needed until its last child is computed.
        """
        if v in outputs:
            return float('inf')
        if not children[v]:
            return topo_index.get(v, 0)
        max_child_idx = max((topo_index.get(c, 0) for c in children[v]), default=0)
        return max_child_idx

    def evict_if_needed(required_set=None):
        """Evict nodes from fast memory to make room.
        required_set: set of nodes that MUST stay in fast memory (ancestors of current target).
        """
        while len(in_fast) > fast_memory_budget:
            # Find candidates to evict
            # Prefer evicting nodes not in required_set
            candidates = [x for x in in_fast if x not in (required_set if required_set else set())]
            
            if not candidates:
                # If we must evict from required set, we are in trouble (budget too small for depth)
                # But let's try to evict the one with the farthest last use
                candidates = list(in_fast)
                
            if not candidates:
                break
                
            # Heuristic: Evict the node with the largest last_use index (least likely to be needed soon)
            # Among ties, prefer non-inputs
            best_to_evict = None
            best_score = -1
            
            for c in candidates:
                last_use = get_last_use(c)
                # We want to keep nodes with small last_use (needed soon)
                # So we evict the one with largest last_use
                # If last_use is inf (output), we should avoid evicting unless absolutely necessary
                if c in outputs:
                    score = -1000000 # Penalize evicting outputs
                else:
                    score = last_use
                    
                # Tie-breaker: prefer evicting non-inputs
                if c in inputs:
                    score -= 0.5
                    
                if score > best_score:
                    best_score = score
                    best_to_evict = c
                    
            if best_to_evict is not None:
                operations.append(['store', best_to_evict])
                in_fast.discard(best_to_evict)
                in_slow.add(best_to_evict)
            else:
                # Should not happen if budget > 0
                break

    def ensure_available(v):
        """Ensure vertex v is in fast memory. Returns True if successful."""
        if time.time() > deadline:
            return False
            
        if v in in_fast:
            return True
            
        if v in inputs:
            # Load from slow memory
            if v in in_slow:
                evict_if_needed({v}) # Make room for v
                if len(in_fast) >= fast_memory_budget:
                    # If still full after evicting, we can't load. 
                    # This implies budget < 1 or logic error. 
                    # But evict_if_needed should have made room if budget > 0.
                    pass
                operations.append(['load', v])
                in_slow.discard(v)
                in_fast.add(v)
                return True
            else:
                # Input should always be in slow memory initially. 
                # If not, it's an error in state tracking.
                return False
                
        # It's a computed node
        if v in in_slow:
            # Load from slow memory
            evict_if_needed({v})
            operations.append(['load', v])
            in_slow.discard(v)
            in_fast.add(v)
            return True
            
        # Not in fast, not in slow. Must recompute.
        # Check if already computed (value exists, but we lost it). 
        # In pebbling, if we didn't store it, we must recompute.
        # Ensure all parents are available
        for p in parents[v]:
            if not ensure_available(p):
                return False
                
        # Compute v
        evict_if_needed({v})
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        return True

    # Process all vertices in topological order to ensure all are computed/stored if needed?
    # Actually, we only need to ensure OUTPUTS are in fast memory at the end.
    # But intermediate nodes might need to be kept if they are shared.
    # The recursive ensure_available handles dependencies.
    
    # We need to "drive" the computation. 
    # We can just call ensure_available for all outputs.
    # But what about nodes that are not ancestors of any output? 
    # The problem says "all outputs must finish blue". It doesn't say all vertices must be computed.
    # However, usually in these tasks, the DAG represents a computation where all leaves are inputs and we want specific outputs.
    # If a vertex is not an ancestor of any output, it doesn't need to be computed.
    
    # Identify ancestors of outputs
    ancestors_of_outputs = set()
    stack = list(outputs)
    while stack:
        v = stack.pop()
        if v in ancestors_of_outputs:
            continue
        ancestors_of_outputs.add(v)
        for p in parents[v]:
            stack.append(p)
            
    # We need to ensure all ancestors are computed/available? 
    # No, ensure_available(v) will recursively handle parents.
    # So we just need to call ensure_available for each output.
    
    # However, there is a subtle point: if we compute a node, evict it, and later need it again, 
    # ensure_available will recompute it. This is correct.
    
    # To minimize operations, we might want to compute nodes in an order that keeps them in cache.
    # But the recursive approach is safe and simple. 
    # Let's try to process outputs in topological order (reverse topo of dependencies?) 
    # Actually, just iterating outputs is fine.
    
    for out_v in outputs:
        if not ensure_available(out_v):
            # Failed to ensure output. 
            # This might happen if budget is too small for the depth.
            # As a fallback, try to force load if it was computed? No, if not in slow, it's gone.
            # We can only return what we have.
            pass

    # Ensure outputs are in fast memory (they should be from ensure_available)
    # Final check
    for out_v in outputs:
        if out_v not in in_fast:
            # If it's in slow, load it
            if out_v in in_slow:
                evict_if_needed({out_v})
                operations.append(['load', out_v])
                in_slow.discard(out_v)
                in_fast.add(out_v)
            # If not in fast and not in slow, we failed.

    if len(operations) > 20000:
        operations = operations[:20000]
        
    return operations
# EVOLVE-BLOCK-END
