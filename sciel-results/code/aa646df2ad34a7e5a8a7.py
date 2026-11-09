"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, OrderedDict

_DEADLINE = 55.0  # Safety margin before 60s limit

def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        return parents.get(vertex, [])
    elif hasattr(dag, 'parents'):
        return dag.parents.get(vertex, []) if isinstance(dag.parents, dict) else list(dag.parents[vertex])
    else:
        return []

def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        outputs = dag.get('outputs', [])
        return list(outputs)
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    else:
        return []

def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        # Try to infer from parents
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    else:
        return []

def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    inputs = []
    for v in all_v:
        if not _get_parents(dag, v):
            inputs.append(v)
    return inputs

def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    parents = {}
    children = defaultdict(list)
    
    for v in all_v:
        ps = _get_parents(dag, v)
        parents[v] = ps
        for p in ps:
            children[p].append(v)
    
    # Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = [v for v in all_v if in_degree[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(all_v):
        # Cycle or error, just return all vertices
        return all_v
    
    return topo

def _compute_use_counts(dag):
    """Compute how many times each vertex's value is used."""
    all_v = _get_all_vertices(dag)
    use_count = defaultdict(int)
    
    for v in all_v:
        for p in _get_parents(dag, v):
            use_count[p] += 1
    
    # Outputs are also "used"
    for o in _get_outputs(dag):
        use_count[o] += 1
    
    return use_count

def _pebble_divide_conquer(dag, fast_memory_budget, deadline):
    """
    Recursive divide-and-conquer pebbling strategy.
    Returns a list of operations.
    """
    start_time = time.time()
    elapsed = start_time - deadline[0]
    
    if elapsed > _DEADLINE:
        return []
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    use_counts = _compute_use_counts(dag)
    
    # Strategy: Process in topological order with LRU cache management
    # Use an OrderedDict to track access order for LRU eviction
    
    operations = []
    in_fast_memory = OrderedDict()  # vertex -> last_access_time
    
    access_counter = 0
    
    def _evict_if_needed(needed_slots=1):
        """Evict least recently used items if we need space."""
        while len(in_fast_memory) + needed_slots > fast_memory_budget:
            if not in_fast_memory:
                return False
            # Evict least recently used (first item in OrderedDict)
            evicted_v, _ = in_fast_memory.popitem(last=False)
            operations.append(['delete', evicted_v])
        return True
    
    # Now process in topological order
    for v in topo:
        if time.time() - deadline[0] > _DEADLINE:
            break
        
        parents = _get_parents(dag, v)
        
        if not parents:
            # Input vertex - it starts blue
            # We need to load it if any of its children will use it
            # Load it when needed (when a child is computed)
            pass
        else:
            # Need to compute v
            # Ensure all parents are in fast memory
            for p in parents:
                if p not in in_fast_memory:
                    # Need to load p, ensure we have space
                    if not _evict_if_needed(1):
                        break
                    operations.append(['load', p])
                    in_fast_memory[p] = access_counter
                    access_counter += 1
                else:
                    # Refresh access time
                    in_fast_memory[p] = access_counter
                    access_counter += 1
            
            # Check if all parents are loaded
            if all(p in in_fast_memory for p in parents):
                # Compute v
                if not _evict_if_needed(1):
                    break
                operations.append(['compute', v])
                in_fast_memory[v] = access_counter
                access_counter += 1
                
                # Decide whether to delete v immediately
                # Keep if: high use count, or it's an output, or memory is not full
                should_delete = False
                if use_counts[v] <= 1 and v not in outputs:
                    should_delete = True
                elif len(in_fast_memory) >= fast_memory_budget and use_counts[v] <= 2:
                    should_delete = True
                
                if should_delete:
                    operations.append(['delete', v])
                    del in_fast_memory[v]
    
    # Phase 2: Ensure all outputs are in fast memory (blue)
    for o in outputs:
        if o not in in_fast_memory:
            if not _evict_if_needed(1):
                break
            operations.append(['load', o])
            in_fast_memory[o] = access_counter
            access_counter += 1
    
    # Store all outputs
    for o in outputs:
        if o in in_fast_memory:
            operations.append(['store', o])
            del in_fast_memory[o]
    
    # Clean up remaining fast memory (optional, for tidiness)
    while in_fast_memory:
        v, _ = in_fast_memory.popitem(last=False)
        operations.append(['delete', v])
    
    return operations

def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    
    # Check that all outputs end up blue (stored or in fast memory)
    outputs = _get_outputs(dag)
    for o in outputs:
        # Outputs must be stored (removed from fast memory) or we need to check
        # The requirement is "all outputs must finish blue"
        # If we store them, they're in slow memory as blue - that should be fine
        pass
    
    return True

def _count_transfers(operations):
    """Count load and store operations."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _naive_schedule_transfers(dag, fast_memory_budget):
    """Estimate naive schedule transfers."""
    all_v = _get_all_vertices(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    use_counts = _compute_use_counts(dag)
    
    # Naive: For each vertex computation, load all parents that aren't cached
    # Simple estimate: each vertex's parents need to be loaded once
    # Plus loading inputs and storing outputs
    transfers = len(inputs) + len(outputs)
    
    # Each non-input vertex needs its parents loaded
    for v in all_v:
        if v not in inputs:
            parents = _get_parents(dag, v)
            transfers += len(parents)
    
    return transfers

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    # Try divide-and-conquer approach
    try:
        ops = _pebble_divide_conquer(dag, fast_memory_budget, deadline)
        
        if not ops:
            # Fallback to baseline
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        
        # Validate
        if _validate_operations(ops, dag, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # If invalid or too long, try simpler approach
        ops_simple = []
        all_v = _get_all_vertices(dag)
        topo = _topological_order(dag)
        inputs = _get_inputs(dag)
        outputs = _get_outputs(dag)
        
        in_fast = set()
        
        for v in topo:
            if time.time() - deadline[0] > _DEADLINE:
                break
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Evict something (simple: remove first added)
                        evict_target = None
                        for item in in_fast:
                            if item not in outputs:
                                evict_target = item
                                break
                        if evict_target:
                            ops_simple.append(['delete', evict_target])
                            in_fast.remove(evict_target)
                        else:
                            break
                    ops_simple.append(['load', p])
                    in_fast.add(p)
            ops_simple.append(['compute', v])
            in_fast.add(v)
        
        for o in outputs:
            if o not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_target = None
                    for item in in_fast:
                        if item not in outputs:
                            evict_target = item
                            break
                    if evict_target:
                        ops_simple.append(['delete', evict_target])
                        in_fast.remove(evict_target)
                    else:
                        break
                ops_simple.append(['load', o])
                in_fast.add(o)
            ops_simple.append(['store', o])
            in_fast.remove(o)
        
        if _validate_operations(ops_simple, dag, fast_memory_budget):
            if len(ops_simple) <= 20000:
                return ops_simple
        
    except Exception:
        pass
    
    # Final fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
