"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    
    dag structure (assumed):
      - dag['vertices'] or dag['nodes']: list of vertex ids or range
      - dag['parents']: dict or list mapping vertex -> list of parent vertices
      - dag['inputs']: set/list of input vertices (start blue)
      - dag['outputs']: set/list of output vertices (must finish blue)
      - Or similar structure
      
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Parse DAG structure - try common formats
    if isinstance(dag, dict):
        # Try to extract components
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        # Maybe dag is a tuple or has attributes
        vertices = None
        parents = None
        inputs = None
        outputs = None
    
    # If we can't parse, fall back to baseline
    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            # Return empty schedule as last resort
            return []
    
    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)
    
    num_vertices = len(vertex_list)
    
    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}
    
    # Ensure all vertices have parent entries
    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []
    
    # Normalize inputs and outputs
    if inputs is None:
        # Inputs are vertices with no parents
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)
    
    if outputs is None:
        # Outputs are vertices with no children
        # Build children map
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)
    
    # Build children map for future usage analysis
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)
    
    # Topological order (assumed vertices are already topologically ordered, 
    # but let's verify and get proper order)
    # Since parents should come before children in topological order,
    # and the problem says "topologically ordered parents", we assume vertex_list
    # is in topological order.
    
    # Determine which vertices are needed (those that are ancestors of outputs or are outputs)
    # Actually, we need to compute all vertices that are on paths to outputs
    needed = set(outputs)
    # Do reverse topological traversal from outputs
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)
    
    # Filter to only needed vertices
    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    
    # For each vertex, determine when it's last needed (last child that needs it)
    last_use = {}
    for v in needed_vertices:
        last_use[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_use[v] = max(last_use[v], vertex_list.index(child) if child in vertex_list else -1)
    
    # Simpler: since we process in topological order, track last child index
    # Rebuild with indices
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    
    last_child_idx = {}
    for v in needed_vertices:
        last_child_idx[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_child_idx[v] = max(last_child_idx[v], vertex_to_idx.get(child, -1))
    
    # Greedy pebbling algorithm
    # State:
    #   in_fast_memory: set of vertices currently in fast memory
    #   computed: set of vertices that have been computed (value exists somewhere)
    #   operations: list of (op, vertex)
    
    in_fast = set()
    computed = set()
    operations = []
    
    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
    
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
    
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        # After compute, result is in fast memory (newly computed)
        in_fast.add(v)
    
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)
    
    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break
        
        # Check if all parents are available
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        
        # Ensure parents are in fast memory or can be loaded
        for p in parents:
            if p in in_fast:
                continue
            # Need to load parent
            # Check if we have space
            if len(in_fast) >= fast_memory_budget:
                # Need to evict something
                # Find a value that is no longer needed (last_child_idx < current index)
                evicted = False
                for u in list(in_fast):
                    # Don't evict v or its parents
                    if u == v:
                        continue
                    if u in parents:
                        continue
                    # Check if u is still needed by future vertices
                    # u is needed if there's a future vertex that has u as parent
                    # and that future vertex is in needed_set
                    u_last = last_child_idx.get(u, -1)
                    v_idx = vertex_to_idx.get(v, -1)
                    if u_last < v_idx:
                        # u is not needed anymore
                        store(u)
                        computed.discard(u)  # value is stored, not in fast
                        evicted = True
                        break
                if not evicted:
                    # All values are still needed, evict the one with latest last_use
                    # to maximize flexibility
                    best_evict = None
                    best_last = -1
                    for u in list(in_fast):
                        if u == v or u in parents:
                            continue
                        u_last = last_child_idx.get(u, -1)
                        if u_last > best_last:
                            best_last = u_last
                            best_evict = u
                    if best_evict is not None:
                        store(best_evict)
                        computed.discard(best_evict)
                    else:
                        # Shouldn't happen, but handle gracefully
                        pass
            
            # Now load parent
            load(p)
        
        # Compute v
        compute(v)
        
        # After computing, if v is not an output and not needed by future, 
        # we could store it immediately, but let's keep it if space allows
        
        # If v is an input, it was already loaded, so compute might not be needed?
        # Actually inputs are pre-computed (start blue), so we just load them
        # But in our model, we load them when needed
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if v in computed:
                load(v)
            # If v was never computed, that's a problem, but shouldn't happen
    
    # Convert operations to proper format
    result = []
    for op, v in operations:
        result.append([op, v])
    
    return result

# EVOLVE-BLOCK-END
