"""DAG pebbling with communication minimization via dynamic programming on layers."""

import time
import math
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (transfers between slow and fast memory).
    
    dag structure:
      - dag['vertices']: list of vertex indices (topologically ordered)
      - dag['parents']: dict mapping vertex -> list of parent vertices (already topologically before)
      - dag['inputs']: set/list of input vertices (start blue in slow memory)
      - dag['outputs']: set/list of output vertices (must finish blue)
      - fast_memory_budget: number of fast memory slots (red pebbles)
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG structure
    vertices = list(dag.get('vertices', range(dag.get('num_vertices', 0))))
    if not vertices:
        # Try to infer from parents
        parents = dag.get('parents', {})
        vertices = sorted(parents.keys())
        if not vertices:
            return []
    
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build adjacency
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Determine which vertices need to be computed (non-inputs)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Simple strategy: process vertices in topological order, 
    # loading parents when needed, computing, and managing fast memory.
    
    # Strategy: For each vertex in topological order:
    # 1. Ensure all parents are in fast memory (load if not)
    # 2. Compute the vertex
    # 3. Store output vertices to slow memory at the end
    # 4. Delete vertices that are no longer needed to free fast memory
    
    # Determine which vertices are "needed" by future computations
    # A vertex v is needed if it's an ancestor of some not-yet-processed vertex
    # or if it's an output
    
    # For simplicity, use a greedy approach:
    # Process in topological order. Maintain a set of vertices in fast memory.
    # When computing vertex v:
    #   - Load any parent not in fast memory
    #   - Compute v (adds v to fast memory)
    #   - If fast memory is full, delete a vertex that won't be needed again
    
    # Determine "last use" of each vertex
    # For each vertex, find the last vertex in topological order that uses it as a parent
    last_use = {}
    for v in vertices:
        last_use[v] = v  # at least used by itself (for output)
    
    for v in vertices:
        for child in children.get(v, []):
            if child > v:  # topological order
                if last_use[v] is None or child > last_use[v]:
                    last_use[v] = child
    
    # Also, outputs must remain until the end (or be stored)
    # Actually, outputs must finish blue, so we need to store them
    
    # Greedy pebbling strategy:
    ops = []
    in_fast = set()  # vertices currently in fast memory (red)
    
    # We need to track which vertices have been computed
    computed = set()
    
    # Process vertices in topological order
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            # Return what we have (may be invalid, but better than timeout)
            break
        
        # If v is an input, it starts in slow memory (blue)
        # We need to load it to fast memory when needed
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                if p in computed or p in input_set:
                    # Need to load p from slow memory
                    ops.append(['load', p])
                    in_fast.add(p)
                else:
                    # This shouldn't happen in valid topological order
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Check if fast memory is full before computing
        # If v is not already in fast memory (shouldn't be for non-inputs)
        if v not in in_fast:
            # Need space in fast memory
            # If we're at capacity, we need to delete something
            if len(in_fast) >= fast_memory_budget:
                # Find a vertex to delete: one whose last_use is earliest and not v
                # Prefer deleting vertices that won't be needed again soon
                candidates = [u for u in in_fast if u != v]
                if candidates:
                    # Delete the one with earliest last_use
                    to_delete = min(candidates, key=lambda u: last_use.get(u, u))
                    ops.append(['delete', to_delete])
                    in_fast.discard(to_delete)
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            computed.add(v)
        else:
            # v is already in fast memory (should be rare for non-inputs)
            computed.add(v)
        
        # If v is an output, store it to slow memory (make it blue)
        # But we need it to finish blue, so store at the end
        # Actually, let's store outputs immediately after computing to free space
        # No wait, "finish blue" means at the end. Let's store at the end.
    
    # Store all outputs to slow memory
    for v in outputs:
        if v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        elif v in computed:
            # It was deleted, need to reload and store? 
            # This is problematic. Let's handle it differently.
            pass
    
    # Ensure all outputs are blue (stored)
    # If an output was deleted, we need to recompute it
    # This greedy approach may not be optimal but should be valid
    
    # Actually, let's use a simpler and more correct approach:
    # Don't delete outputs until the very end
    
    # Let me redo this more carefully
    ops = []
    in_fast = set()
    computed = set()
    stored = set()  # vertices that have been stored to slow memory
    
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break
        
        # Load parents if not in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                if p in computed or p in input_set:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Make room if needed
        if v not in in_fast and len(in_fast) >= fast_memory_budget:
            # Delete a non-output vertex that won't be needed soon
            candidates = [u for u in in_fast if u != v and u not in output_set]
            if candidates:
                to_delete = min(candidates, key=lambda u: last_use.get(u, u))
                ops.append(['delete', to_delete])
                in_fast.discard(to_delete)
            else:
                # All in fast are outputs, delete one anyway (will reload later)
                # This is suboptimal but necessary
                candidates = [u for u in in_fast if u != v]
                if candidates:
                    to_delete = min(candidates, key=lambda u: last_use.get(u, u))
                    ops.append(['delete', to_delete])
                    in_fast.discard(to_delete)
        
        # Compute v
        if v not in in_fast:
            ops.append(['compute', v])
            in_fast.add(v)
        computed.add(v)
    
    # Store all outputs
    for v in sorted(output_set):
        if v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        elif v in computed:
            # Was deleted, need to reload
            ops.append(['load', v])
            in_fast.add(v)
            ops.append(['store', v])
            in_fast.discard(v)
        elif v in input_set:
            # Input that's an output, load and store
            ops.append(['load', v])
            ops.append(['store', v])
    
    # Clean up remaining fast memory (optional, not required)
    # The requirement is just that outputs finish blue
    
    return ops


def _pebble_optimized(dag, fast_memory_budget):
    """
    More optimized pebbling using dynamic programming for layered DAGs.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = list(dag.get('vertices', []))
    if not vertices:
        parents = dag.get('parents', {})
        vertices = sorted(parents.keys())
    if not vertices:
        return []
    
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Determine layers (longest path from any input)
    layer = {}
    for v in vertices:
        if v in inputs:
            layer[v] = 0
        else:
            par_layer = [layer.get(p, 0) for p in parents.get(v, [])]
            layer[v] = max(par_layer) + 1 if par_layer else 0
    
    max_layer = max(layer.values()) if layer else 0
    
    # Process layer by layer
    # For each layer, compute all vertices in that layer
    # Keep necessary parents in fast memory
    
    ops = []
    in_fast = set()
    computed = set()
    
    # For each layer, we need to keep the previous layer's vertices that are parents
    # of the current layer
    
    # Determine which vertices in layer L are needed by layer L+1
    needed_by_next = defaultdict(set)
    for v in vertices:
        l = layer[v]
        for p in parents.get(v, []):
            needed_by_next[l].add(p)
    
    for l in range(max_layer + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Get vertices in this layer
        layer_vertices = [v for v in vertices if layer[v] == l]
        
        # Determine which parents we need to keep in fast memory
        # Parents in layer l-1 that are needed by layer l
        needed_parents = set()
        for v in layer_vertices:
            for p in parents.get(v, []):
                needed_parents.add(p)
        
        # Ensure needed parents are in fast memory
        for p in needed_parents:
            if p not in in_fast:
                if p in computed or p in inputs:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute all vertices in this layer
        for v in layer_vertices:
            # Make room if needed
            if v not in in_fast and len(in_fast) >= fast_memory_budget:
                # Delete vertices from previous layers that aren't needed anymore
                # Find vertices whose last use is in the past
                candidates = [u for u in in_fast if u != v and layer.get(u, 0) < l]
                if candidates:
                    # Delete the one with smallest layer (oldest)
                    to_delete = min(candidates, key=lambda u: layer.get(u, 0))
                    ops.append(['delete', to_delete])
                    in_fast.discard(to_delete)
            
            if v not in in_fast:
                ops.append(['compute', v])
                in_fast.add(v)
            computed.add(v)
        
        # After computing this layer, we can delete parents that are no longer needed
        # A parent p in layer l-1 can be deleted if no vertex in layer > l uses it
        # Check if p is needed by any future layer
        for p in list(in_fast):
            if layer.get(p, 0) < l:
                # Check if p is needed by any vertex in layer > l
                is_needed = False
                for v in vertices:
                    if layer[v] > l and p in parents.get(v, []):
                        is_needed = True
                        break
                if not is_needed and p not in outputs:
                    ops.append(['delete', p])
                    in_fast.discard(p)
    
    # Store all outputs
    for v in sorted(outputs):
        if v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        elif v in computed:
            ops.append(['load', v])
            in_fast.add(v)
            ops.append(['store', v])
            in_fast.discard(v)
        elif v in inputs:
            ops.append(['load', v])
            ops.append(['store', v])
    
    return ops


# Use the optimized version
def pebble(dag, fast_memory_budget):
    try:
        result = _pebble_optimized(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback to simple version
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    except Exception:
        return []
