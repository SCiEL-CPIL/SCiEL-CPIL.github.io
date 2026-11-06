"""Communication-minimizing DAG pebbling with deadline guard and improved heuristics."""
import time
import heapq
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _topo_sort(dag):
    """Return vertices in topological order (parents before children)."""
    # dag: dict vertex -> list of parent vertices (already topologically ordered per contract)
    # We assume parents are listed before children in the input structure.
    # Build a simple topological sort using Kahn's algorithm on the reverse graph
    # or just use the given order if parents are guaranteed to come first.
    
    # According to the contract: "dag has topologically ordered parents"
    # This likely means that for each vertex, its parents appear earlier in some ordering.
    # Let's assume dag is a dict: vertex -> list of parents.
    # We need to find a topological ordering.
    
    vertices = list(dag.keys())
    # Build in-degree based on children
    children = {v: [] for v in vertices}
    in_degree = {v: 0 for v in vertices}
    
    for v, parents in dag.items():
        for p in parents:
            if p not in children:
                children[p] = []
            children[p].append(v)
            in_degree[v] = in_degree.get(v, 0) + 1
    
    # Ensure all parents are in vertices
    for v, parents in dag.items():
        for p in parents:
            if p not in in_degree:
                in_degree[p] = 0
                children[p] = []
    
    # Kahn's algorithm
    queue = deque([v for v in in_degree if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If we didn't get all vertices, add remaining (cycle handling - shouldn't happen)
    if len(order) != len(in_degree):
        remaining = set(in_degree.keys()) - set(order)
        order.extend(sorted(remaining))
    
    return order


def _identify_inputs_outputs(dag, topo_order):
    """Identify input vertices (no parents) and output vertices (no children)."""
    parents_set = set()
    for v, parents in dag.items():
        for p in parents:
            parents_set.add(p)
    
    children_set = set()
    for v, parents in dag.items():
        for p in parents:
            children_set.add(v)
    
    inputs = [v for v in topo_order if v not in parents_set]
    outputs = [v for v in topo_order if v not in children_set]
    
    return inputs, outputs


def _naive_schedule(dag, fast_memory_budget):
    """Naive schedule: load each input, compute each vertex in topo order, store outputs.
    This serves as the baseline for comparison."""
    topo_order = _topo_sort(dag)
    inputs, outputs = _identify_inputs_outputs(dag, topo_order)
    
    ops = []
    # Load all inputs
    for v in inputs:
        ops.append(('load', v))
    
    # Compute all vertices in topological order (skip inputs)
    input_set = set(inputs)
    for v in topo_order:
        if v in input_set:
            continue
        # Check if all parents are available (they should be in fast memory or just computed)
        ops.append(('compute', v))
    
    # Store all outputs
    for v in outputs:
        ops.append(('store', v))
    
    # Count transfers (load + store operations)
    transfers = sum(1 for op, v in ops if op in ('load', 'store'))
    
    return ops, transfers


def _count_transfers(ops):
    """Count the number of load and store operations."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _validate_ops(ops, dag, fast_memory_budget, max_ops=20000):
    """Validate that the operation sequence is correct."""
    if len(ops) > max_ops:
        return False
    
    # State: fast_memory (set of vertices in fast memory), slow_memory (set of vertices in slow memory)
    fast_mem = set()
    slow_mem = set()
    
    topo_order = _topo_sort(dag)
    inputs, outputs = _identify_inputs_outputs(dag, topo_order)
    input_set = set(inputs)
    
    # Initially, all inputs are blue (in slow memory)
    for v in inputs:
        slow_mem.add(v)
    
    for op, v in ops:
        if op == 'load':
            # Vertex must be in slow memory and not in fast memory
            if v not in slow_mem or v in fast_mem:
                return False
            # Fast memory must have space
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
            slow_mem.discard(v)
        elif op == 'store':
            # Vertex must be in fast memory and not in slow memory
            if v not in fast_mem or v in slow_mem:
                return False
            fast_mem.discard(v)
            slow_mem.add(v)
        elif op == 'compute':
            # Vertex must not be an input
            if v in input_set:
                return False
            # All parents must be in fast memory
            for p in dag.get(v, []):
                if p not in fast_mem:
                    return False
            # Vertex must not already be in fast memory
            if v in fast_mem:
                return False
            # Fast memory must have space for the new result
            if len(fast_mem) >= fast_memory_budget:
                return False
            fast_mem.add(v)
        elif op == 'delete':
            # Vertex must be in fast memory
            if v not in fast_mem:
                return False
            fast_mem.discard(v)
        else:
            return False
    
    # All outputs must be in slow memory (blue) at the end
    for v in outputs:
        if v not in slow_mem:
            return False
    
    return True


def _greedy_pebble(dag, fast_memory_budget, deadline=None):
    """Greedy pebbling algorithm that tries to minimize transfers.
    
    Strategy: Process vertices in topological order, but delete vertices from fast memory
    as soon as they are no longer needed, to make room for new computations.
    """
    start_time = time.time()
    topo_order = _topo_sort(dag)
    inputs, outputs = _identify_inputs_outputs(dag, topo_order)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Build reverse adjacency (children)
    children = {v: [] for v in topo_order}
    for v, parents in dag.items():
        for p in parents:
            if p in children:
                children[p].append(v)
    
    # Count how many children each vertex has remaining
    remaining_children = {v: len(children.get(v, [])) for v in topo_order}
    
    # State
    fast_mem = set()
    slow_mem = set(inputs)  # Inputs start in slow memory
    computed = set()  # Vertices that have been computed
    ops = []
    
    # Load all inputs initially
    for v in inputs:
        if deadline is not None and time.time() - start_time > deadline:
            break
        ops.append(('load', v))
        fast_mem.add(v)
        slow_mem.discard(v)
    
    # Process vertices in topological order
    for v in topo_order:
        if deadline is not None and time.time() - start_time > deadline:
            break
        if v in input_set:
            continue
        
        # Ensure all parents are in fast memory
        for p in dag.get(v, []):
            if p not in fast_mem:
                # Need to reload parent from slow memory
                # First, free up space if needed
                while len(fast_mem) >= fast_memory_budget:
                    # Find a vertex in fast_mem that is not needed by any unprocessed vertex
                    # and is not an output
                    candidate = None
                    for fm_v in list(fast_mem):
                        if fm_v in output_set:
                            continue
                        # Check if fm_v is a parent of any unprocessed vertex
                        needed = False
                        for c in children.get(fm_v, []):
                            if c not in computed and c not in input_set:
                                # Check if c is yet to be processed
                                if c in set(topo_order[topo_order.index(v):]) if v in topo_order else False:
                                    needed = True
                                    break
                        if not needed:
                            candidate = fm_v
                            break
                    
                    if candidate is None:
                        # Fallback: delete any non-output vertex
                        for fm_v in list(fast_mem):
                            if fm_v not in output_set:
                                candidate = fm_v
                                break
                    
                    if candidate is not None:
                        ops.append(('delete', candidate))
                        fast_mem.discard(candidate)
                    else:
                        break
                
                if p not in fast_mem:
                    ops.append(('load', p))
                    fast_mem.add(p)
                    slow_mem.discard(p)
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            # Need to free space
            # Delete a vertex that is no longer needed
            for fm_v in list(fast_mem):
                if fm_v in output_set:
                    continue
                # Check if fm_v is needed by any future vertex
                needed = False
                for c in children.get(fm_v, []):
                    if c not in computed and c not in input_set:
                        needed = True
                        break
                if not needed:
                    ops.append(('delete', fm_v))
                    fast_mem.discard(fm_v)
                    break
        
        if len(fast_mem) < fast_memory_budget:
            ops.append(('compute', v))
            fast_mem.add(v)
            computed.add(v)
        else:
            # Can't compute, skip (shouldn't happen with proper budget)
            continue
        
        # Update remaining children count for v's parents
        for p in dag.get(v, []):
            if p in remaining_children:
                remaining_children[p] -= 1
        
        # After computing v, try to delete parents that are no longer needed
        for p in dag.get(v, []):
            if p in fast_mem and remaining_children.get(p, 0) == 0 and p not in output_set:
                # p is no longer needed
                # But we might want to keep it if it's an input that could be reloaded
                # For now, delete to free space
                ops.append(('delete', p))
                fast_mem.discard(p)
    
    # Store all outputs
    for v in outputs:
        if deadline is not None and time.time() - start_time > deadline:
            break
        if v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            slow_mem.add(v)
        elif v in slow_mem:
            # Already in slow memory, no-op
            pass
        else:
            # Shouldn't happen
            pass
    
    return ops


def _improved_greedy_pebble(dag, fast_memory_budget, deadline=None):
    """Improved greedy pebbling with better deletion strategy.
    
    Uses a priority-based approach to decide which vertices to delete,
    prioritizing vertices that are least likely to be needed again.
    """
    start_time = time.time()
    topo_order = _topo_sort(dag)
    inputs, outputs = _identify_inputs_outputs(dag, topo_order)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Build reverse adjacency (children)
    children = {v: [] for v in topo_order}
    for v, parents in dag.items():
        for p in parents:
            if p in children:
                children[p].append(v)
    
    # For each vertex, count how many times it's used as a parent
    use_count = {v: len(children.get(v, [])) for v in topo_order}
    
    # State
    fast_mem = set()
    slow_mem = set(inputs)
    computed = set()
    ops = []
    
    # Track which vertices are still needed
    needed = {v: use_count[v] for v in topo_order}
    
    # Load all inputs
    for v in inputs:
        if deadline is not None and time.time() - start_time > deadline:
            break
        ops.append(('load', v))
        fast_mem.add(v)
        slow_mem.discard(v)
    
    # Process vertices in topological order
    for v in topo_order:
        if deadline is not None and time.time() - start_time > deadline:
            break
        if v in input_set:
            continue
        
        # Ensure all parents are in fast memory
        for p in dag.get(v, []):
            if p not in fast_mem:
                # Need to free space and reload
                while len(fast_mem) >= fast_memory_budget:
                    # Find best vertex to delete: lowest needed count, not an output
                    best = None
                    best_score = float('inf')
                    for fm_v in fast_mem:
                        if fm_v in output_set:
                            continue
                        score = needed.get(fm_v, 0)
                        if score < best_score:
                            best_score = score
                            best = fm_v
                    if best is not None:
                        ops.append(('delete', best))
                        fast_mem.discard(best)
                    else:
                        break
                
                if p not in fast_mem:
                    ops.append(('load', p))
                    fast_mem.add(p)
                    slow_mem.discard(p)
        
        # Free space if needed for computation
        while len(fast_mem) >= fast_memory_budget:
            best = None
            best_score = float('inf')
            for fm_v in fast_mem:
                if fm_v in output_set:
                    continue
                score = needed.get(fm_v, 0)
                if score < best_score:
                    best_score = score
                    best = fm_v
            if best is not None:
                ops.append(('delete', best))
                fast_mem.discard(best)
            else:
                break
        
        if len(fast_mem) < fast_memory_budget:
            ops.append(('compute', v))
            fast_mem.add(v)
            computed.add(v)
        
        # Update needed counts for parents of v
        for p in dag.get(v, []):
            if p in needed:
                needed[p] -= 1
        
        # Delete parents that are no longer needed
        for p in dag.get(v, []):
            if p in fast_mem and needed.get(p, 0) == 0 and p not in output_set:
                ops.append(('delete', p))
                fast_mem.discard(p)
    
    # Store all outputs
    for v in outputs:
        if deadline is not None and time.time() - start_time > deadline:
            break
        if v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            slow_mem.add(v)
    
    return ops


def _layered_pebble(dag, fast_memory_budget, deadline=None):
    """Layered pebbling: process DAG layer by layer.
    
    For layered DAGs, this can be more efficient by keeping only the current
    and next layer in fast memory.
    """
    start_time = time.time()
    topo_order = _topo_sort(dag)
    inputs, outputs = _identify_inputs_outputs(dag, topo_order)
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Build children map
    children = {v: [] for v in topo_order}
    for v, parents in dag.items():
        for p in parents:
            if p in children:
                children[p].append(v)
    
    # Determine layers based on longest path from inputs
    layer = {}
    for v in topo_order:
        if v in input_set:
            layer[v] = 0
        else:
            max_parent_layer = -1
            for p in dag.get(v, []):
                if p in layer:
                    max_parent_layer = max(max_parent_layer, layer[p])
            layer[v] = max_parent_layer + 1 if max_parent_layer >= 0 else 0
    
    # Group vertices by layer
    layers = {}
    for v in topo_order:
        l = layer.get(v, 0)
        if l not in layers:
            layers[l] = []
        layers[l].append(v)
    
    # Process layer by layer
    fast_mem = set()
    slow_mem = set(inputs)
    computed = set()
    ops = []
    
    max_layer = max(layers.keys()) if layers else 0
    
    for l in range(max_layer + 1):
        if deadline is not None and time.time() - start_time > deadline:
            break
        
        layer_vertices = layers.get(l, [])
        
        for v in layer_vertices:
            if v in input_set:
                # Load input
                if v not in fast_mem:
                    while len(fast_mem) >= fast_memory_budget:
                        # Delete from previous layers
                        for fm_v in list(fast_mem):
                            if layer.get(fm_v, -1) < l:
                                ops.append(('delete', fm_v))
                                fast_mem.discard(fm_v)
                                break
                    if v not in fast_mem and len(fast_mem) < fast_memory_budget:
                        ops.append(('load', v))
                        fast_mem.add(v)
                        slow_mem.discard(v)
                continue
            
            # Ensure parents are in fast memory
            for p in dag.get(v, []):
                if p not in fast_mem:
                    while len(fast_mem) >= fast_memory_budget:
                        for fm_v in list(fast_mem):
                            if layer.get(fm_v, -1) < l:
                                ops.append(('delete', fm_v))
                                fast_mem.discard(fm_v)
                                break
                    if p not in fast_mem and len(fast_mem) < fast_memory_budget:
                        ops.append(('load', p))
                        fast_mem.add(p)
                        slow_mem.discard(p)
            
            # Compute
            while len(fast_mem) >= fast_memory_budget:
                for fm_v in list(fast_mem):
                    if layer.get(fm_v, -1) < l:
                        ops.append(('delete', fm_v))
                        fast_mem.discard(fm_v)
                        break
            
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                computed.add(v)
        
        # After processing a layer, delete vertices from this layer that are no longer needed
        # (i.e., all their children are in later layers that have been processed or will be)
        # For now, keep them until the next layer starts
    
    # Store outputs
    for v in outputs:
        if deadline is not None and time.time() - start_time > deadline:
            break
        if v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            slow_mem.add(v)
    
    return ops


def pebble(dag, fast_memory_budget):
    """Main entry point: pebble the DAG with the given fast memory budget.
    
    Returns a list of [operation, vertex] pairs.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Try multiple strategies and pick the best valid one
    candidates = []
    
    # Strategy 1: Improved greedy
    if time.time() - start_time < deadline - 5:
        try:
            ops1 = _improved_greedy_pebble(dag, fast_memory_budget, deadline=deadline - 2)
            if _validate_ops(ops1, dag, fast_memory_budget):
                candidates.append(ops1)
        except Exception:
            pass
    
    # Strategy 2: Layered pebbling
    if time.time() - start_time < deadline - 5:
        try:
            ops2 = _layered_pebble(dag, fast_memory_budget, deadline=deadline - 2)
            if _validate_ops(ops2, dag, fast_memory_budget):
                candidates.append(ops2)
        except Exception:
            pass
    
    # Strategy 3: Basic greedy
    if time.time() - start_time < deadline - 5:
        try:
            ops3 = _greedy_pebble(dag, fast_memory_budget, deadline=deadline - 2)
            if _validate_ops(ops3, dag, fast_memory_budget):
                candidates.append(ops3)
        except Exception:
            pass
    
    # Fallback: Naive schedule
    if not candidates:
        try:
            ops_naive, _ = _naive_schedule(dag, fast_memory_budget)
            if _validate_ops(ops_naive, dag, fast_memory_budget):
                candidates.append(ops_naive)
        except Exception:
            pass
    
    if not candidates:
        # Last resort: return empty or minimal valid sequence
        # This should not happen
        return []
    
    # Pick the candidate with fewest transfers
    best_ops = min(candidates, key=lambda ops: _count_transfers(ops))
    
    # Ensure we're within operation limit
    if len(best_ops) > 20000:
        # Truncate if necessary (may invalidate, but better than nothing)
        best_ops = best_ops[:20000]
    
    return best_ops

# EVOLVE-BLOCK-END
