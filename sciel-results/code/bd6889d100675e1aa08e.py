"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """
    Returns a list of vertices in topological order.
    vertices: list of vertex IDs
    parents: dict mapping vertex -> list of parent IDs
    """
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
        return vertices
    return order

def _verify_schedule(ops, dag, fast_memory_budget):
    """
    Verifies if the schedule is valid.
    Returns True if valid, False otherwise.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # State: 'blue' (slow), 'red' (fast), 'void'
    state = {v: 'blue' if not parents.get(v, []) else 'void' for v in vertices}
    # Inputs start blue. Others start void.
    # Wait, "Inputs start blue".
    # What about non-inputs? They are not in memory.
    
    fast_count = 0
    max_fast = 0
    
    for op, v in ops:
        if op == 'load':
            if state[v] != 'blue':
                return False
            state[v] = 'red'
            fast_count += 1
        elif op == 'store':
            if state[v] != 'red':
                return False
            state[v] = 'blue'
            fast_count -= 1
        elif op == 'delete':
            if state[v] != 'red':
                return False
            state[v] = 'void'
            fast_count -= 1
        elif op == 'compute':
            # All parents must be red
            for p in parents.get(v, []):
                if state[p] != 'red':
                    return False
            if state[v] != 'void':
                # Recomputation allowed? "Recomputations allowed"
                # If it's already red, we can't compute it again without deleting?
                # Usually compute takes void -> red.
                # If it's red, it's already there.
                # If it's blue, we need to load? No, compute creates it.
                # Let's assume compute requires v to be void.
                return False
            state[v] = 'red'
            fast_count += 1
        
        max_fast = max(max_fast, fast_count)
        if fast_count > fast_memory_budget:
            return False
        
    # All outputs must be blue
    for v in outputs:
        if state[v] != 'blue':
            return False
            
    return True

def _naive_schedule(dag, fast_memory_budget):
    """
    Generates a naive schedule:
    1. Load all inputs
    2. Compute vertices in topological order
    3. Store outputs
    4. Delete all intermediate results
    
    Assumes dag is a dict with keys 'vertices', 'parents', 'outputs'.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    ops = []
    in_memory = set()
    
    # 1. Load all inputs
    for v in order:
        if not parents.get(v, []):
            ops.append(['load', v])
            in_memory.add(v)
            
    # 2. Compute
    # We need to know which vertices are "needed" in the future.
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children_map[p].append(v)
            
    out_degree_uncomputed = {v: len(children_map[v]) for v in vertices}
    computed = set()
    
    for v in order:
        if not parents.get(v, []):
            # Input
            computed.add(v)
            continue
            
        # Non-input
        # Ensure parents are in memory
        # In naive, we keep everything.
        # If memory is tight, this fails. But naive is baseline.
        
        ops.append(['compute', v])
        in_memory.add(v)
        computed.add(v)
        
        # Update parents' uncomputed children count
        for p in parents[v]:
            out_degree_uncomputed[p] -= 1
            
        # If v is output, store it if no children left
        if v in outputs and out_degree_uncomputed[v] == 0:
            ops.append(['store', v])
            in_memory.remove(v)
            
    # 3. Cleanup
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

def _recursive_pebble(dag, fast_memory_budget):
    """
    Recursive pebbling strategy for layered DAGs.
    Partitions the DAG into blocks (layers) and processes them recursively.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    if not vertices:
        return []
    
    # Identify layers based on depth from inputs
    # Depth of input is 0. Depth of v is 1 + max(depth of parents).
    order = _topological_sort(vertices, parents)
    depth = {}
    for v in order:
        if not parents.get(v, []):
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth[p] for p in parents[v])
            
    max_depth = max(depth.values()) if depth else 0
    
    # If max depth is small, we can just do a simple sweep
    # If max depth is large, we split it.
    
    # Let's try a simple block strategy:
    # Split vertices into two sets: A (lower depth) and B (higher depth).
    # Boundary: vertices in A that are parents of vertices in B.
    
    # For now, let's try a "sweep" strategy which is better than naive for wide DAGs
    # by deleting vertices as soon as they are no longer needed.
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children_map[p].append(v)
            
    ops = []
    in_memory = set()
    
    # Pre-load inputs
    for v in order:
        if not parents.get(v, []):
            ops.append(['load', v])
            in_memory.add(v)
            
    # Track how many children each vertex has that are NOT yet processed
    # Actually, we process in topological order.
    # A vertex p can be deleted if all its children are processed.
    # Wait, if we process in topo order, when we process child c, we know p is a parent.
    # We can decrement a counter for p.
    
    remaining_children = {v: len(children_map[v]) for v in vertices}
    
    for v in order:
        if not parents.get(v, []):
            # Input, already loaded
            continue
            
        # Check if parents are in memory
        # If not, we have a problem.
        # In this simple sweep, we assume we keep parents until all children are done.
        # So parents should be in memory.
        
        ops.append(['compute', v])
        in_memory.add(v)
        
        # Update remaining children count for parents
        for p in parents[v]:
            remaining_children[p] -= 1
            # If p is no longer needed (all children processed) and p is not an output,
            # we can delete it to free space.
            # But we must be careful: if p is an output, we must store it.
            # If p is an intermediate, we delete it.
            
            # Can we delete p now?
            # Yes, if remaining_children[p] == 0.
            # But wait, if p is an output, we store it.
            # If p is not an output, we delete it.
            
            # Let's check memory pressure.
            # If we delete p now, we free space.
            # But what if p is needed for something else?
            # remaining_children[p] == 0 means all its children are processed.
            # So p is no longer needed for computation.
            
            # So we can immediately clean up p.
            if remaining_children[p] == 0:
                if p in outputs:
                    # Store it
                    if p in in_memory:
                        ops.append(['store', p])
                        in_memory.remove(p)
                else:
                    # Delete it
                    if p in in_memory:
                        ops.append(['delete', p])
                        in_memory.remove(p)
                        
    # Cleanup any remaining in_memory
    for v in list(in_memory):
        if v in outputs:
            ops.append(['store', v])
        else:
            ops.append(['delete', v])
            
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    try:
        # Try the recursive/sweep pebbling strategy
        schedule = _recursive_pebble(dag, fast_memory_budget)
        
        # Verify
        if len(schedule) > 20000:
            raise Exception("Too many operations")
            
        if not _verify_schedule(schedule, dag, fast_memory_budget):
            raise Exception("Invalid schedule from recursive pebble")
            
        # Check time
        if time.time() > deadline:
            raise Exception("Timeout")
            
        return schedule
        
    except Exception as e:
        # Fallback to naive
        try:
            schedule = _naive_schedule(dag, fast_memory_budget)
            if len(schedule) <= 20000 and _verify_schedule(schedule, dag, fast_memory_budget):
                return schedule
        except Exception:
            pass
            
        # Last resort: empty list (invalid, but better than crash? No, must return list)
        # If both fail, return naive even if invalid?
        # The task says "invalid result... scores 0".
        # So we should return the best valid schedule we can find.
        # If naive is invalid, we have a problem.
        # But naive is usually valid if budget is sufficient.
        
        return []

# EVOLVE-BLOCK-END
