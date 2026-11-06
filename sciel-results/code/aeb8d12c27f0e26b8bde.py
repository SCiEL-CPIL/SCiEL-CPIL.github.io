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

def _validate_schedule(dag, ops, fast_memory_budget):
    """
    Validates a schedule by simulating the state machine.
    Returns True if valid, False otherwise.
    """
    vertices = set(dag['vertices'])
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # State: 
    # - blue: set of vertices in slow memory (inputs start blue, outputs must end blue)
    # - red: set of vertices in fast memory
    # - computed: set of vertices that have been computed at least once
    
    # Initially, all inputs are blue. What are inputs? Vertices with no parents.
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    blue = set(inputs)
    red = set()
    computed = set()
    
    for op, v in ops:
        if v not in vertices:
            return False
        
        if op == 'load':
            # Must be in blue, not in red
            if v not in blue or v in red:
                return False
            if len(red) >= fast_memory_budget:
                return False
            red.add(v)
            blue.remove(v)
            
        elif op == 'store':
            # Must be in red, not in blue (or can it be in both? No, usually disjoint)
            # Actually, if it's in red, it's not in blue.
            if v not in red or v in blue:
                return False
            red.remove(v)
            blue.add(v)
            
        elif op == 'compute':
            # Must have all parents in red
            if v in red:
                # Already computed and in memory? Recompute is allowed, but does it overwrite?
                # "Recomputations allowed". If v is already in red, computing it again is a no-op?
                # Or does it free a slot? No, it stays in red.
                # But usually you compute to get it into red. If it's already there, it's redundant but valid?
                # Let's assume it's valid if parents are in red.
                pass
            
            for p in parents.get(v, []):
                if p not in red:
                    return False
            
            if v not in red:
                if len(red) >= fast_memory_budget:
                    return False
                red.add(v)
            computed.add(v)
            
        elif op == 'delete':
            # Must be in red
            if v not in red:
                return False
            red.remove(v)
            
        else:
            return False
            
    # Check final state
    # All outputs must be blue
    for v in outputs:
        if v not in blue:
            return False
            
    return True

def _simple_schedule(dag, fast_memory_budget):
    """
    A robust simple schedule:
    1. Load all inputs.
    2. Compute vertices in topological order.
    3. When memory is full, delete vertices that are no longer needed.
    4. Store outputs when they are no longer needed as parents, or at the end.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    
    # Precompute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    ops = []
    in_memory = set()
    
    # Count remaining children for each vertex
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # 1. Load all inputs
    for v in order:
        if not parents.get(v, []):
            if len(in_memory) >= fast_memory_budget:
                # Cannot load all inputs if budget is too small.
                # This simple schedule fails.
                return None
            ops.append(['load', v])
            in_memory.add(v)
            
    # 2. Compute in topological order
    for v in order:
        if not parents.get(v, []):
            # Input, already loaded
            continue
            
        # Ensure parents are in memory.
        # With the simple "load all inputs first" strategy, parents might have been deleted.
        # This simple strategy is only valid if we don't delete parents before all children are computed.
        # But we delete when remaining_children == 0.
        # So if a parent has multiple children, it stays until the last child is computed.
        # Since we compute in topological order, when we compute v, all its parents' other children 
        # (that come before v in topo order) are already computed.
        # But what about children that come AFTER v?
        # If p is a parent of v, and p has another child u that comes after v, 
        # then remaining_children[p] > 0 when we compute v.
        # So p is NOT deleted.
        # So p is still in memory.
        # Thus, parents are always in memory when we compute their children.
        
        # Check if v fits in memory
        if len(in_memory) >= fast_memory_budget:
            # Need to free space.
            # Find a vertex in in_memory that is not needed (remaining_children == 0) and not an output (or output can be stored).
            # Actually, if it's an output, we can store it to free space.
            # If it's not an output, we can delete it.
            
            # But wait, if we are about to compute v, and memory is full, we need to free a slot.
            # We can only free a slot by deleting/storing a vertex that is NOT a parent of any uncomputed vertex.
            # `remaining_children[u] == 0` means u is not a parent of any uncomputed vertex.
            
            freed = False
            for u in list(in_memory):
                if u == v:
                    continue
                if remaining_children[u] == 0:
                    if u in outputs:
                        ops.append(['store', u])
                    else:
                        ops.append(['delete', u])
                    in_memory.remove(u)
                    freed = True
                    break
                    
            if not freed:
                # Cannot free space. Schedule invalid for this budget.
                return None
                
        ops.append(['compute', v])
        in_memory.add(v)
        
        # Update remaining_children for parents of v
        for p in parents[v]:
            remaining_children[p] -= 1
            
    # 3. Final cleanup
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
        # Try the simple schedule
        schedule = _simple_schedule(dag, fast_memory_budget)
        
        if schedule is not None:
            # Validate before returning
            if _validate_schedule(dag, schedule, fast_memory_budget):
                return schedule
            else:
                # Should not happen if logic is correct, but fallback
                pass
        
        # Fallback: if simple schedule failed (e.g., budget too small for all inputs),
        # we might need a more complex strategy.
        # For now, return an empty list which will score 0, or try a different approach.
        # Given the "layered DAGs" and typical budgets, the simple schedule should work.
        
        # If we get here, the simple schedule is invalid or failed.
        # Let's try a very conservative schedule:
        # Load one input, compute what we can, store/delete, repeat.
        # This is complex to implement correctly under budget.
        
        # For now, return empty.
        return []
        
    except Exception as e:
        # Fallback to baseline if my logic fails
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
