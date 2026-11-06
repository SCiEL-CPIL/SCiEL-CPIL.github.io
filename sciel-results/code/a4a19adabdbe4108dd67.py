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

def _compute_recomputation_cost(vertices, parents, outputs):
    """
    Compute recomputation cost for each vertex based on logarithmic depth.
    Vertices with higher cost are more expensive to recompute, so we prioritize keeping them in memory.
    """
    # Compute depth (longest path from any input)
    depth = {}
    for v in vertices:
        if not parents.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents[v]) + 1
    
    # Recomputation cost: log of (1 + number of descendants) + depth
    # This approximates the cost of recomputing a vertex if it's evicted
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Count descendants
    def count_descendants(v, memo={}):
        if v in memo:
            return memo[v]
        count = 0
        for c in children[v]:
            count += 1 + count_descendants(c)
        memo[v] = count
        return count
    
    import math
    cost = {}
    for v in vertices:
        desc = count_descendants(v)
        # Higher depth and more descendants = higher recomputation cost
        cost[v] = math.log(1 + desc) + depth[v]
    
    return cost

def _list_schedule(dag, fast_memory_budget):
    """
    List-scheduling heuristic that prioritizes vertices with highest recomputation cost.
    Keeps critical path data in fast memory to minimize transfers.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    # Precompute children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute recomputation costs
    cost = _compute_recomputation_cost(vertices, parents, outputs)
    
    ops = []
    in_memory = set()
    computed = set()
    
    # Track remaining children for each vertex
    remaining_children = {v: len(children[v]) for v in vertices}
    
    # Inputs: vertices with no parents
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Load inputs in order of highest cost first
    inputs_sorted = sorted(inputs, key=lambda v: -cost[v])
    for v in inputs_sorted:
        if len(in_memory) >= fast_memory_budget:
            return None  # Budget too small
        ops.append(['load', v])
        in_memory.add(v)
        computed.add(v)
    
    # Ready queue: vertices whose parents are all in memory
    import heapq
    ready = []
    
    def is_ready(v):
        return all(p in in_memory for p in parents.get(v, []))
    
    # Initialize ready queue with non-input vertices that are ready
    for v in vertices:
        if v not in inputs and is_ready(v):
            heapq.heappush(ready, (-cost[v], v))
    
    while ready:
        # Get highest cost ready vertex
        _, v = heapq.heappop(ready)
        
        # Skip if already computed
        if v in computed:
            continue
        
        # Ensure all parents are in memory (they should be, but double-check)
        for p in parents.get(v, []):
            if p not in in_memory:
                # Need to recompute parent - this shouldn't happen with our logic
                # but if it does, we need to handle it
                return None
        
        # Free space if needed
        if len(in_memory) >= fast_memory_budget:
            # Find a vertex to evict: prefer lower cost, no remaining children
            candidates = []
            for u in in_memory:
                if u == v:
                    continue
                if remaining_children[u] == 0:
                    # Can safely evict
                    priority = cost[u]  # Lower cost = evict first
                    candidates.append((priority, u))
            
            if not candidates:
                # Cannot free space
                return None
            
            # Evict lowest cost candidate
            candidates.sort()
            _, u_to_evict = candidates[0]
            if u_to_evict in outputs:
                ops.append(['store', u_to_evict])
            else:
                ops.append(['delete', u_to_evict])
            in_memory.remove(u_to_evict)
        
        # Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        computed.add(v)
        
        # Update remaining children for parents
        for p in parents[v]:
            remaining_children[p] -= 1
        
        # Add newly ready children to ready queue
        for c in children[v]:
            if c not in computed and is_ready(c):
                heapq.heappush(ready, (-cost[c], c))
    
    # Check if all vertices are computed
    if len(computed) != len(vertices):
        return None
    
    # Final cleanup: store outputs, delete others
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
        # Try the list-scheduling heuristic first (better prioritization)
        schedule = _list_schedule(dag, fast_memory_budget)
        
        if schedule is not None:
            # Validate before returning
            if _validate_schedule(dag, schedule, fast_memory_budget):
                return schedule
        
        # Fallback: try the simple schedule
        schedule = _simple_schedule(dag, fast_memory_budget)
        
        if schedule is not None:
            if _validate_schedule(dag, schedule, fast_memory_budget):
                return schedule
        
        # If both fail, return empty
        return []
        
    except Exception as e:
        # Fallback to baseline if my logic fails
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

# EVOLVE-BLOCK-END
