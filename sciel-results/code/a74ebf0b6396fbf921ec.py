"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with smart eviction.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        raise ValueError(f"Unexpected dag type: {type(dag)}")
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # ref_count[v] = number of children of v that still need v
    ref_count = {}
    for v in vertices:
        ref_count[v] = len(children.get(v, []))
    
    # For eviction priority: prefer evicting vertices with ref_count == 0
    # Among those, prefer evicting vertices that are NOT outputs (outputs must be stored anyway)
    # Use a min-heap: (priority, vertex)
    # Lower priority = evict first
    
    # Fast memory state
    fast_mem = set()
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    # Process in topological order
    for v in vertices:
        if time.time() - start_time > DEADLINE:
            break
        if op_count >= MAX_OPS:
            break
        
        if v in inputs:
            # Load input into fast memory
            if v not in fast_mem:
                operations.append(['load', v])
                op_count += 1
                fast_mem.add(v)
        else:
            # Compute v: ensure all parents are in fast memory
            for p in parents.get(v, []):
                if p not in fast_mem:
                    # Need to reload p
                    operations.append(['load', p])
                    op_count += 1
                    fast_mem.add(p)
            
            operations.append(['compute', v])
            op_count += 1
            fast_mem.add(v)
        
        # Decrement ref_count of parents (they've been consumed by v)
        for p in parents.get(v, []):
            ref_count[p] -= 1
        
        # Evict vertices with ref_count == 0 to stay within budget
        # Also evict to stay within fast_memory_budget
        while len(fast_mem) > fast_memory_budget:
            # Find a vertex to evict: prefer ref_count == 0, non-output
            best = None
            best_priority = None
            for candidate in fast_mem:
                rc = ref_count[candidate]
                is_output = candidate in outputs
                # Priority: lower is better to evict
                # ref_count == 0 is best, then higher ref_count
                # Non-outputs preferred over outputs
                priority = (rc, is_output)
                if best_priority is None or priority < best_priority:
                    best_priority = priority
                    best = candidate
            
            if best is None:
                break
            
            # Evict best
            if ref_count[best] == 0:
                # No longer needed, store it
                operations.append(['store', best])
                op_count += 1
            else:
                # Still needed but over budget - must store (will reload later)
                operations.append(['store', best])
                op_count += 1
            
            fast_mem.remove(best)
        
        # Additionally, proactively evict vertices with ref_count == 0 
        # to keep memory usage low (even if within budget)
        # This can help if later layers need more space
        if len(fast_mem) > fast_memory_budget * 0.5:
            for candidate in list(fast_mem):
                if ref_count[candidate] == 0 and candidate not in outputs:
                    operations.append(['store', candidate])
                    op_count += 1
                    fast_mem.remove(candidate)
                    if len(fast_mem) <= fast_memory_budget * 0.5:
                        break
    
    # Ensure all outputs are stored at the end
    for v in outputs:
        if v in fast_mem and op_count < MAX_OPS:
            operations.append(['store', v])
            op_count += 1
            fast_mem.remove(v)
    
    return operations
# EVOLVE-BLOCK-END

