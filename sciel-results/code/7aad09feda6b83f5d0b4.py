"""DAG pebbling with communication minimization."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import heapq

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load+store operations) in DAG pebbling.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG structure - handle various formats
    if isinstance(dag, dict):
        parents_map = dag.get('parents', {})
        vertices = dag.get('vertices', None)
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        topo_order = dag.get('topo_order', None)
    else:
        # Try other formats
        parents_map = {}
        vertices = []
        inputs = set()
        outputs = set()
        topo_order = None
    
    # If vertices not specified, infer from parents_map
    if vertices is None:
        all_v = set(parents_map.keys())
        for v, ps in parents_map.items():
            for p in ps:
                all_v.add(p)
        vertices = list(all_v)
    
    # Build children map
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    # Topological sort if needed
    if topo_order is None or set(topo_order) != set(vertices):
        in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
        heap = [v for v in vertices if in_degree[v] == 0]
        heapq.heapify(heap)
        topo_order = []
        while heap:
            v = heapq.heappop(heap)
            topo_order.append(v)
            for c in children_map[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    heapq.heappush(heap, c)
    
    if not vertices:
        return []
    
    fast_memory = set()
    operations = []
    processed = set()
    
    # Initialize: inputs start in fast memory
    for inp in sorted(inputs):
        fast_memory.add(inp)
    
    # Process in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            processed.add(v)
            continue
        
        # v needs to be computed
        parents = parents_map.get(v, [])
        
        # Ensure all parents are in fast memory
        for p in parents:
            if p not in fast_memory:
                if len(fast_memory) >= fast_memory_budget:
                    # Find best eviction candidate (lowest unprocessed children)
                    best_evict = None
                    best_score = float('inf')
                    for fm_v in fast_memory:
                        if fm_v in outputs:
                            continue
                        unproc = sum(1 for c in children_map[fm_v] if c not in processed)
                        if unproc < best_score:
                            best_score = unproc
                            best_evict = fm_v
                    if best_evict is not None:
                        if best_score == 0:
                            operations.append(['delete', best_evict])
                        else:
                            operations.append(['store', best_evict])
                        fast_memory.discard(best_evict)
                
                operations.append(['load', p])
                fast_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_memory.add(v)
        processed.add(v)
        
        # Evict if over budget
        while len(fast_memory) > fast_memory_budget:
            best_evict = None
            best_score = float('inf')
            for fm_v in fast_memory:
                if fm_v in outputs:
                    continue
                unproc = sum(1 for c in children_map[fm_v] if c not in processed)
                if unproc < best_score:
                    best_score = unproc
                    best_evict = fm_v
            if best_evict is None:
                break
            if best_score == 0:
                operations.append(['delete', best_evict])
            else:
                operations.append(['store', best_evict])
            fast_memory.discard(best_evict)
    
    # Ensure all outputs are in fast memory
    for out in sorted(outputs):
        if out not in fast_memory:
            if len(fast_memory) >= fast_memory_budget:
                best_evict = None
                best_score = float('inf')
                for fm_v in fast_memory:
                    if fm_v in outputs:
                        continue
                    unproc = sum(1 for c in children_map[fm_v] if c not in processed)
                    if unproc < best_score:
                        best_score = unproc
                        best_evict = fm_v
                if best_evict is not None:
                    if best_score == 0:
                        operations.append(['delete', best_evict])
                    else:
                        operations.append(['store', best_evict])
                    fast_memory.discard(best_evict)
            operations.append(['load', out])
            fast_memory.add(out)
    
    return operations

# EVOLVE-BLOCK-END
