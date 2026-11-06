# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Uses layer-by-layer greedy scheduling with memory-aware eviction.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG - try common structures
    vertices = []
    parents = {}
    outputs = []
    inputs = []
    
    if isinstance(dag, dict):
        # Try various key names
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif 'parents' in dag:
            vertices = list(dag['parents'].keys())
        
        if 'parents' in dag:
            parents = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in dag['parents'].items()}
        elif 'edges' in dag:
            # edges might be (child, parent) pairs
            for edge in dag['edges']:
                if len(edge) == 2:
                    child, parent = edge
                    if child not in parents:
                        parents[child] = []
                    parents[child].append(parent)
        
        if 'outputs' in dag:
            outputs = list(dag['outputs'])
        elif 'sinks' in dag:
            outputs = list(dag['sinks'])
        
        if 'inputs' in dag:
            inputs = list(dag['inputs'])
        elif 'sources' in dag:
            inputs = list(dag['sources'])
    
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                p = item[1]
                vertices.append(v)
                parents[v] = list(p) if isinstance(p, (list, tuple)) else [p]
            elif isinstance(item, dict):
                v = item.get('vertex', item.get('id'))
                p = item.get('parents', item.get('predecessors', []))
                if v is not None:
                    vertices.append(v)
                    parents[v] = list(p) if isinstance(p, (list, tuple)) else ([p] if p else [])
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify inputs and outputs
    if not inputs:
        inputs = [v for v in vertices if not parents.get(v, [])]
    
    if not outputs:
        outputs = [v for v in vertices if not children.get(v, [])]
    
    # Compute usage count (how many times a value is needed downstream)
    usage = {}
    def compute_usage(v):
        if v in usage:
            return usage[v]
        usage[v] = 1
        for c in children.get(v, []):
            usage[v] += compute_usage(c)
        return usage[v]
    
    for v in vertices:
        compute_usage(v)
    
    # Compute layer (longest path from any input)
    layer = {}
    for v in vertices:
        if not parents.get(v, []):
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in parents[v]) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Group by layer
    layers = defaultdict(list)
    for v in vertices:
        layers[layer[v]].append(v)
    
    # Generate operations
    ops = []
    fast_mem = set()
    computed = set()
    op_count = 0
    MAX_OPS = 20000
    
    for li in range(max_layer + 1):
        if time.time() - start_time > DEADLINE or op_count >= MAX_OPS:
            break
        
        layer_verts = sorted(layers[li], key=lambda v: (-usage[v], v))
        
        for v in layer_verts:
            if op_count >= MAX_OPS:
                break
            
            pvs = parents.get(v, [])
            
            # Load any parents not in fast memory
            for p in pvs:
                if p not in fast_mem:
                    ops.append(['load', p])
                    op_count += 1
                    fast_mem.add(p)
            
            # Compute v
            ops.append(['compute', v])
            op_count += 1
            computed.add(v)
            
            # Add v to fast memory if possible
            if len(fast_mem) < fast_memory_budget:
                fast_mem.add(v)
            else:
                # Evict least useful non-output vertex
                candidates = [x for x in fast_mem if x not in outputs and x != v]
                if candidates:
                    evict = min(candidates, key=lambda x: usage[x])
                    ops.append(['delete', evict])
                    op_count += 1
                    fast_mem.discard(evict)
                    fast_mem.add(v)
        
        # Evict vertices whose children are all computed
        for v in list(fast_mem):
            if v in outputs:
                continue
            ch = children.get(v, [])
            if all(c in computed for c in ch):
                ops.append(['delete', v])
                op_count += 1
                fast_mem.discard(v)
    
    # Ensure outputs are in fast memory
    for o in outputs:
        if o not in fast_mem:
            ops.append(['load', o])
            op_count += 1
            fast_mem.add(o)
    
    if len(ops) > MAX_OPS:
        ops = ops[:MAX_OPS]
    
    return ops
# EVOLVE-BLOCK-END
