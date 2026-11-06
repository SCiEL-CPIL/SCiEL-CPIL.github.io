# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0
    MAX_OPS = 20000
    
    try:
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        topo_order = dag.get('topo_order', [])
        
        if not topo_order and vertices:
            adj_temp = {v: [] for v in vertices}
            in_deg = {v: 0 for v in vertices}
            for parent, child in edges:
                adj_temp[parent].append(child)
                in_deg[child] = in_deg.get(child, 0) + 1
            queue = [v for v in vertices if in_deg[v] == 0]
            topo_order = []
            while queue:
                v = queue.pop(0)
                topo_order.append(v)
                for child in adj_temp[v]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        
        if not topo_order:
            return []
        
        adj = {v: [] for v in vertices}
        parents = {v: [] for v in vertices}
        for parent, child in edges:
            adj[parent].append(child)
            parents[child].append(parent)
        
        remaining_children = {v: len(adj[v]) for v in vertices}
        
        fast_mem = set(inputs)
        stored = set()
        computed = set()
        operations = []
        
        def priority(v):
            if v in outputs:
                return 10**9
            return remaining_children[v] * 10 + (1 if v in inputs else 0)
        
        def evict_one():
            if not fast_mem:
                return False
            candidates = [v for v in fast_mem if v not in outputs]
            if not candidates:
                return False
            min_v = min(candidates, key=priority)
            operations.append(['delete', min_v])
            fast_mem.discard(min_v)
            return True
        
        for v in topo_order:
            if time.time() - start_time > DEADLINE or len(operations) >= MAX_OPS:
                break
            
            for p in parents[v]:
                if p not in fast_mem:
                    if p in stored:
                        if len(fast_mem) >= fast_memory_budget:
                            evict_one()
                        operations.append(['load', p])
                        fast_mem.add(p)
            
            if v not in computed:
                if v not in fast_mem:
                    if len(fast_mem) >= fast_memory_budget:
                        evict_one()
                    operations.append(['compute', v])
                    fast_mem.add(v)
                else:
                    operations.append(['compute', v])
                computed.add(v)
            
            for p in parents[v]:
                remaining_children[p] = max(0, remaining_children[p] - 1)
            
            if v in outputs or remaining_children[v] > 0:
                operations.append(['store', v])
                stored.add(v)
            
            if v not in outputs and remaining_children[v] == 0:
                if v in fast_mem:
                    operations.append(['delete', v])
                    fast_mem.discard(v)
        
        for v in outputs:
            if v not in fast_mem:
                if v in stored:
                    if len(fast_mem) >= fast_memory_budget:
                        evict_one()
                    operations.append(['load', v])
                    fast_mem.add(v)
        
        return operations
    except Exception:
        return []
# EVOLVE-BLOCK-END
