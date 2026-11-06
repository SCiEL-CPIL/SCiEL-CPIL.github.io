import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    # Parse DAG
    # dag structure: likely a dict or list of dicts.
    # Based on "topologically ordered parents and output vertices", 
    # let's assume dag is a list of nodes or a dict mapping node_id to info.
    # Common format: list of (node_id, parents, is_output) or similar.
    # Let's inspect typical inputs for such tasks.
    # If dag is a dict: {node_id: {'parents': [...], 'output': bool}}
    # If dag is a list: [(node_id, parents_list, is_output)]
    
    # We need to be robust to input format.
    # Let's try to infer.
    nodes = []
    parents_map = {}
    is_output_map = {}
    node_ids = set()
    
    if isinstance(dag, dict):
        for k, v in dag.items():
            node_ids.add(k)
            if isinstance(v, dict):
                parents_map[k] = v.get('parents', [])
                is_output_map[k] = v.get('output', False)
            elif isinstance(v, list):
                # Maybe [parents, is_output]
                parents_map[k] = v[0] if len(v) > 0 else []
                is_output_map[k] = v[1] if len(v) > 1 else False
            else:
                parents_map[k] = []
                is_output_map[k] = False
    elif isinstance(dag, list):
        for item in dag:
            if isinstance(item, dict):
                nid = item.get('id') or item.get('node') or item.get('name')
                if nid is None:
                    continue
                node_ids.add(nid)
                parents_map[nid] = item.get('parents', [])
                is_output_map[nid] = item.get('output', False)
            elif isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    nid = item[0]
                    node_ids.add(nid)
                    parents_map[nid] = item[1] if len(item) > 1 and isinstance(item[1], list) else []
                    is_output_map[nid] = item[2] if len(item) > 2 else False
    else:
        # Fallback: assume empty
        pass

    if not node_ids:
        return []

    # Topological Sort
    # Kahn's algorithm
    in_degree = {n: 0 for n in node_ids}
    children_map = defaultdict(list)
    
    for n in node_ids:
        for p in parents_map[n]:
            if p in node_ids:
                in_degree[n] += 1
                children_map[p].append(n)
            else:
                # Parent not in node_ids? Assume it's an external input, so no edge in graph structure for topo sort of internal nodes?
                # But we need to load it.
                pass

    queue = deque([n for n in node_ids if in_degree[n] == 0])
    topo_order = []
    
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children_map[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If topo_order is empty or shorter than node_ids, there's a cycle or issue.
    # Fallback: just use list(node_ids)
    if len(topo_order) != len(node_ids):
        topo_order = list(node_ids)

    # Compute Next Use Index
    # For each node, find the earliest index in topo_order where it is needed as a parent.
    next_use = {n: float('inf') for n in node_ids}
    
    # Iterate through topo_order, and for each node, update next_use of its parents
    # Actually, next_use[u] is the min index j such that u is a parent of topo_order[j].
    # So we can iterate j from 0 to N-1:
    #   for each parent p of topo_order[j]:
    #       if next_use[p] == inf: next_use[p] = j
    
    for j, node in enumerate(topo_order):
        for p in parents_map[node]:
            if p in node_ids and next_use[p] == float('inf'):
                next_use[p] = j

    # Outputs must be stored at the end. Their next_use is inf, which is correct for eviction priority (evict last).
    # But we must ensure they are stored.

    fast_mem = set()
    stored = set()
    ops = []
    
    def evict():
        if not fast_mem:
            return
        # Find node in fast_mem with max next_use
        # If tie, pick any.
        victim = max(fast_mem, key=lambda x: next_use.get(x, float('inf')))
        fast_mem.remove(victim)
        if is_output_map.get(victim, False):
            ops.append(['store', victim])
            stored.add(victim)
        else:
            ops.append(['delete', victim])
            
    def ensure_in_fast_mem(node):
        if node in fast_mem:
            return
        if node not in node_ids:
            # External input?
            # Treat as input
            if len(fast_mem) >= fast_memory_budget:
                evict()
            ops.append(['load', node])
            fast_mem.add(node)
            return

        if not parents_map.get(node, []):
            # Input node
            if len(fast_mem) >= fast_memory_budget:
                evict()
            ops.append(['load', node])
            fast_mem.add(node)
        else:
            # Internal node: recompute
            for p in parents_map[node]:
                ensure_in_fast_mem(p)
            
            if len(fast_mem) >= fast_memory_budget:
                evict()
            
            ops.append(['compute', node])
            fast_mem.add(node)

    for node in topo_order:
        # Ensure parents are in fast mem
        for p in parents_map[node]:
            ensure_in_fast_mem(p)
        
        # Compute node
        if len(fast_mem) >= fast_memory_budget:
            evict()
        
        if not parents_map.get(node, []):
            # It's an input, already loaded in ensure_in_fast_mem?
            # No, ensure_in_fast_mem was called for parents.
            # If node is input, it has no parents, so ensure_in_fast_mem wasn't called for it in the loop above.
            # So we must load it now.
            if node not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evict()
                ops.append(['load', node])
                fast_mem.add(node)
        else:
            # It's computed
            if node not in fast_mem:
                ops.append(['compute', node])
                fast_mem.add(node)
            # If it was already in fast_mem (e.g. from a previous recomputation?), we don't need to compute again?
            # But in topo order, we visit each node once.
            # If we evicted it and then needed it again, ensure_in_fast_mem would have recomputed it.
            # So here, if it's not in fast_mem, it means it wasn't kept, so we compute it.
            
        # Check time
        if time.time() > deadline:
            break

    # Final cleanup: Store all outputs that are in fast_mem
    for node in list(fast_mem):
        if is_output_map.get(node, False):
            ops.append(['store', node])
            stored.add(node)
        else:
            ops.append(['delete', node])

    return ops

