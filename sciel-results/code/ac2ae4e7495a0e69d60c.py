# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _parse_dag(dag):
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        parents_raw = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        if vertices is None:
            v_set = set(parents_raw.keys())
            for ps in parents_raw.values():
                v_set.update(ps)
            vertices = sorted(list(v_set))
            
        parents_map = {v: list(parents_raw.get(v, [])) for v in vertices}
        
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
                
        if not inputs:
            inputs = {v for v in vertices if not parents_map.get(v, [])}
        if not outputs:
            outputs = {v for v in vertices if not children_map.get(v, [])}
            
        return vertices, parents_map, inputs, outputs
    elif isinstance(dag, list):
        # Check if it's a list of edges or list of parents
        # Heuristic: if elements are lists of ints, and len(dag) > 0, check first element
        if len(dag) > 0:
            first = dag[0]
            if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], int):
                # Could be parents or edges.
                # If it's edges, usually [u, v]. If parents, [p1, p2...].
                # Hard to distinguish without more info. 
                # Common convention for "list of parents" is dag[i] = parents of i.
                # Common convention for "list of edges" is list of [u, v].
                # Let's assume list of parents first, as "dag has topologically ordered parents" suggests vertex-indexed structure.
                # If it fails (e.g. index out of bounds or negative), we might need edge parsing.
                # For now, assume list of parents.
                pass
            else:
                # Maybe list of edges?
                pass
        
        n = len(dag)
        vertices = list(range(n))
        parents_map = {i: list(dag[i]) for i in range(n)}
        
        children_map = defaultdict(list)
        for v in vertices:
            for p in parents_map.get(v, []):
                children_map[p].append(v)
                
        inputs = {v for v in vertices if not parents_map.get(v, [])}
        outputs = {v for v in vertices if not children_map.get(v, [])}
        
        return vertices, parents_map, inputs, outputs
    else:
        raise ValueError("Unsupported DAG format")

def pebble(dag, fast_memory_budget):
    """
    Minimize communication for DAG pebbling.
    """
    deadline = time.time() + 85  # Safety margin
    
    try:
        vertices, parents_map, inputs, outputs = _parse_dag(dag)
    except Exception:
        return []

    n = len(vertices)
    if n == 0:
        return []
    
    # Build adjacency: children of each vertex
    children = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children[p].append(v)
    
    # Topological order
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        topo_order = list(vertices)
        
    pos_map = {v: i for i, v in enumerate(topo_order)}

    fast_mem = set()
    slow_mem = set(inputs)
    operations = []
    computed = set()
    
    for v in topo_order:
        if time.time() > deadline:
            break
        
        parents_of_v = parents_map.get(v, [])
        parents_in_fast = all(p in fast_mem for p in parents_of_v)
        
        if not parents_in_fast:
            for p in parents_of_v:
                if p not in fast_mem:
                    if p in slow_mem:
                        operations.append(['load', p])
                        fast_mem.add(p)
                    else:
                        pass
        
        needed_slots = len(fast_mem) + 1
        if needed_slots > fast_memory_budget:
            # Free space
            # Find deletable vertices
            # A vertex u is deletable if it has no unprocessed children
            # Unprocessed child means child c where c not in computed
            
            deletable = []
            for u in fast_mem:
                if u == v:
                    continue
                has_unprocessed_child = False
                for c in children[u]:
                    if c not in computed:
                        has_unprocessed_child = True
                        break
                if not has_unprocessed_child:
                    deletable.append(u)
            
            deletable_set = set(deletable)
            parents_set = set(parents_of_v)
            
            while len(fast_mem) + 1 > fast_memory_budget and fast_mem:
                to_delete = None
                # Prefer deletable non-parents
                for u in list(fast_mem):
                    if u not in parents_set and u in deletable_set:
                        to_delete = u
                        break
                if to_delete is None:
                    # Delete any non-parent (even if it has unprocessed children? 
                    # No, that would break computation. 
                    # If we MUST delete a vertex with unprocessed children, we must store it.
                    # But if we store it, we can reload it later.
                    # However, the current logic only deletes if it's "safe" (no unprocessed children) 
                    # OR if we are forced to delete a parent (which we handle by storing first).
                    # What if we are forced to delete a non-parent that HAS unprocessed children?
                    # That would mean we need to evict it to make room, but it's needed later.
                    # We should store it.
                    # The previous code deleted ANY non-parent if no deletable found.
                    # This is risky. It should store it.
                    
                    # Let's find any non-parent
                    for u in list(fast_mem):
                        if u not in parents_set:
                            to_delete = u
                            break
                            
                if to_delete is None:
                    # Must delete a parent
                    for u in list(fast_mem):
                        if u in parents_set:
                            if u not in slow_mem:
                                operations.append(['store', u])
                                slow_mem.add(u)
                            operations.append(['delete', u])
                            fast_mem.discard(u)
                            break
                    break
                else:
                    # If to_delete is not in deletable_set, it has unprocessed children.
                    # We must store it before deleting.
                    if to_delete not in slow_mem:
                        operations.append(['store', to_delete])
                        slow_mem.add(to_delete)
                    operations.append(['delete', to_delete])
                    fast_mem.discard(to_delete)
                    
        operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
    
    # Finalize outputs
    for v in topo_order:
        if time.time() > deadline:
            break
        if v in outputs:
            if v in fast_mem:
                if v not in slow_mem:
                    operations.append(['store', v])
                    slow_mem.add(v)
                operations.append(['delete', v])
                fast_mem.discard(v)
    
    # Clean up
    for v in list(fast_mem):
        if time.time() > deadline:
            break
        if v not in slow_mem:
            operations.append(['store', v])
            slow_mem.add(v)
        operations.append(['delete', v])
        fast_mem.discard(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
