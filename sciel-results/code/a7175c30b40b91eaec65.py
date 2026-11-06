"""Pebbling solver using Dijkstra on state space."""
# EVOLVE-BLOCK-START
import heapq
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55.0
    
    parents = dag.get('parents', [])
    outputs = dag.get('outputs', [])
    n_vertices = len(parents)
    
    if n_vertices == 0:
        return []
    
    input_vertices = [i for i in range(n_vertices) if len(parents[i]) == 0]
    output_set = frozenset(outputs)
    
    initial_red = frozenset()
    initial_blue = frozenset(input_vertices)
    
    start_state = (initial_red, initial_blue)
    
    dist = {start_state: 0}
    parent_map = {}
    heap = [(0, 0, initial_red, initial_blue)]
    step_counter = 0
    visited = set()
    
    best_solution = None
    
    while heap:
        if time.time() > deadline:
            break
        
        cost, _, red, blue = heapq.heappop(heap)
        state = (red, blue)
        
        if state in visited:
            continue
        visited.add(state)
        
        if output_set.issubset(blue):
            # Reconstruct
            path = []
            cur = state
            while cur in parent_map:
                prev, op, v = parent_map[cur]
                path.append([op, v])
                cur = prev
            path.reverse()
            best_solution = path
            break
        
        # COMPUTE
        for v in range(n_vertices):
            if v in red or v in blue:
                continue
            if all(p in red for p in parents[v]):
                nr = red | frozenset((v,))
                nb = blue
                ns = (nr, nb)
                if cost < dist.get(ns, float('inf')):
                    dist[ns] = cost
                    parent_map[ns] = (state, 'compute', v)
                    step_counter += 1
                    heapq.heappush(heap, (cost, step_counter, nr, nb))
        
        # LOAD
        for v in blue:
            if v not in red:
                nr = red | frozenset((v,))
                nb = blue
                ns = (nr, nb)
                nc = cost + 1
                if nc < dist.get(ns, float('inf')):
                    dist[ns] = nc
                    parent_map[ns] = (state, 'load', v)
                    step_counter += 1
                    heapq.heappush(heap, (nc, step_counter, nr, nb))
        
        # STORE
        for v in red:
            nr = red - frozenset((v,))
            nb = blue | frozenset((v,))
            ns = (nr, nb)
            nc = cost + 1
            if nc < dist.get(ns, float('inf')):
                dist[ns] = nc
                parent_map[ns] = (state, 'store', v)
                step_counter += 1
                heapq.heappush(heap, (nc, step_counter, nr, nb))
        
        # DELETE
        for v in red:
            nr = red - frozenset((v,))
            nb = blue
            ns = (nr, nb)
            if cost < dist.get(ns, float('inf')):
                dist[ns] = cost
                parent_map[ns] = (state, 'delete', v)
                step_counter += 1
                heapq.heappush(heap, (cost, step_counter, nr, nb))
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    return _fallback(parents, outputs, input_vertices, n_vertices)

def _fallback(parents, outputs, input_vertices, n_vertices):
    ops = []
    for v in input_vertices:
        ops.append(['load', v])
    for v in range(n_vertices):
        if v in input_vertices:
            continue
        ops.append(['compute', v])
        if v not in outputs:
            ops.append(['store', v])
    for v in outputs:
        if v not in input_vertices:
            ops.append(['store', v])
    return ops

# EVOLVE-BLOCK-END
