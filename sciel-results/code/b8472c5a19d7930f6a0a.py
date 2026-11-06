import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0
    
    num_vertices = len(dag)
    children = [[] for _ in range(num_vertices)]
    for i in range(num_vertices):
        for p in dag[i]:
            children[p].append(i)
            
    inputs = [i for i in range(num_vertices) if not dag[i]]
    outputs = [i for i in range(num_vertices) if not children[i]]
    
    level = [0] * num_vertices
    for i in range(num_vertices):
        if dag[i]:
            level[i] = max(level[p] for p in dag[i]) + 1
        else:
            level[i] = 0
            
    max_level = max(level) if level else 0
    
    levels = [[] for _ in range(max_level + 1)]
    for i in range(num_vertices):
        levels[level[i]].append(i)
        
    state = [0] * num_vertices
    in_fast = set()
    ops = []
    
    # Count remaining children for each vertex
    remaining_children = [len(children[i]) for i in range(num_vertices)]
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
        
    for l in range(max_level + 1):
        if not check_time():
            break
            
        verts_in_level = levels[l]
        
        # Load parents
        needed_parents = set()
        for v in verts_in_level:
            for p in dag[v]:
                needed_parents.add(p)
                
        for p in needed_parents:
            if p not in in_fast:
                if state[p] == 0:
                    ops.append(["load", p])
                    state[p] = 1
                    in_fast.add(p)
                else:
                    in_fast.add(p)
                    
        # Compute vertices
        for v in verts_in_level:
            if not check_time():
                break
                
            ops.append(["compute", v])
            state[v] = 1
            in_fast.add(v)
            
            # Decrement remaining children for parents of v
            for p in dag[v]:
                remaining_children[p] -= 1
                
            # Memory management
            if len(in_fast) > fast_memory_budget:
                evicted = False
                # Evict vertices that have no remaining children
                # Prioritize older levels
                for prev_l in range(l - 1, -1, -1):
                    for u in levels[prev_l]:
                        if u in in_fast and remaining_children[u] == 0:
                            if u in outputs:
                                ops.append(["store", u])
                            else:
                                ops.append(["delete", u])
                            state[u] = 0
                            in_fast.remove(u)
                            evicted = True
                            break
                    if evicted:
                        break
                            
                if not evicted:
                    # If no vertex is safe to evict, we are in trouble.
                    # This can happen if budget is 1 and we have a chain?
                    # If budget is 1, we can only hold 1 vertex.
                    # Load p, compute v. p is no longer needed. Evict p.
                    # So it should work.
                    # If we can't evict, we might have to store v?
                    if v in in_fast and remaining_children[v] == 0 and v not in outputs:
                        ops.append(["delete", v])
                        state[v] = 0
                        in_fast.remove(v)
                    elif v in in_fast:
                        # If v is output or has children, we must keep it?
                        # But we are over budget.
                        # We might have to store it.
                        ops.append(["store", v])
                        state[v] = 0
                        in_fast.remove(v)
                        
        # Cleanup: evict all vertices from level l-1 that are no longer needed
        if l > 0:
            for u in levels[l - 1]:
                if u in in_fast and remaining_children[u] == 0:
                    if u in outputs:
                        ops.append(["store", u])
                    else:
                        ops.append(["delete", u])
                    state[u] = 0
                    in_fast.remove(u)
                    
    # Final cleanup
    for v in outputs:
        if state[v] == 1:
            ops.append(["store", v])
            state[v] = 0
            in_fast.remove(v)
            
    return ops
