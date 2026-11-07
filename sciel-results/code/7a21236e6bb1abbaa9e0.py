import random
import time

def construct():
    n = 15
    deadline = time.time() + 78.0
    
    # Precompute: for each pair (s,t), list of all middle pairs (u,v) such that
    # s-u-v-t is a valid path (all distinct). Then check if edges (s,u),(u,v),(v,t) all exist.
    # For speed: precompute for each (s,t) the list of (u,v) pairs.
    
    st_paths = {}
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            mids = []
            for u in range(n):
                if u == s or u == t:
                    continue
                for v in range(n):
                    if v == s or v == t or v == u:
                        continue
                    mids.append((u, v))
            st_paths[(s, t)] = mids
    
    all_edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges_list.append((i, j))
    all_edge_set = set(all_edges_list)
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
    
    def validate(adj_set, deg):
        """adj_set is a set of (u,v) tuples with u<v. deg is list of degrees."""
        # Group vertices by degree
        deg_to_verts = {}
        for i in range(n):
            d = deg[i]
            if d not in deg_to_verts:
                deg_to_verts[d] = []
            deg_to_verts[d].append(i)
        
        # For each degree value, check all pairs of vertices with that degree
        for d, verts in deg_to_verts.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    s, t = verts[i], verts[j]
                    # Check if path s-u-v-t exists
                    for u, v in st_paths[(s, t)]:
                        if (edge_key(s, u) in adj_set and 
                            edge_key(u, v) in adj_set and 
                            edge_key(v, t) in adj_set):
                            return False
        return True
    
    # Greedy initialization
    best_edges = set()
    best_count = 0
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        cur_edges = set()
        shuffled = list(all_edges_list)
        random.shuffle(shuffled)
        for e in shuffled:
            u, v = e
            cur_edges.add(e)
            if not validate(cur_edges, None):
                # Recompute degrees
                deg = [0] * n
                for a, b in cur_edges:
                    deg[a] += 1
                    deg[b] += 1
                if not validate(cur_edges, deg):
                    cur_edges.discard(e)
        
        if len(cur_edges) > best_count:
            best_edges = set(cur_edges)
            best_count = len(cur_edges)
    
    # Local search
    while time.time() < deadline:
        cur_edges = set(best_edges)
        deg = [0] * n
        for a, b in cur_edges:
            deg[a] += 1
            deg[b] += 1
        
        # Try adding edges
        non_edges = [e for e in all_edges_list if e not in cur_edges]
        random.shuffle(non_edges)
        improved = False
        for e in non_edges:
            if time.time() > deadline:
                break
            u, v = e
            new_deg = deg[:]
            new_deg[u] += 1
            new_deg[v] += 1
            cur_edges.add(e)
            if validate(cur_edges, new_deg):
                deg = new_deg
                improved = True
            else:
                cur_edges.discard(e)
        
        if len(cur_edges) > best_count:
            best_edges = set(cur_edges)
            best_count = len(cur_edges)
        
        # 1-swap
        if len(cur_edges) > 0:
            edge_list = list(cur_edges)
            random.shuffle(edge_list)
            non_edges = [e for e in all_edges_list if e not in cur_edges]
            swap_found = False
            for e_rem in edge_list:
                if time.time() > deadline:
                    break
                cur_edges.discard(e_rem)
                temp_deg = [0] * n
                for a, b in cur_edges:
                    temp_deg[a] += 1
                    temp_deg[b] += 1
                
                random.shuffle(non_edges)
                for e_add in non_edges[:20]:
                    if time.time() > deadline:
                        break
                    if e_add == e_rem:
                        continue
                    u, v = e_add
                    new_deg = temp_deg[:]
                    new_deg[u] += 1
                    new_deg[v] += 1
                    cur_edges.add(e_add)
                    if validate(cur_edges, new_deg):
                        deg = new_deg
                        swap_found = True
                        break
                    else:
                        cur_edges.discard(e_add)
                
                if swap_found:
                    if len(cur_edges) > best_count:
                        best_edges = set(cur_edges)
                        best_count = len(cur_edges)
                    break
                else:
                    cur_edges.add(e_rem)
            
            if swap_found:
                continue
    
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
