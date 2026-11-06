import itertools
import time
import random

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin for 300s limit
    
    # Helper to generate Paley graph
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges
    
    # Helper to check if graph is (5,5)-free using bitmasks
    def is_55_free(n, edges):
        if time.time() - start_time > timeout:
            return False
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for combo in itertools.combinations(range(n), 5):
            is_k5 = True
            is_i5 = True
            for i in range(5):
                u = combo[i]
                for j in range(i + 1, 5):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_i5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_i5:
                        break
                if not is_k5 and not is_i5:
                    break
            if is_k5 or is_i5:
                return False
        return True
    
    # Fallback: Paley 37
    n_seed, edges_seed = get_paley(37)
    
    # 1. Try Paley(41)
    n_paley41, edges_41 = get_paley(41)
    if is_55_free(n_paley41, edges_41):
        return n_paley41, edges_41
    
    # 2. Try to extend Paley(37) to 38 vertices
    n_paley37, edges_37 = get_paley(37)
    
    adj_37 = [0] * n_paley37
    for u, v in edges_37:
        adj_37[u] |= (1 << v)
        adj_37[v] |= (1 << u)
        
    # Precompute K4s and I4s in G_37
    k4s = []
    i4s = []
    for combo in itertools.combinations(range(n_paley37), 4):
        is_k4 = True
        is_i4 = True
        for i in range(4):
            u = combo[i]
            for j in range(i + 1, 4):
                v = combo[j]
                if adj_37[u] & (1 << v):
                    is_i4 = False
                else:
                    is_k4 = False
            if not is_k4 and not is_i4:
                break
        if is_k4:
            k4s.append(combo)
        if is_i4:
            i4s.append(combo)
            
    # Try to find S (subset of vertices connected to new vertex 37)
    # Conditions: No K4 is subset of S, No I4 is disjoint from S
    new_vertex = n_paley37
    found = False
    best_S = None
    
    # Local search with multiple restarts
    for restart in range(2000):
        if time.time() - start_time > timeout - 20:
            break
        
        random.seed(42 + restart)
        
        # Initialize S randomly
        S = set(random.sample(range(n_paley37), n_paley37 // 2))
        
        for _ in range(500):
            if time.time() - start_time > timeout - 20:
                break
            
            # Check violations
            viol_k4 = None
            for k4 in k4s:
                if all(v in S for v in k4):
                    viol_k4 = k4
                    break
            
            viol_i4 = None
            if not viol_k4:
                for i4 in i4s:
                    if all(v not in S for v in i4):
                        viol_i4 = i4
                        break
            
            if not viol_k4 and not viol_i4:
                found = True
                best_S = S
                break
            
            if viol_k4:
                v_to_remove = random.choice(viol_k4)
                S.remove(v_to_remove)
            elif viol_i4:
                v_to_add = random.choice(viol_i4)
                S.add(v_to_add)
        
        if found:
            break
    
    if found:
        new_edges = list(edges_37)
        for v in best_S:
            new_edges.append((v, new_vertex))
        n_new = n_paley37 + 1
        if is_55_free(n_new, new_edges):
            return n_new, new_edges
    
    # 3. Try local search for 38-vertex graph
    def try_find_local(n, max_attempts=3, max_steps=3000, seed_base=0):
        all_5sets = list(itertools.combinations(range(n), 5))
        edge_to_5sets = {}
        for idx, combo in enumerate(all_5sets):
            for i in range(5):
                for j in range(i + 1, 5):
                    edge = (combo[i], combo[j])
                    if edge not in edge_to_5sets:
                        edge_to_5sets[edge] = []
                    edge_to_5sets[edge].append(idx)
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 15:
                break
            
            random.seed(seed_base + attempt)
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_viol = 0
            for combo in all_5sets:
                is_k5 = True
                is_i5 = True
                for i in range(5):
                    u = combo[i]
                    for j in range(i + 1, 5):
                        v = combo[j]
                        if adj[u] & (1 << v):
                            is_i5 = False
                        else:
                            is_k5 = False
                        if not is_k5 and not is_i5:
                            break
                    if not is_k5 and not is_i5:
                        break
                if is_k5 or is_i5:
                    current_viol += 1
            
            for step in range(max_steps):
                if time.time() - start_time > timeout - 15:
                    break
                
                if current_viol == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append((i, j))
                    return n, edges
                
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                affected = edge_to_5sets.get((i, j), [])
                
                viol_before = 0
                for idx in affected:
                    combo = all_5sets[idx]
                    is_k5 = True
                    is_i5 = True
                    for a in range(5):
                        u = combo[a]
                        for b in range(a + 1, 5):
                            v = combo[b]
                            if adj[u] & (1 << v):
                                is_i5 = False
                            else:
                                is_k5 = False
                            if not is_k5 and not is_i5:
                                break
                        if not is_k5 and not is_i5:
                            break
                    if is_k5 or is_i5:
                        viol_before += 1
                
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
                viol_after = 0
                for idx in affected:
                    combo = all_5sets[idx]
                    is_k5 = True
                    is_i5 = True
                    for a in range(5):
                        u = combo[a]
                        for b in range(a + 1, 5):
                            v = combo[b]
                            if adj[u] & (1 << v):
                                is_i5 = False
                            else:
                                is_k5 = False
                            if not is_k5 and not is_i5:
                                break
                        if not is_k5 and not is_i5:
                            break
                    if is_k5 or is_i5:
                        viol_after += 1
                
                if viol_after <= viol_before or (viol_before > 0 and random.random() < 0.1):
                    current_viol += (viol_after - viol_before)
                else:
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
            
            if current_viol == 0:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return n, edges
        
        return None, None
    
    for target_n in [38, 39, 40]:
        if time.time() - start_time > timeout - 30:
            break
        
        n_found, edges_found = try_find_local(target_n, max_attempts=3, max_steps=3000, seed_base=target_n * 100)
        
        if n_found == target_n and edges_found is not None:
            if is_55_free(target_n, edges_found):
                return target_n, edges_found
    
    # Fallback: return the Paley 37
    return n_seed, edges_seed

# Test runner
if __name__ == "__main__":
    n, edges = construct()
    print(f"Vertices: {n}, Edges: {len(edges)}")
