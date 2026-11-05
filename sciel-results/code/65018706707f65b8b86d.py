"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    """Construct a K2,3-free graph on 20 vertices maximizing edges."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 20
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set()
    
    # Use bitmasks for fast common-neighbor checks
    # adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    
    def common_neighbors(u, v):
        """Return count of common neighbors of u and v."""
        return bin(adj[u] & adj[v]).count('1')
    
    def is_valid_add(u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # Adding edge (u,v) creates K2,3 if there exist 3 vertices 
        # that are common neighbors of u and v (after adding)
        # Actually: K2,3 has two vertices on one side, three on the other.
        # The two vertices on one side must have at least 3 common neighbors.
        # When we add edge (u,v), u and v become adjacent.
        # For K2,3: we need two vertices with >=3 common neighbors.
        # The new edge (u,v) means u and v are now a pair. 
        # They have common neighbors = N(u) ∩ N(v) (excluding each other since simple graph)
        # Wait: u's neighbors include v now, v's neighbors include u now.
        # Common neighbors of u and v are vertices w such that w~u and w~v.
        # u is not a neighbor of u, v is not a neighbor of v.
        # So common neighbors of u,v after adding = (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) 
        # but u∉N(u), v∉N(v), so = (N(u) ∩ N(v)) ∪ (if v∈N(u) then... no)
        # Actually N(u) after adding includes v. N(v) after adding includes u.
        # Common neighbors of u and v = {w : w∈N(u) and w∈N(v)}
        # = (N_old(u) ∪ {v}) ∩ (N_old(v) ∪ {u})
        # Since v∉N_old(u) and u∉N_old(v) (simple graph, no self-loops):
        # = (N_old(u) ∩ N_old(v)) ∪ ({v} ∩ N_old(v)) ∪ (N_old(u) ∩ {u}) ∪ ({v} ∩ {u})
        # {v} ∩ N_old(v) = empty (no self-loop)
        # N_old(u) ∩ {u} = empty
        # So common neighbors = N_old(u) ∩ N_old(v)
        # But wait, we also need to check ALL pairs, not just (u,v).
        # Adding edge (u,v) could create K2,3 for other pairs too?
        # No! Adding an edge only affects the common neighbor counts of pairs 
        # that include u or v. Specifically, for any pair (x,y):
        # - If x=u and y=v: their common neighbors change
        # - If x=u, y≠v: y's neighbor set unchanged, u's gains v. 
        #   Common neighbors of u,y: N(u)∩N(y). Before: N_old(u)∩N(y). After: (N_old(u)∪{v})∩N(y) = N_old(u)∩N(y) ∪ ({v}∩N(y)).
        #   So if v∈N(y), then common neighbors of u,y increase by 1.
        #   This could make it 3 if it was 2 before.
        # Similarly for pairs (v,y) where y≠u.
        
        # So we need to check:
        # 1. Common neighbors of (u,v) < 3
        # 2. For each neighbor y of v (y≠u): common neighbors of (u,y) after < 3
        # 3. For each neighbor x of u (x≠v): common neighbors of (x,v) after < 3
        
        # Check 1: common neighbors of u and v
        cn_uv = bin(adj[u] & adj[v]).count('1')
        if cn_uv >= 3:
            return False
        
        # Check 2: for each y in N(v), y≠u, common neighbors of (u,y) after adding
        # After adding, N(u) gains v. So common neighbors of (u,y) = (N_old(u)∪{v}) ∩ N(y)
        # = N_old(u)∩N(y) + (1 if v∈N(y) else 0)
        # Note: v∈N(y) iff y∈N(v), which is true by our loop.
        # So common neighbors of (u,y) after = cn_old(u,y) + 1
        for y in range(n):
            if y == u or y == v:
                continue
            if adj[v] & (1 << y):  # y is neighbor of v
                cn_uy = bin(adj[u] & adj[y]).count('1')
                # After adding, this becomes cn_uy + 1 (since v is now in N(u) and v is in N(y))
                if cn_uy + 1 >= 3:
                    return False
        
        # Check 3: for each x in N(u), x≠v, common neighbors of (x,v) after adding
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[u] & (1 << x):  # x is neighbor of u
                cn_xv = bin(adj[x] & adj[v]).count('1')
                # After adding, N(v) gains u. Common neighbors of (x,v) = N(x) ∩ (N_old(v)∪{u})
                # = N(x)∩N_old(v) + (1 if u∈N(x) else 0)
                # u∈N(x) iff x∈N(u), which is true.
                if cn_xv + 1 >= 3:
                    return False
        
        return True
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.remove((min(u,v), max(u,v)))
    
    # Greedy initialization with degree-biased ordering
    # Sort vertices by some heuristic, try to add edges greedily
    random.seed(42)
    
    # Try multiple random greedy initializations
    best_edges = 0
    best_adj = None
    best_edge_set = None
    
    for init_attempt in range(50):
        if time.time() > deadline - 60:
            break
        
        adj_try = [0] * n
        es_try = set()
        
        # Random order of edges, but bias towards high-degree potential
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        for (u, v) in edge_order:
            # Check if adding this edge is valid
            # Temporarily check
            cn_uv = bin(adj_try[u] & adj_try[v]).count('1')
            if cn_uv >= 3:
                continue
            
            valid = True
            for y in range(n):
                if y == u or y == v:
                    continue
                if adj_try[v] & (1 << y):
                    cn_uy = bin(adj_try[u] & adj_try[y]).count('1')
                    if cn_uy + 1 >= 3:
                        valid = False
                        break
            if not valid:
                continue
            for x in range(n):
                if x == u or x == v:
                    continue
                if adj_try[u] & (1 << x):
                    cn_xv = bin(adj_try[x] & adj_try[v]).count('1')
                    if cn_xv + 1 >= 3:
                        valid = False
                        break
            if not valid:
                continue
            
            adj_try[u] |= (1 << v)
            adj_try[v] |= (1 << u)
            es_try.add((min(u,v), max(u,v)))
        
        if len(es_try) > best_edges:
            best_edges = len(es_try)
            best_adj = adj_try[:]
            best_edge_set = set(es_try)
    
    # Start from best greedy solution
    adj = best_adj[:]
    edge_set = set(best_edge_set)
    
    # Simulated annealing / local search with edge swaps
    # Move types:
    # 1. Add an edge (if valid)
    # 2. Remove an edge and add two edges (if valid) - 1-out 2-in
    # 3. Remove two edges and add one edge (if valid) - 2-out 1-in (to escape local optima)
    
    current_edges = len(edge_set)
    best_edges_found = current_edges
    best_adj_copy = adj[:]
    best_es_copy = set(edge_set)
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5 and T > T_min:
        if no_improve > max_no_improve:
            # Restart with perturbation
            no_improve = 0
            T = max(T, 0.5)
            # Perturb: remove some random edges
            for _ in range(random.randint(3, 8)):
                if not edge_set:
                    break
                e = random.choice(list(edge_set))
                remove_edge(e[0], e[1])
            current_edges = len(edge_set)
            continue
        
        move_type = random.random()
        if move_type < 0.4:
            # Try adding an edge
            # Sample a non-edge
            for _ in range(100):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if (min(u,v), max(u,v)) in edge_set:
                    continue
                # Check validity
                if is_valid_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    no_improve = 0
                    if current_edges > best_edges_found:
                        best_edges_found = current_edges
                        best_adj_copy = adj[:]
                        best_es_copy = set(edge_set)
                    break
                else:
                    no_improve += 1
        elif move_type < 0.7:
            # 1-out 2-in: remove one edge, add two
            if not edge_set:
                continue
            e_remove = random.choice(list(edge_set))
            u1, v1 = e_remove
            remove_edge(u1, v1)
            current_edges -= 1
            
            # Try to add two edges
            added = 0
            added_edges = []
            # Sample candidate edges
            candidates = []
            for _ in range(50):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                e = (min(u,v), max(u,v))
                if e in edge_set:
                    continue
                if e == e_remove:
                    continue
                candidates.append((u, v, e))
            
            random.shuffle(candidates)
            for (u, v, e) in candidates:
                if added >= 2:
                    break
                if is_valid_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    added += 1
                    added_edges.append(e)
            
            if added == 0:
                # Revert
                add_edge(u1, v1)
                current_edges += 1
                no_improve += 1
            elif added == 1:
                # Net zero change, accept with some probability
                if random.random() < 0.5:
                    no_improve += 1
                else:
                    # Revert the added edge, keep removal
                    ae = added_edges[0]
                    remove_edge(ae[0], ae[1])
                    current_edges -= 1
                    add_edge(u1, v1)
                    current_edges += 1
                    no_improve += 1
            else:  # added == 2, net +1
                no_improve = 0
                if current_edges > best_edges_found:
                    best_edges_found = current_edges
                    best_adj_copy = adj[:]
                    best_es_copy = set(edge_set)
        else:
            # 2-out 1-in: remove two edges, add one (to explore)
            if len(edge_set) < 2:
                continue
            edges_list = list(edge_set)
            e1 = random.choice(edges_list)
            e2 = random.choice([e for e in edges_list if e != e1])
            u1, v1 = e1
            u2, v2 = e2
            remove_edge(u1, v1)
            remove_edge(u2, v2)
            current_edges -= 2
            
            # Try to add one edge
            added = False
            for _ in range(50):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                e = (min(u,v), max(u,v))
                if e in edge_set or e == e1 or e == e2:
                    continue
                if is_valid_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    added = True
                    break
            
            if not added:
                # Revert
                add_edge(u1, v1)
                add_edge(u2, v2)
                current_edges += 2
                no_improve += 1
            else:
                # Net -1, accept with low probability
                if random.random() < 0.1:
                    no_improve += 1
                else:
                    # Revert
                    ae_u = u
                    ae_v = v
                    remove_edge(ae_u, ae_v)
                    current_edges -= 1
                    add_edge(u1, v1)
                    add_edge(u2, v2)
                    current_edges += 2
                    no_improve += 1
        
        T *= cooling
    
    # Final polish: try all possible edge additions
    for (u, v) in all_edges:
        if (u, v) in edge_set:
            continue
        if is_valid_add(u, v):
            add_edge(u, v)
            current_edges += 1
    
    if current_edges > best_edges_found:
        best_edges_found = current_edges
        best_adj_copy = adj[:]
        best_es_copy = set(edge_set)
    
    # Return best found
    return [list(e) for e in best_es_copy]
# EVOLVE-BLOCK-END
