import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set()
    
    # Precompute edge indices
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # For fast adjacency queries
    adj = [set() for _ in range(n)]
    deg = [0] * n
    
    def is_valid_add(u, v):
        """Check if adding edge (u,v) keeps the graph valid."""
        # After adding (u,v), for this edge:
        # For all s in adj[u] (s != v), t in adj[v] (t != u, t != s): deg[s] != deg[t]
        # Note: deg doesn't change for existing vertices when adding an edge between two new neighbors
        # Actually deg[u] and deg[v] increase by 1, but the constraint checks deg of s and t (endpoints)
        # s and t are not u or v (they're neighbors), so their degrees don't change
        # But wait: s could be adjacent to v? No, s is in adj[u], and we check s != v
        # t is in adj[v], t != u, t != s
        
        # The new edge (u,v) creates new 3-paths: s-u-v-t for s in adj[u]\{v}, t in adj[v]\{u}
        # We need deg[s] != deg[t] for all such pairs (s,t) with s != t
        
        neighbors_u = adj[u] - {v}
        neighbors_v = adj[v] - {u}
        
        for s in neighbors_u:
            for t in neighbors_v:
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
        
        # Also need to check: does adding this edge create violations for OTHER existing edges?
        # An existing edge (a,b) might now have new 3-paths involving the new edge
        # Specifically, if u or v is an endpoint of some existing edge, new paths may form
        # For edge (a,b) where a==u: paths s-a-b-t where s in adj[a]\{b}, t in adj[b]\{a}
        #   New s could be v (if v in adj[u]), new t could be... no, t is in adj[b]\{a}
        #   Actually, adding (u,v) means v is now in adj[u], so for edge (u,b) with b in adj[u],
        #   new paths v-u-b-t for t in adj[b]\{u}. Need deg[v] != deg[t] for all t in adj[b]\{u}
        #   But deg[v] just increased by 1! So we need to check with updated degrees.
        
        # This is getting complex. Let me just do a full validation after adding.
        # For efficiency, let's do the full check.
        
        # Tentatively add
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
        valid = True
        # Check all edges
        for a in range(n):
            for b in adj[a]:
                if a < b:
                    # Check edge (a,b): for all s in adj[a]\{b}, t in adj[b]\{a}, s!=t: deg[s] != deg[t]
                    for s in adj[a]:
                        if s == b:
                            continue
                        for t in adj[b]:
                            if t == a or t == s:
                                continue
                            if deg[s] == deg[t]:
                                valid = False
                                break
                    if not valid:
                        break
            if not valid:
                break
        
        # Rollback
        adj[u].remove(v)
        adj[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1
        
        return valid
    
    def full_validate():
        """Full validation of current graph."""
        for a in range(n):
            for b in adj[a]:
                if a < b:
                    for s in adj[a]:
                        if s == b:
                            continue
                        for t in adj[b]:
                            if t == a or t == s:
                                continue
                            if deg[s] == deg[t]:
                                return False
        return True
    
    best_edges = []
    best_count = 0
    
    def try_add_edge(u, v):
        nonlocal best_edges, best_count
        if (u, v) in edge_set or (v, u) in edge_set:
            return True
        if is_valid_add(u, v):
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edges = list(edge_set)
            return True
        return False
    
    # Strategy: Greedy + local search
    # Phase 1: Greedy construction
    random.seed(42)
    edge_order = all_edges[:]
    random.shuffle(edge_order)
    
    for u, v in edge_order:
        if time.time() > DEADLINE - 5:
            break
        try_add_edge(u, v)
    
    # Phase 2: Local search - try to add edges, remove conflicting ones
    # Represent state
    current_edges = set(best_edges)
    adj2 = [set() for _ in range(n)]
    deg2 = [0] * n
    for u, v in current_edges:
        adj2[u].add(v)
        adj2[v].add(u)
        deg2[u] += 1
        deg2[v] += 1
    
    def validate_state(adj_set, deg_arr):
        for a in range(n):
            for b in adj_set[a]:
                if a < b:
                    for s in adj_set[a]:
                        if s == b:
                            continue
                        for t in adj_set[b]:
                            if t == a or t == s:
                                continue
                            if deg_arr[s] == deg_arr[t]:
                                return False
        return True
    
    def can_add_edge(adj_set, deg_arr, u, v):
        """Check if edge (u,v) can be added to maintain validity."""
        if v in adj_set[u]:
            return True  # already there
        # Tentatively add
        adj_set[u].add(v)
        adj_set[v].add(u)
        deg_arr[u] += 1
        deg_arr[v] += 1
        
        valid = validate_state(adj_set, deg_arr)
        
        # Rollback
        adj_set[u].remove(v)
        adj_set[v].remove(u)
        deg_arr[u] -= 1
        deg_arr[v] -= 1
        
        return valid
    
    # Local search with restarts
    best_final_edges = list(current_edges)
    best_final_count = len(current_edges)
    
    def local_search(start_edges, time_limit):
        """Hill climbing: try to add edges, if stuck try swapping."""
        adj_local = [set() for _ in range(n)]
        deg_local = [0] * n
        edge_local = set()
        
        for u, v in start_edges:
            adj_local[u].add(v)
            adj_local[v].add(u)
            deg_local[u] += 1
            deg_local[v] += 1
            edge_local.add((u, v))
        
        best_local = set(edge_local)
        best_local_count = len(edge_local)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding each missing edge
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_local:
                        continue
                    if can_add_edge(adj_local, deg_local, u, v):
                        # Add it
                        edge_local.add((u, v))
                        adj_local[u].add(v)
                        adj_local[v].add(u)
                        deg_local[u] += 1
                        deg_local[v] += 1
                        improved = True
                        if len(edge_local) > best_local_count:
                            best_local_count = len(edge_local)
                            best_local = set(edge_local)
            # Try removing one edge and adding two others (2-opt)
            if not improved and time.time() < time_limit:
                # Try: remove edge e, then try to add two new edges
                edges_list = list(edge_local)
                random.shuffle(edges_list)
                for e in edges_list:
                    if time.time() > time_limit:
                        break
                    u, v = e
                    # Remove e
                    adj_local[u].remove(v)
                    adj_local[v].remove(u)
                    deg_local[u] -= 1
                    deg_local[v] -= 1
                    edge_local.remove(e)
                    
                    # Try to add two new edges
                    added = []
                    for a in range(n):
                        for b in range(a+1, n):
                            if (a, b) in edge_local or (a,b) == e:
                                continue
                            if can_add_edge(adj_local, deg_local, a, b):
                                edge_local.add((a, b))
                                adj_local[a].add(b)
                                adj_local[b].add(a)
                                deg_local[a] += 1
                                deg_local[b] += 1
                                added.append((a, b))
                                if len(added) >= 2:
                                    break
                        if len(added) >= 2:
                            break
                    
                    if len(added) >= 2 and len(edge_local) > best_local_count:
                        best_local_count = len(edge_local)
                        best_local = set(edge_local)
                        improved = True
                    else:
                        # Rollback: remove added, re-add e
                        for a, b in added:
                            edge_local.remove((a, b))
                            adj_local[a].remove(b)
                            adj_local[b].remove(a)
                            deg_local[a] -= 1
                            deg_local[b] -= 1
                        edge_local.add(e)
                        adj_local[u].add(v)
                        adj_local[v].add(u)
                        deg_local[u] += 1
                        deg_local[v] += 1
        
        return best_local, best_local_count
    
    # Run multiple local search iterations with different starting points
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Create perturbed start
        if r == 0:
            start = set(best_final_edges)
        else:
            # Start from best, remove some random edges, add some random
            start = set(best_final_edges)
            edges_list = list(start)
            random.shuffle(edges_list)
            for e in edges_list[:3]:
                start.remove(e)
            # Try to add some random edges
            for u, v in random.sample(all_edges, 5):
                if u < v and (u, v) not in start:
                    adj_tmp = [set() for _ in range(n)]
                    deg_tmp = [0] * n
                    for a, b in start:
                        adj_tmp[a].add(b)
                        adj_tmp[b].add(a)
                        deg_tmp[a] += 1
                        deg_tmp[b] += 1
                    if can_add_edge(adj_tmp, deg_tmp, u, v):
                        start.add((u, v))
        
        tl = min(time.time() + 2.0, DEADLINE - 5)
        result, count = local_search(start, tl)
        
        if count > best_final_count:
            best_final_count = count
            best_final_edges = list(result)
    
    # Final validation
    adj_final = [set() for _ in range(n)]
    deg_final = [0] * n
    for u, v in best_final_edges:
        adj_final[u].add(v)
        adj_final[v].add(u)
        deg_final[u] += 1
        deg_final[v] += 1
    
    if not validate_state(adj_final, deg_final):
        # Fallback to star
        best_final_edges = [(0, v) for v in range(1, n)]
    
    # Convert to list of [u, v] with proper types
    result = []
    for u, v in best_final_edges:
        result.append([int(u), int(v)])
    
    return result
