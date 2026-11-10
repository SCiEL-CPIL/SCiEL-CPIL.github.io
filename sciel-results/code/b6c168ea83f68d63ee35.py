import random
import time
import math

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    # Precompute all triples for final validation, though we aim to keep it valid always
    # Actually, we don't need triples list if we use the fast check.
    
    # Adjacency as list of bitmasks
    adj = [0] * n
    
    def is_valid_adj(adj):
        """Check if adj is K3,3-free."""
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (adj[i] & adj[j] & adj[k]).bit_count() > 2:
                        return False
        return True

    def can_add_edge(adj, u, v):
        """
        Check if edge (u,v) can be added without creating K3,3.
        Assumes (u,v) is NOT currently an edge.
        """
        if (adj[u] >> v) & 1:
            return True
            
        # Check triples involving u: (u, x, y)
        # New common neighbors for (u, x, y) will include v if v is neighbor of x and y.
        # So we need |N(u) & N(x) & N(y)| <= 1 for all x, y in N(v) \ {u}
        # Note: x, y must be distinct from u and v.
        # Since u is not in N(v) initially? No, u and v are not connected, so u not in N(v) and v not in N(u).
        # So x, y are from N(v).
        
        # Iterate over pairs of neighbors of v
        # We need to check if there exist x, y in N(v) such that |N(u) & N(x) & N(y)| == 2
        # If such x, y exist, adding (u,v) makes |N(u) & N(x) & N(y)| become 3 (adding v), which is invalid.
        
        neighbors_v = [i for i in range(n) if i != v and (adj[v] >> i) & 1]
        
        for i in range(len(neighbors_v)):
            x = neighbors_v[i]
            if x == u: continue
            for j in range(i+1, len(neighbors_v)):
                y = neighbors_v[j]
                if y == u: continue
                # Check common neighbors of u, x, y
                common = adj[u] & adj[x] & adj[y]
                if common.bit_count() > 1:
                    return False
        
        # Check triples involving v: (v, x, y)
        # Similarly, for x, y in N(u) \ {v}, we need |N(v) & N(x) & N(y)| <= 1
        neighbors_u = [i for i in range(n) if i != u and (adj[u] >> i) & 1]
        
        for i in range(len(neighbors_u)):
            x = neighbors_u[i]
            if x == v: continue
            for j in range(i+1, len(neighbors_u)):
                y = neighbors_u[j]
                if y == v: continue
                common = adj[v] & adj[x] & adj[y]
                if common.bit_count() > 1:
                    return False
                    
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_construct():
        """Construct a valid graph using greedy edge addition."""
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def random_valid_graph():
        """Generate a random graph and prune it to be valid."""
        adj = [0] * n
        # Start with a sparse random graph
        p = 0.3
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Prune invalid edges
        changed = True
        while changed:
            changed = False
            # Check all triples
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        common = adj[a] & adj[b] & adj[c]
                        if common.bit_count() > 2:
                            # Remove a random edge incident to one of the common neighbors
                            common_list = [i for i in range(n) if (common >> i) & 1]
                            v = random.choice(common_list)
                            # Remove edge (v, a) or (v, b) or (v, c)? 
                            # Removing (v, a) reduces common neighbors of (a,b,c) by 1?
                            # Yes, if v is a common neighbor, removing (v,a) means v is no longer in N(a), so not in intersection.
                            adj[a] &= ~(1 << v)
                            adj[v] &= ~(1 << a)
                            changed = True
                            break
                    if changed: break
                if changed: break
        return adj

    def simulated_annealing(adj, time_limit):
        """Run simulated annealing to improve the graph."""
        current_adj = adj[:]
        current_edges = edge_count(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        T = 10.0  # Initial temperature
        cooling = 0.999
        
        start_time = time.time()
        steps = 0
        
        while time.time() < time_limit:
            steps += 1
            move_type = random.random()
            
            new_adj = current_adj[:]
            delta_edges = 0
            valid = True
            
            if move_type < 0.4:
                # Add edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((current_adj[i] >> j) & 1)]
                if not missing:
                    T *= cooling
                    continue
                u, v = random.choice(missing)
                if can_add_edge(current_adj, u, v):
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    delta_edges = 1
                else:
                    valid = False
            elif move_type < 0.7:
                # Remove edge
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if (current_adj[i] >> j) & 1]
                if not existing:
                    T *= cooling
                    continue
                u, v = random.choice(existing)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                delta_edges = -1
            else:
                # Swap: remove one, add one
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if (current_adj[i] >> j) & 1]
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((current_adj[i] >> j) & 1)]
                if not existing or not missing:
                    T *= cooling
                    continue
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    valid = False
                else:
                    # Tentatively remove u1,v1
                    new_adj[u1] &= ~(1 << v1)
                    new_adj[v1] &= ~(1 << u1)
                    # Check if u2,v2 can be added
                    if can_add_edge(new_adj, u2, v2):
                        new_adj[u2] |= (1 << v2)
                        new_adj[v2] |= (1 << u2)
                        delta_edges = 0 # Net change 0, but structure changes
                    else:
                        valid = False
                        # Rollback removal for consistency if we were to keep it, but we reject anyway
                        new_adj[u1] |= (1 << v1)
                        new_adj[v1] |= (1 << u1)
                        
            if not valid:
                T *= cooling
                continue
                
            new_edges = current_edges + delta_edges
            
            # Acceptance criterion
            if new_edges > current_edges:
                current_adj = new_adj
                current_edges = new_edges
                if current_edges > best_edges:
                    best_adj = current_adj[:]
                    best_edges = current_edges
            else:
                # Delta E = -(new_edges - current_edges) = current_edges - new_edges >= 0
                delta_E = current_edges - new_edges
                if delta_E > 0:
                    prob = math.exp(-delta_E / T)
                    if random.random() < prob:
                        current_adj = new_adj
                        current_edges = new_edges
                # If delta_E == 0, accept with some probability to explore
                else:
                    if random.random() < 0.5:
                        current_adj = new_adj
                        
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_adj, best_edges

    best_edges = 0
    best_adj = None
    
    # Initial greedy
    adj = greedy_construct()
    ec = edge_count(adj)
    if ec > best_edges:
        best_edges = ec
        best_adj = adj[:]
        
    # Run SA from greedy
    if time.time() < deadline - 20:
        sa_adj, sa_edges = simulated_annealing(adj, time.time() + 30)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj[:]

    # Multiple random restarts with SA
    while time.time() < deadline - 10:
        adj = random_valid_graph()
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
        # Run SA for a short duration
        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break
        sa_adj, sa_edges = simulated_annealing(adj, time.time() + min(10, remaining))
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj[:]

    # Final polish: try to add any missing edges
    for i in range(n):
        for j in range(i+1, n):
            if not ((best_adj[i] >> j) & 1):
                if can_add_edge(best_adj, i, j):
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)
                    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Found {len(edges)} edges in {end-start:.2f}s")
    # Verify validity
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if (adj[i] & adj[j] & adj[k]).bit_count() > 2:
                    valid = False
    print(f"Valid: {valid}")
