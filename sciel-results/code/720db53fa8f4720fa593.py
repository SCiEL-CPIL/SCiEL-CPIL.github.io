import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def can_add(adj, u, v):
        """
        Check if adding edge (u,v) violates K3,3-free condition.
        Condition: No 3 vertices have 3 common neighbors.
        Adding (u,v) creates a violation if there exists a set of 3 vertices S
        such that |N(S)| becomes 3.
        
        As derived:
        1. For any a, b in N(v):
           v is a common neighbor of {u, a, b} after adding (u,v) IF u is adjacent to a,b? No.
           v is a neighbor of u (new), a, b.
           So v is a common neighbor of {u, a, b} if a,b are neighbors of v.
           We need |N({u,a,b})| <= 2.
           Current common neighbors of {u,a,b} are N(u) & N(a) & N(b).
           Let C = N(u) & N(a) & N(b).
           Since v is not in N(u) yet, v is not in C.
           After adding (u,v), v becomes a common neighbor of {u,a,b} IF v is in N(a) and N(b).
           We are iterating a,b in N(v), so v IS in N(a) and N(b).
           So new common neighbors = C ∪ {v}.
           We need |C ∪ {v}| <= 2.
           Since v not in C, this means |C| + 1 <= 2 => |C| <= 1.
           So check: for all a,b in N(v), (adj[u] & adj[a] & adj[b]).bit_count() <= 1.
           
        2. Symmetrically for u:
           For any a, b in N(u):
           u becomes a common neighbor of {v, a, b}.
           Check: for all a,b in N(u), (adj[v] & adj[a] & adj[b]).bit_count() <= 1.
        """
        if adj[u] & (1 << v):
            return False
            
        # Check condition 1: a, b in N(v)
        # Neighbors of v excluding u (u is not neighbor yet)
        mask_v = adj[v]
        # Iterate over pairs in N(v)
        # Optimization: if degree is high, this is slow. But max deg is ~15.
        # 15 choose 2 is 105 checks. Each is a few bit ops. Very fast.
        
        # Get list of neighbors of v
        neigh_v = []
        for i in range(n):
            if i != u and (mask_v & (1 << i)):
                neigh_v.append(i)
                
        for i in range(len(neigh_v)):
            a = neigh_v[i]
            adj_a = adj[a]
            for j in range(i + 1, len(neigh_v)):
                b = neigh_v[j]
                adj_b = adj[b]
                # Common neighbors of u, a, b
                common = adj[u] & adj_a & adj_b
                if common.bit_count() > 1:
                    return False
        
        # Check condition 2: a, b in N(u)
        mask_u = adj[u]
        neigh_u = []
        for i in range(n):
            if i != v and (mask_u & (1 << i)):
                neigh_u.append(i)
                
        for i in range(len(neigh_u)):
            a = neigh_u[i]
            adj_a = adj[a]
            for j in range(i + 1, len(neigh_u)):
                b = neigh_u[j]
                adj_b = adj[b]
                common = adj[v] & adj_a & adj_b
                if common.bit_count() > 1:
                    return False
                    
        return True

    def get_edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_init():
        """Construct a valid graph using greedy addition."""
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def simulated_annealing(adj, max_time):
        """Run Simulated Annealing to maximize edges."""
        current_adj = adj[:]
        current_edges = get_edge_count(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Precompute missing and existing edges for faster sampling?
        # We can just pick random pairs and check.
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        
        while time.time() < max_time:
            if T < T_min:
                break
                
            # Random Move
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to add a random missing edge
                if not missing_edges:
                    break
                idx = random.randint(0, len(missing_edges) - 1)
                u, v = missing_edges[idx]
                
                if can_add(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    missing_edges.pop(idx)
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                else:
                    # Rejection logic could be added, but for maximization with hard constraints,
                    # we just don't add it. To escape local optima, we need removals.
                    pass
            else:
                # Try to remove a random existing edge (to allow reconfiguration)
                # Pick a random vertex u, then a random neighbor v
                u = random.randint(0, n - 1)
                if current_adj[u] == 0:
                    continue
                # Get neighbors of u
                neigh_u = []
                for i in range(n):
                    if i != u and (current_adj[u] & (1 << i)):
                        neigh_u.append(i)
                if not neigh_u:
                    continue
                v = random.choice(neigh_u)
                
                # Remove (u, v)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges -= 1
                
                # Add to missing
                missing_edges.append((u, v))
                
                # With some probability, immediately try to add a different edge
                # This is effectively a "swap" or "kick"
                if random.random() < 0.8 and missing_edges:
                    # Pick a random missing edge to try to add
                    idx2 = random.randint(0, len(missing_edges) - 1)
                    u2, v2 = missing_edges[idx2]
                    if can_add(current_adj, u2, v2):
                        current_adj[u2] |= (1 << v2)
                        current_adj[v2] |= (1 << u2)
                        current_edges += 1
                        missing_edges.pop(idx2)
                        
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = current_adj[:]
                    else:
                        # If we can't add, we stay with the removed edge (worse state)
                        # But SA usually accepts worse states. Here, we just revert if it doesn't lead to improvement?
                        # Actually, standard SA accepts worse states with probability exp(-dE/T).
                        # Since our objective is just edge count, and constraint is hard,
                        # we can treat "edge count" as energy. Lower is better? No, higher is better.
                        # So we want to maximize.
                        # If we remove an edge, we lose 1. If we don't add one, we are at -1.
                        # We should accept this with probability exp(1/T).
                        # This allows escaping local optima.
                        if random.random() > (T / 1.0): # exp(1/T) approx
                             # Revert
                             current_adj[u] |= (1 << v)
                             current_adj[v] |= (1 << u)
                             current_edges += 1
                             missing_edges.pop(idx2) # remove the one we tried to add
                             # Wait, we added (u,v) to missing, then tried to add (u2,v2).
                             # If we revert, we put (u,v) back, and (u2,v2) stays missing.
                             # The list management is getting tricky.
                             pass
                else:
                     # If we just removed, we accept the worse state with prob exp(1/T)
                     if random.random() > (T / 1.0):
                         # Revert
                         current_adj[u] |= (1 << v)
                         current_adj[v] |= (1 << u)
                         current_edges += 1
                         missing_edges.pop(missing_edges.index((u, v)))
                         
            T *= cooling
            
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        random.seed(restart_count * 12345)
        
        # Greedy Init
        adj = greedy_init()
        edges = get_edge_count(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # Allocate time for SA
        # If we have plenty of time, do a longer SA.
        time_left = deadline - 5 - time.time()
        sa_time = min(5.0, time_left * 0.5)
        
        if sa_time > 0.1:
            adj, edges = simulated_annealing(adj, start_time + time.time() - start_time + sa_time)
            
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
                
        # If time is running out, stop
        if time_left < 10:
            break

    # Final polish: Greedy fill on best_adj
    if time.time() < deadline - 1:
        # Try to add any missing edges greedily
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (best_adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if can_add(best_adj, u, v):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
        best_edges = get_edge_count(best_adj)

    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
