import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u: {u, a, b} where a, b are neighbors of v
        # We need |N(u) ∩ N(a) ∩ N(b)| <= 1 for all a, b in N(v) \ {u}
        neighbors_v = []
        mask_v = adj[v]
        # Exclude u from the list of neighbors to check? 
        # If a == u, the triple is {u, u, b} which is not a triple of distinct vertices.
        # The K3,3 condition applies to 3 distinct vertices.
        # So we only care about a, b distinct from u and v.
        while mask_v:
            lsb = mask_v & (-mask_v)
            idx = lsb.bit_length() - 1
            if idx != u:
                neighbors_v.append(idx)
            mask_v ^= lsb
            
        # Check all pairs a, b in neighbors_v
        for i in range(len(neighbors_v)):
            a = neighbors_v[i]
            for j in range(i + 1, len(neighbors_v)):
                b = neighbors_v[j]
                # Check if a and b are distinct from u and v (they are, by construction)
                # Check |N(u) ∩ N(a) ∩ N(b)|
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b} where a, b are neighbors of u
        # We need |N(v) ∩ N(a) ∩ N(b)| <= 1 for all a, b in N(u) \ {v}
        neighbors_u = []
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & (-mask_u)
            idx = lsb.bit_length() - 1
            if idx != v:
                neighbors_u.append(idx)
            mask_u ^= lsb

        for i in range(len(neighbors_u)):
            a = neighbors_u[i]
            for j in range(i + 1, len(neighbors_u)):
                b = neighbors_u[j]
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_fill(adj, deadline):
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def find_violation(adj):
        for x in range(n):
            for y in range(x + 1, n):
                for z in range(y + 1, n):
                    common = adj[x] & adj[y] & adj[z]
                    if common.bit_count() >= 3:
                        neighbors = []
                        mask = common
                        while mask and len(neighbors) < 3:
                            lsb = mask & (-mask)
                            idx = lsb.bit_length() - 1
                            neighbors.append(idx)
                            mask ^= lsb
                        return (x, y, z, neighbors[0], neighbors[1], neighbors[2])
        return None

    def repair_graph(adj, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            violation = find_violation(adj)
            if violation:
                x, y, z, n1, n2, n3 = violation
                edges_to_consider = [
                    (x, n1), (x, n2), (x, n3),
                    (y, n1), (y, n2), (y, n3),
                    (z, n1), (z, n2), (z, n3)
                ]
                u, v = random.choice(edges_to_consider)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                changed = True
        return adj

    def local_search(adj, deadline):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Greedy fill first
        current_adj = greedy_fill(current_adj, deadline)
        current_edges = count_edges(current_adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
            
        no_improve = 0
        max_no_improve = 100 # Increase patience
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            # Strategy 1: Try to add a random missing edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            
            if missing:
                u, v = random.choice(missing)
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    no_improve = 0
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                    continue # Restart loop with new state
            
            # Strategy 2: Try a swap (remove one edge, add another)
            # This helps rearrange the graph to allow future additions
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            if existing:
                # Try a few random swaps
                swap_success = False
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    rem_u, rem_v = random.choice(existing)
                    add_u, add_v = random.choice(missing) if missing else None
                    if add_u is None:
                        break
                    
                    # Tentatively remove (rem_u, rem_v)
                    current_adj[rem_u] &= ~(1 << rem_v)
                    current_adj[rem_v] &= ~(1 << rem_u)
                    
                    if is_valid_addition(current_adj, add_u, add_v):
                        current_adj[add_u] |= (1 << add_v)
                        current_adj[add_v] |= (1 << add_u)
                        current_edges += 0 # Edge count unchanged
                        swap_success = True
                        break
                    else:
                        # Revert
                        current_adj[rem_u] |= (1 << rem_v)
                        current_adj[rem_v] |= (1 << rem_u)
                
                if swap_success:
                    # After a swap, try to greedy fill to see if we can add more
                    current_adj = greedy_fill(current_adj, deadline)
                    current_edges = count_edges(current_adj)
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue
            
            # Strategy 3: Kick (remove random edges, then greedy fill)
            kick_size = random.choice([1, 2, 3])
            if existing:
                random.shuffle(existing)
                for _ in range(min(kick_size, len(existing))):
                    a, b = existing.pop()
                    current_adj[a] &= ~(1 << b)
                    current_adj[b] &= ~(1 << a)
                    current_edges -= 1
                    
                current_adj = greedy_fill(current_adj, deadline)
                current_edges = count_edges(current_adj)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                break

        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts
    num_restarts = 2000 # Increase restarts if faster
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random dense graph
        adj = [0] * n
        for u, v in all_pairs:
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Repair
        adj = repair_graph(adj, deadline)
        
        # Local search
        adj, edges = local_search(adj, deadline)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            adj, edges = local_search(best_adj, deadline)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
