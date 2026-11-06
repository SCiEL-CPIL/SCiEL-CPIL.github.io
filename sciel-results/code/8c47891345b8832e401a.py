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
        
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old |N({u, a, b})| <= 1.
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b}
        # If u is in N(a) and N(b), then u is added to N({v, a, b}).
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                
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
        """Find a K3,3 violation. Returns (x, y, z, n1, n2, n3) where {x,y,z} has 3 common neighbors."""
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
        """Remove edges to eliminate K3,3 violations."""
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

    def try_edge_swap(adj, deadline):
        """Try to improve via edge swap: remove edge (a,b), add edge (c,d).
        Returns (success, new_adj, delta_edges) or (False, None, 0)."""
        existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if not existing or not missing:
            return False, None, 0
        
        a, b = random.choice(existing)
        c, d = random.choice(missing)
        
        # Skip if same edge or overlapping
        if (a, b) == (c, d) or (a, b) == (d, c):
            return False, None, 0
        if a in (c, d) or b in (c, d):
            return False, None, 0
        
        # Temporarily remove (a,b)
        adj[a] &= ~(1 << b)
        adj[b] &= ~(1 << a)
        
        # Check if we can add (c,d)
        can_add = is_valid_addition(adj, c, d)
        
        if can_add:
            adj[c] |= (1 << d)
            adj[d] |= (1 << c)
            return True, adj, 0  # Edge count stays same
        else:
            # Restore (a,b)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            return False, None, 0

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
            
        # Local search loop with edge swaps
        no_improve = 0
        max_no_improve = 80
        swap_attempts = 0
        max_swap_attempts = 5
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            # Try to add a random missing edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            if not missing:
                break
            
            u, v = random.choice(missing)
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                no_improve = 0
                swap_attempts = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
            else:
                # Try edge swap first
                if swap_attempts < max_swap_attempts:
                    success, new_adj, delta = try_edge_swap(current_adj, deadline)
                    if success:
                        current_adj = new_adj
                        swap_attempts += 1
                        # After swap, try greedy fill
                        current_adj = greedy_fill(current_adj, deadline)
                        current_edges = count_edges(current_adj)
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = current_adj[:]
                            no_improve = 0
                        continue
                    else:
                        swap_attempts += 1
                
                # Kick: remove 1-3 random edges
                kick_size = random.choice([1, 1, 2, 2, 3])
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
                if not existing:
                    break
                
                random.shuffle(existing)
                for _ in range(min(kick_size, len(existing))):
                    a, b = existing.pop()
                    current_adj[a] &= ~(1 << b)
                    current_adj[b] &= ~(1 << a)
                    current_edges -= 1
                    
                # Greedy fill
                current_adj = greedy_fill(current_adj, deadline)
                current_edges = count_edges(current_adj)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts
    num_restarts = 1000
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
            
    # Final polish on best solution
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
