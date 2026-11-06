import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    
    # adj[u] is a bitmask of neighbors of u
    # bit i is set if edge (u, i) exists
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Condition: For any triple T, |N(T)| <= 2.
        
        When adding edge (u, v):
        - For any triple {u, a, b} where a, b are neighbors of v, 
          v becomes a common neighbor of u, a, b.
          We need current |N(u) ∩ N(a) ∩ N(b)| <= 1.
        - For any triple {v, a, b} where a, b are neighbors of u,
          u becomes a common neighbor of v, a, b.
          We need current |N(v) ∩ N(a) ∩ N(b)| <= 1.
        """
        if adj[u] & (1 << v):
            return False
            
        # Check triples containing u: {u, a, b}
        # v must be in N(a) ∩ N(b) for v to be added to N({u, a, b})
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
                # v is in N(a) ∩ N(b), so adding u-v adds v to N({u, a, b})
                # We must ensure current |N({u, a, b})| <= 1
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b}
        # u must be in N(a) ∩ N(b) for u to be added to N({v, a, b})
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
                # u is in N(a) ∩ N(b), so adding u-v adds u to N({v, a, b})
                # We must ensure current |N({v, a, b})| <= 1
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def local_search(adj, max_swap_iters=300):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        # Phase 1: Greedy add - try to add edges in random order
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) 
                    if not (current_adj[i] & (1 << j))]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    improved = True
        
        # Phase 2: Edge swaps - remove one edge, try to add two
        swap_iterations = 0
        while swap_iterations < max_swap_iters:
            if time.time() > deadline:
                break
            swap_iterations += 1
            
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                             if current_adj[i] & (1 << j)]
            if not existing_edges:
                break
            
            random.shuffle(existing_edges)
            
            found_swap = False
            for u, v in existing_edges:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                # Temporarily remove edge (u, v)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                # Try to add two edges from missing edges
                missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                                if not (current_adj[i] & (1 << j))]
                random.shuffle(missing_edges)
                
                added_count = 0
                for a, b in missing_edges:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 1:
                    # Net gain or neutral, keep the swap
                    current_edges += added_count - 1
                    found_swap = True
                else:
                    # Revert the swap
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
            if not found_swap:
                break
        
        # Phase 3: Final greedy add pass
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) 
                    if not (current_adj[i] & (1 << j))]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    improved = True
        
        return current_adj, current_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Generate initial candidates with multiple restarts
    num_restarts = 3000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i)
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish on the best solution
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            adj, edges = local_search(best_adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
            else:
                break

    # Convert bitmask to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
