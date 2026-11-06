import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) to the graph defined by adj 
        violates the K3,3-free condition.
        Assumes (u, v) is NOT currently an edge.
        """
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
                
                # v is in N(a) ∩ N(b)
                # So adding u-v adds v to N({u, a, b})
                # We must ensure current |N({u, a, b})| <= 1
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
                
                # u is in N(a) ∩ N(b)
                # So adding u-v adds u to N({v, a, b})
                # We must ensure current |N({v, a, b})| <= 1
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_all_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def get_all_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def simulated_annealing(initial_adj, max_steps=50000):
        current_adj = initial_adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.01
        
        edges_list = get_all_edges(current_adj)
        non_edges_list = get_all_non_edges(current_adj)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if T < min_T:
                break
                
            move_type = random.randint(0, 2)
            
            if move_type == 0: # Add edge
                if not non_edges_list:
                    continue
                u, v = random.choice(non_edges_list)
                if is_valid_addition(current_adj, u, v):
                    # Accept
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    # Update lists
                    non_edges_list.remove((u, v))
                    if (v, u) in non_edges_list: non_edges_list.remove((v, u)) # Not needed if we store sorted
                    edges_list.append((u, v))
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                # If invalid, do nothing (reject)
                
            elif move_type == 1: # Remove edge
                if not edges_list:
                    continue
                u, v = random.choice(edges_list)
                # Always valid to remove
                # Accept with probability 1 if improving? No, removing reduces edges.
                # We accept if T is high enough.
                # Delta E = -1 (energy increase in minimization sense, or cost increase)
                # We want to maximize edges, so losing an edge is bad.
                # P(accept) = exp(1/T) is always > 1? No.
                # Let's define Cost = -Edges.
                # Delta Cost = 1.
                # P(accept) = exp(-Delta Cost / T) = exp(-1/T).
                if random.random() < (1.0 / T): # Simple approximation, or just exp(-1/T)
                    # Actually, standard SA: P = exp(-Delta E / T). Here Delta E = 1 (worse).
                    # So P = exp(-1/T).
                    pass 
                # Let's use a simpler heuristic: accept with prob 1/T or fixed prob?
                # Let's use exp(-1/T)
                if random.random() < __import__('math').exp(-1.0 / T):
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges -= 1
                    edges_list.remove((u, v))
                    non_edges_list.append((u, v))
                    
            else: # Swap (Remove 1, Add 1)
                if not edges_list or not non_edges_list:
                    continue
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges_list)
                
                # Temporarily remove (u,v)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                # Check if (x,y) can be added
                if is_valid_addition(current_adj, x, y):
                    # Accept
                    current_adj[x] |= (1 << y)
                    current_adj[y] |= (1 << x)
                    # Update lists
                    edges_list.remove((u, v))
                    non_edges_list.append((u, v))
                    non_edges_list.remove((x, y))
                    edges_list.append((x, y))
                    # Edge count unchanged
                else:
                    # Revert
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    # Reject swap
                
            T *= cooling_rate
            
        return best_adj, best_edges

    # Initial construction
    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts with SA
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i)
        adj, edges = simulated_annealing(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish on the best solution with more SA steps
    if time.time() < deadline - 5:
        # Run SA again on the best found, maybe with higher max steps
        adj, edges = simulated_annealing(best_adj, max_steps=100000)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj

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
