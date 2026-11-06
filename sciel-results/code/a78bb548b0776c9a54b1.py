import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    # Safety margin: 145s limit to stay under 150s kill
    deadline = start_time + 145.0
    
    # Initialize adjacency as list of bitsets
    # adj[i] is integer with bit j set if edge (i,j) exists
    adj = [0] * n
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    total_possible_edges = len(all_edges)
    
    # Helper to add edge
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Helper to remove edge
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Helper to check if edge exists
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    # Helper to get neighbors as list
    def get_neighbors(u):
        bits = adj[u]
        neighbors = []
        while bits:
            lsb = bits & -bits
            v = lsb.bit_length() - 1
            neighbors.append(v)
            bits ^= lsb
        return neighbors

    # Check if adding edge (u, v) creates a C6
    # Path u-x-y-z-v of length 4 in G - (u,v)
    def would_create_c6(u, v):
        nu = adj[u]
        nv = adj[v]
        
        bits_x = nu
        while bits_x:
            lsb_x = bits_x & -bits_x
            x = lsb_x.bit_length() - 1
            bits_x ^= lsb_x
            
            if x == v:
                continue
                
            bits_z = nv
            while bits_z:
                lsb_z = bits_z & -bits_z
                z = lsb_z.bit_length() - 1
                bits_z ^= lsb_z
                
                if z == u or z == x:
                    continue
                
                common = adj[x] & adj[z]
                mask = ~((1 << u) | (1 << v) | (1 << x) | (1 << z))
                valid_y = common & mask
                
                if valid_y != 0:
                    return True
        return False

    # Count total C6s in the graph.
    # A C6 is a cycle of length 6.
    # We can count them by looking for paths of length 5 that close up?
    # Or more efficiently: For each pair of vertices (u,v) that are NOT adjacent,
    # count the number of paths of length 5 between them? No, that's expensive.
    
    # Alternative: Count C6s by iterating over all 6-tuples? No, O(n^6).
    # Better: Use the property that a C6 contains 6 edges.
    # Let's count "C6-candidates" by checking if removing an edge breaks a C6?
    
    # Actually, for Simulated Annealing, we need a fast energy function.
    # Energy = -alpha * (number of edges) + beta * (number of C6s).
    # We want to minimize Energy.
    # If we only allow moves that don't create C6s, we are back to local search.
    # To allow invalid states, we need to count C6s.
    
    # Efficient C6 count:
    # Number of C6s = 1/6 * sum_{u,v} (number of paths of length 5 between u and v that are simple and don't use edge (u,v) if it exists? No.)
    
    # Let's use a different metric: "C6-ness".
    # For each pair of non-adjacent vertices (u,v), let P5(u,v) be the number of simple paths of length 5 between u and v.
    # If we add edge (u,v), we create P5(u,v) new C6s.
    # So, the number of C6s in the graph is NOT simply sum of P5(u,v) for non-edges, because C6s can exist without a "chord" or "missing edge" in that specific way?
    # Wait. A C6 is a cycle. If we have a C6, it has 6 edges.
    # If we consider the graph G, the number of C6s is exactly the number of ways to pick 6 vertices that form a cycle.
    
    # Is there a faster way to count C6s in a 40-vertex graph?
    # n=40 is small.
    # Number of 6-subsets is C(40,6) ~ 3.8 million.
    # Checking if a subset forms a C6 involves checking 6! = 720 permutations? Or just checking if the induced subgraph is a 6-cycle?
    # Induced subgraph on 6 vertices is a C6 if and only if it has 6 edges and is connected and each vertex has degree 2 within the subgraph.
    # Checking degree 2 for 6 vertices is fast.
    # 3.8M subsets * (check 6 degrees) ~ 3.8M * 36 ops ~ 136M ops.
    # In Python, this is too slow for every step (136M ops is ~1-2 seconds).
    # We need a faster energy function or a different approach.
    
    # Alternative: Keep the graph valid (C6-free) but use a smarter search.
    # The current program uses Greedy + Perturbation.
    # Let's try **Simulated Annealing on Valid States only** but with a better move set.
    # Moves:
    # 1. Add an edge (if valid).
    # 2. Remove an edge.
    # 3. Swap: Remove e1, Add e2 (if valid).
    # 4. Double Swap: Remove e1, e2, Add e3, e4 (if valid).
    
    # The problem with greedy is it gets stuck.
    # Simulated Annealing can accept "bad" moves (removing edges) to escape.
    # But since we must maintain C6-freeness, "bad" means "fewer edges".
    # We accept fewer edges with probability exp(-delta_E / T).
    # Here delta_E is negative (we lose edges), so we accept with prob 1? No.
    # Objective: Maximize Edges.
    # If we move from m edges to m-k edges, delta = -k.
    # Accept with prob exp(k/T). This is always > 1? No, exp(-delta/T) where delta = new - old.
    # If new < old, delta is negative. exp(-(-k)/T) = exp(k/T) > 1.
    # Standard SA: accept if delta < 0 (improvement) or with prob exp(-delta/T).
    # Here, "energy" E = -m. Minimize E.
    # Move m -> m-k. Delta E = - (m-k) - (-m) = k > 0.
    # Accept with prob exp(-k/T).
    
    # This allows us to remove edges to escape local maxima.
    
    # Let's implement SA on valid states.
    
    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    # Initialize with a dense random graph that is likely C6-free?
    # Random graph G(n, p) with p=0.2 has ~320 edges. Likely has many C6s.
    # We need a valid start.
    # Start with 8 disjoint K5s (80 edges).
    
    def init_graph():
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_adj = None
    best_count = 0
    
    random.seed(42)
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # We will run multiple restarts or a long SA chain.
    # Given time limit, a single long SA chain with occasional "restarts" (if stuck) is good.
    
    adj = init_graph()
    current_count = count_edges(adj)
    best_count = current_count
    best_adj = adj[:]
    
    T = T_start
    
    # Pre-compute list of all edges for random selection
    # We need to pick random edges to remove, and random non-edges to add.
    
    while time.time() < deadline - 2.0:
        # 1. Try to add an edge
        # Pick a random non-edge
        # To pick a random non-edge efficiently, we can try random pairs until we find one that is not an edge.
        # Or maintain a list of non-edges. Maintaining is expensive.
        # Since n=40, total pairs 780. Current edges ~100. Non-edges ~680.
        # Random pair has ~87% chance of being non-edge.
        
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if has_edge(u, v):
            # It's an edge. Try to remove it?
            # Or just skip to next iteration to try adding?
            # In SA, we should consider both add and remove moves.
            # Let's decide move type:
            # 50% chance add, 50% chance remove.
            
            if random.random() < 0.5:
                # Try Remove
                remove_edge(u, v)
                new_count = current_count - 1
                delta_E = new_count - current_count # -1
                # Accept if delta_E < 0 (improvement in -E? No, E=-m. Delta E = -new_m - (-old_m) = old_m - new_m = 1)
                # Wait. Objective: Maximize m.
                # Let's stick to standard: Minimize Cost = -m.
                # Old Cost = -m. New Cost = -(m-1) = -m+1.
                # Delta Cost = 1.
                # Accept if random < exp(-1/T).
                if random.random() < (2.718281828 ** (-1.0 / T)):
                    current_count = new_count
                else:
                    add_edge(u, v) # Reject, restore
            else:
                # Try Add (but it's already an edge, so skip)
                pass
        else:
            # It's a non-edge. Try to Add.
            if not would_create_c6(u, v):
                add_edge(u, v)
                current_count += 1
                # Always accept improvement
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Cannot add.
                # Maybe try to remove a different edge to enable this add?
                # That's a 2-move.
                # For now, just skip.
                pass
        
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Optional: If stuck for a long time, restart?
        # Hard to detect "stuck" easily without extra state.
        # The SA temperature decay should handle exploration.
        
    # Final check: Ensure best_adj is valid
    # It should be, as we only accept valid moves.
    
    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        bits = best_adj[u]
        while bits:
            lsb = bits & -bits
            v = lsb.bit_length() - 1
            if v > u:
                result_edges.append([u, v])
            bits ^= lsb
            
    return result_edges

# EVOLVE-BLOCK-END
