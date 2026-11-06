import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit

    n = 40
    
    # adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) to the current graph creates a C6.
        Returns True if it is safe to add (no C6 created), False otherwise.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        # We need to check if there is a path of length 3 between u and v
        # i.e., u-a-b-c-d-v
        # a in N(u), d in N(v), a != d
        # path a-b-c-d exists in G \ {u, v}
        
        # Iterate through all pairs (a, d)
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            d = nv_mask & ~lb_a  # d != a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                # We need a path a-b-c-d
                # b must be in N(a) \ {u, v, d, a}
                # c must be in N(d) \ {u, v, a, d}
                # and b ~ c
                
                # Mask for b candidates
                b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
                
                # Mask for c candidates
                c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                
                # Check if any neighbor of any b in b_candidates is in c_candidates
                # Let U = union of neighbors of all b in b_candidates
                # If U & c_candidates is non-empty, then a C6 exists.
                
                union_neighbors = 0
                bm = b_candidates
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    union_neighbors |= adj[bi]
                    
                    # Early exit if we already found a connection
                    if union_neighbors & c_candidates:
                        return False
                
            # end d loop
        # end a loop
        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Initialize base structure: 8 disjoint K5s
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = base_adj[:]
    best_count = count_edges(best_adj)
    
    # Function to greedily add edges to a given adjacency list
    def greedy_add(adj, candidate_pairs, deadline):
        # We need to shuffle the candidate pairs for diversity
        # But for local search, we might want to try specific ones
        # Here we just iterate through all non-edges and add if possible
        # To make it faster, we can maintain a list of non-edges
        # But for simplicity, we iterate all pairs
        
        # Create a list of non-edges in current adj
        non_edges = []
        for u, v in all_pairs:
            if not ((adj[u] >> v) & 1):
                non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        added = False
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not ((adj[u] >> v) & 1):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
        return added

    # Phase 1: Initial Greedy Saturation
    current_adj = base_adj[:]
    greedy_add(current_adj, all_pairs, deadline)
    current_count = count_edges(current_adj)
    
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]
    
    # Phase 2: Local Search (Remove-Add)
    # We try to remove an edge and then greedily add edges to see if we can get more total edges
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        # Pick a random edge to remove
        # Get list of current edges
        current_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (current_adj[u] >> v) & 1:
                    current_edges.append((u, v))
        
        if not current_edges:
            break
            
        u_rem, v_rem = random.choice(current_edges)
        
        # Remove the edge
        trial_adj = current_adj[:]
        trial_adj[u_rem] &= ~(1 << v_rem)
        trial_adj[v_rem] &= ~(1 << u_rem)
        
        # Greedily add edges
        greedy_add(trial_adj, all_pairs, deadline)
        
        trial_count = count_edges(trial_adj)
        
        if trial_count > current_count:
            # Accept the move
            current_adj = trial_adj
            current_count = trial_count
            
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
                
                # Early exit if we hit a very high score
                if best_count >= 105:
                    break
        else:
            # Revert (current_adj remains unchanged)
            # We could also try a different edge removal strategy or simulated annealing
            # but for now, simple hill climbing
            
            # If we are stuck for a while, we might want to randomize the greedy order more
            # or try removing multiple edges. 
            # For simplicity, we just continue.
            pass

    # Final check: one more greedy pass on the best found graph
    # This might find edges that were missed due to the order of operations in local search
    final_adj = best_adj[:]
    greedy_add(final_adj, all_pairs, deadline)
    final_count = count_edges(final_adj)
    
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj

    # Return the edge list of the best graph found
    return get_edge_list(best_adj)
