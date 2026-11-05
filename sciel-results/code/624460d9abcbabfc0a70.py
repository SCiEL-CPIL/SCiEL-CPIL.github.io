import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_pairs)
    
    # Precompute triple pairs for fast violation checking
    # A violation exists if there is a triple (i,j,k) with |N(i) & N(j) & N(k)| >= 3
    # We can check this by iterating over pairs (i,j) and seeing if their common neighbors
    # have any triple among themselves? No, that's not right.
    # The condition is: for any distinct i,j,k, |N(i) & N(j) & N(k)| < 3.
    # This is equivalent to: for any pair (i,j), the set C = N(i) & N(j) must not contain
    # any pair (k,l) such that k,l are also connected to each other? No.
    # If C contains k and l, and k,l are connected to i and j, then i,j,k,l form a K4?
    # No, K3,3 means two sets of 3 vertices with all 9 edges between them.
    # The condition "no three vertices have three common neighbours" is exactly the K3,3-free condition?
    # Let's verify. If i,j,k have common neighbours a,b,c, then {i,j,k} and {a,b,c} form a K3,3.
    # So yes, the condition is: for all distinct i,j,k, |N(i) & N(j) & N(k)| <= 2.
    
    # How to check this fast?
    # For each pair (i,j), let C = N(i) & N(j).
    # If |C| >= 3, we need to check if any k in C has 3 neighbors in C?
    # No, k is in C, so k is a neighbor of i and j.
    # We need |N(i) & N(j) & N(k)| < 3.
    # N(i) & N(j) & N(k) is the set of vertices connected to i, j, and k.
    # Since k is connected to i and j, k is NOT in N(i) & N(j) unless we consider self-loops (we don't).
    # Wait, N(i) is the set of neighbors of i.
    # So N(i) & N(j) & N(k) is the set of vertices v such that v~i, v~j, v~k.
    # If we fix i,j, and look at C = N(i) & N(j).
    # For any k in C, the number of common neighbors of i,j,k is |C & N(k)|.
    # So we need |C & N(k)| <= 2 for all k in C.
    # This means: for any pair (i,j), and any k in N(i) & N(j), k must have at most 2 neighbors in N(i) & N(j).
    # Equivalently: the subgraph induced by C = N(i) & N(j) has maximum degree <= 2?
    # No, N(k) & C is the neighbors of k within C.
    # So yes, for every pair (i,j), the subgraph induced by C_ij = N(i) & N(j) must have max degree <= 2.
    
    # Even faster check:
    # Iterate over all pairs (i,j).
    # C = adj[i] & adj[j]
    # If C has < 3 bits, skip.
    # Else, for each k in C, check (adj[k] & C).bit_count() <= 2.
    
    # Let's precompute the list of pairs (i,j) to iterate.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_violations(adj):
        """Count the number of violating triples (i,j,k)."""
        # Actually, we just need to know if it's 0 or >0 for validity,
        # but for annealing, a penalty proportional to the number of violations
        # or just a binary penalty might work. Let's use a binary penalty first,
        # or better, the number of violating pairs (i,j) where max degree in C_ij > 2.
        # Or even simpler: just return 1 if invalid, 0 if valid?
        # A smoother penalty is better for annealing.
        # Let's count the number of pairs (i,j) such that there exists k in C_ij with deg_C(k) >= 3.
        # Or just count the number of k in C_ij with deg_C(k) >= 3.
        
        viol_count = 0
        for i, j in pairs:
            cij = adj[i] & adj[j]
            if cij.bit_count() < 3:
                continue
            # Check each k in cij
            c = cij
            while c:
                low = c & (-c)
                k = low.bit_length() - 1
                if (adj[k] & cij).bit_count() >= 3:
                    viol_count += 1
                    # We can break early if we just need validity,
                    # but for penalty, counting is better.
                    # However, counting all might be slow.
                    # Let's just count 1 per violating (i,j) pair to save time?
                    # Or break after first violation per pair?
                    # Let's count the number of violating k's.
                c ^= low
        return viol_count

    def is_valid(adj):
        # Fast check for zero violations
        for i, j in pairs:
            cij = adj[i] & adj[j]
            if cij.bit_count() < 3:
                continue
            c = cij
            while c:
                low = c & (-c)
                k = low.bit_length() - 1
                if (adj[k] & cij).bit_count() >= 3:
                    return False
                c ^= low
        return True

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    # Initialize with a dense random graph
    def random_adj(p):
        adj = [0] * n
        for i, j in pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj

    # Simulated Annealing
    best_adj = None
    best_score = -1
    best_edges = []
    
    # Run multiple restarts with different initial densities
    restarts = [0.8, 0.7, 0.6, 0.5, 0.9]
    random.shuffle(restarts)
    
    for p_start in restarts:
        if time.time() > deadline - 10:
            break
            
        adj = random_adj(p_start)
        
        # Initial state
        curr_adj = adj[:]
        curr_viol = count_violations(curr_adj)
        curr_edges = count_edges(curr_adj)
        curr_score = curr_edges - 10 * curr_viol  # Penalty weight
        
        best_adj_local = curr_adj[:]
        best_score_local = curr_score
        best_edges_local = adj_to_edges(curr_adj) if curr_viol == 0 else []
        
        # Annealing parameters
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        steps = 0
        
        while time.time() < deadline - 5:
            steps += 1
            if T < T_min:
                break
            
            # Pick a random edge to flip
            idx = random.randint(0, num_edges - 1)
            u, v = all_pairs[idx]
            
            # Flip the edge
            old_u, old_v = curr_adj[u], curr_adj[v]
            if curr_adj[u] & (1 << v):
                # Remove edge
                curr_adj[u] &= ~(1 << v)
                curr_adj[v] &= ~(1 << u)
            else:
                # Add edge
                curr_adj[u] |= (1 << v)
                curr_adj[v] |= (1 << u)
            
            # Calculate new violations
            # We can optimize this by only checking affected pairs,
            # but for simplicity and given n=20 is small, full recount is okay?
            # 190 pairs, each check is fast.
            # Let's try full recount first.
            new_viol = count_violations(curr_adj)
            new_edges = count_edges(curr_adj)
            new_score = new_edges - 10 * new_viol
            
            delta = new_score - curr_score
            
            if delta > 0:
                # Accept
                curr_adj = curr_adj[:] # Actually we already modified it in place
                curr_viol = new_viol
                curr_edges = new_edges
                curr_score = new_score
                
                if curr_viol == 0 and curr_edges > (len(best_edges_local)):
                    best_adj_local = curr_adj[:]
                    best_score_local = curr_score
                    best_edges_local = adj_to_edges(curr_adj)
            else:
                # Accept with probability exp(delta / T)
                if random.random() < np.exp(delta / T):
                    curr_viol = new_viol
                    curr_edges = new_edges
                    curr_score = new_score
                    
                    if curr_viol == 0 and curr_edges > (len(best_edges_local)):
                        best_adj_local = curr_adj[:]
                        best_score_local = curr_score
                        best_edges_local = adj_to_edges(curr_adj)
                else:
                    # Reject: revert
                    curr_adj[u] = old_u
                    curr_adj[v] = old_v
            
            T *= cooling
            
            # Update global best
            if best_edges_local and len(best_edges_local) > len(best_edges):
                best_edges = best_edges_local[:]
                best_adj = best_adj_local[:]
                best_score = best_score_local

    # If no valid graph found, return a trivial one
    if not best_edges:
        # Construct a valid graph manually: a path
        return [[i, i + 1] for i in range(n - 1)]
        
    return best_edges

