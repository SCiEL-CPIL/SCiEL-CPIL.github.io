import random
import time
import math

N = 20
# Total pairs
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
# Precompute triples for violation check? 
# There are C(20,3) = 1140 triples.
# For each triple (a,b,c), we need to check N(a) & N(b) & N(c).
# We can precompute the index of the triple?
# Or just iterate.

def count_violations(adj):
    """
    Returns the number of triples (a,b,c) with >= 3 common neighbors.
    """
    violations = 0
    # To speed up, we can iterate a, then b, then c?
    # Or just nested loops.
    # 20*19*18 / 6 = 1140 iterations.
    # Each does 3 bitwise ANDs and a bit_count.
    # This is very fast.
    for a in range(N):
        na = adj[a]
        for b in range(a + 1, N):
            nab = na & adj[b]
            # If popcount(nab) < 3, then any c will have < 3 common neighbors with a,b?
            # No. |N(a) & N(b) & N(c)| <= |N(a) & N(b)|.
            # So if popcount(nab) < 3, then for any c, the intersection will have < 3 bits.
            # So we can skip c loop if popcount(nab) < 3.
            if nab.bit_count() < 3:
                continue
            for c in range(b + 1, N):
                nabc = nab & adj[c]
                if nabc.bit_count() >= 3:
                    violations += 1
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_edges = 0
    best_adj = [0] * N
    
    # We will run multiple SA restarts
    num_restarts = 5 # Adjust based on time
    
    for r in range(num_restarts):
        # Initialize random graph
        # p = 0.5 is a good start for N=20
        # Max edges ~ 190. Ex(20, K3,3) is likely ~ 80-90.
        # p=0.5 gives ~95 edges.
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        cur_edges = sum(adj[i].bit_count() for i in range(N)) // 2
        cur_viol = count_violations(adj)
        
        # If initial is valid and better, update
        if cur_viol == 0 and cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = adj[:]
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        steps = 50000 # Estimate time per step
        
        # Lambda penalty
        # We want valid states to be much better than invalid.
        # Max edges 190. Max violations 1140.
        # If lambda = 10, a single violation costs 10 edges.
        # This encourages validity.
        LAMBDA = 10.0
        
        for s in range(steps):
            if time.time() > deadline:
                return [[i, j] for i in range(N) for j in range(i + 1, N) if (adj[i] >> j) & 1] # Wait, best_adj!
            
            # Pick a random edge to toggle
            # To bias towards adding edges (since we want max), 
            # we can pick a random non-edge with high prob, or random edge with low prob.
            # Or just random pair.
            # If we pick a pair that is an edge, we remove it.
            # If non-edge, we add it.
            
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            is_edge = bool((adj[u] >> v) & 1)
            
            # Calculate delta
            # Toggle edge (u,v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_edges = cur_edges + (1 if not is_edge else -1)
            new_viol = count_violations(adj)
            delta_viol = new_viol - cur_viol
            
            # Energy H = -Edges + LAMBDA * Violations
            # Delta H = -(Delta Edges) + LAMBDA * (Delta Violations)
            delta_E = - (1 if not is_edge else -1) + LAMBDA * delta_viol
            
            # Acceptance
            accept = False
            if delta_E < 0:
                accept = True
            else:
                if T > 0:
                    if random.random() < math.exp(-delta_E / T):
                        accept = True
            
            if accept:
                cur_edges = new_edges
                cur_viol = new_viol
                if cur_viol == 0 and cur_edges > best_edges:
                    best_edges = cur_edges
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        # After SA, we can try a local greedy improvement
        # Try adding any valid edge
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    if not ((adj[i] >> j) & 1):
                        # Check if adding (i,j) is valid
                        # We can just count violations of the new graph?
                        # Or check incremental?
                        # For speed in this small phase, count_violations is fine.
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        new_v = count_violations(adj)
                        if new_v == 0:
                            cur_edges += 1
                            improved = True
                        else:
                            adj[i] ^= (1 << j)
                            adj[j] ^= (1 << i)
            if cur_edges > best_edges and cur_viol == 0:
                best_edges = cur_edges
                best_adj = adj[:]

    # Construct final edge list from best_adj
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
