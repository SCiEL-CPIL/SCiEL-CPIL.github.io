"""Combined approach: C9 union baseline + bitmask K4-free checking + local search."""
import time
import random
import math
import sys

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 18
    full_mask = (1 << N) - 1
    
    # Helper: Check if a bitmask represents an independent set in graph defined by adj (list of bitmasks)
    # Optimized: iterate through bits and check adjacency
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True
    
    # Helper: Check if adding edge (u,v) keeps graph K4-free
    # A K4 is formed if u,v have common neighbors w,x such that w-x is an edge.
    # So common neighbors of u and v must form an independent set.
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True  # Already an edge
        common = adj[u] & adj[v]
        return is_independent(common, adj)
    
    # Exact B&B for largest triangle-free induced subgraph
    # A triangle-free induced subgraph S means: no three vertices in S form a triangle
    def largest_triangle_free(adj, time_limit=None):
        best = [0]
        
        # Greedy heuristic to get a good initial lower bound
        # Try multiple random orders
        for _ in range(10):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                # Check if v can be added: N(v) ∩ S must be independent
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = S.bit_count()
            if sz > best[0]:
                best[0] = sz
        
        if best[0] == N:
            return N
        
        # Precompute: for each vertex, its neighbors as a bitmask
        # For B&B, we need to check if adding v to S creates a triangle
        # v creates a triangle with S iff there exist u,w in S such that u-v and v-w and u-w are edges
        # i.e., N(v) ∩ S contains an edge
        
        # Branch and bound with better pruning
        def bnb(S_mask, C_mask, sz):
            if time_limit and time.time() > time_limit:
                return
            if sz + C_mask.bit_count() <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Prune: remove candidates that cannot be added to current S
            new_C = 0
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if is_independent(adj[v] & S_mask, adj):
                    new_C |= lsb
                c ^= lsb
            
            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return
            
            C_mask = new_C
            if sz + C_mask.bit_count() <= best[0]:
                return
            
            # Branch: pick a vertex with high degree in C_mask for better pruning
            # Choose vertex that appears in most edges within C_mask
            best_v = -1
            best_score_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                # Count how many candidates are adjacent to v
                deg_in_C = (adj[v] & C_mask).bit_count()
                if deg_in_C > best_score_v:
                    best_score_v = deg_in_C
                    best_v = v
                c ^= lsb
            
            if best_v < 0:
                lsb = C_mask & -C_mask
                best_v = lsb.bit_length() - 1
            
            v_bit = 1 << best_v
            C_mask ^= v_bit
            
            # Branch 1: Include best_v
            bnb(S_mask | v_bit, C_mask, sz + 1)
            
            # Branch 2: Exclude best_v
            bnb(S_mask, C_mask, sz)
        
        bnb(0, full_mask, 0)
        return best[0]
    
    # Get edge list from adjacency bitmasks
    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Simulated annealing for better exploration
    def simulated_annealing(adj, best_score, deadline):
        current_score = largest_triangle_free(adj, deadline - 5.0)
        if current_score < best_score:
            best_score = current_score
        
        T = 1.0
        cooling = 0.995
        adj_sa = adj[:]
        
        while time.time() < deadline and T > 0.01:
            # Pick a random edge to flip
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            old_val = (adj_sa[i] >> j) & 1
            
            if old_val == 0:
                if not can_add_edge(adj_sa, i, j):
                    continue
                adj_sa[i] |= (1 << j)
                adj_sa[j] |= (1 << i)
            else:
                adj_sa[i] &= ~(1 << j)
                adj_sa[j] &= ~(1 << i)
            
            new_score = largest_triangle_free(adj_sa, deadline - 5.0)
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    # Copy the improved solution
                    for k in range(N):
                        adj[k] = adj_sa[k]
            else:
                # Revert
                if old_val == 0:
                    adj_sa[i] &= ~(1 << j)
                    adj_sa[j] &= ~(1 << i)
                else:
                    adj_sa[i] |= (1 << j)
                    adj_sa[j] |= (1 << i)
            
            T *= cooling
        
        return best_score
    
    # Create C9 ∪ C9 baseline
    # Two disjoint 9-cycles: vertices 0-8 form one cycle, 9-17 form another
    def make_c9_union_adj():
        adj = [0] * N
        # First cycle: 0-1-2-3-4-5-6-7-8-0
        for i in range(9):
            a = i
            b = (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        # Second cycle: 9-10-11-12-13-14-15-16-17-9
        for i in range(9):
            a = 9 + i
            b = 9 + (i + 1) % 9
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        return adj
    
    # Verify K4-free using bitmask operations
    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if not ((adj[i] >> j) & 1):
                    continue
                # Common neighbors of i and j
                common = adj[i] & adj[j]
                # Check if common neighbors contain an edge (which would form K4 with i,j)
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    if adj[k] & (common ^ lsb):
                        return False
                    c ^= lsb
        return True
    
    random.seed(42)
    
    # Start with C9 ∪ C9
    current_adj = make_c9_union_adj()
    current_score = largest_triangle_free(current_adj, deadline - 5.0)
    best_adj = current_adj[:]
    best_score = current_score
    
    # Local search: try edge flips (add or remove), plus 2-edge swaps
    def local_search(adj, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible single edge flips
            for i in range(N):
                if time.time() > deadline:
                    return best_score
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        return best_score
                    old_val = (adj[i] >> j) & 1
                    
                    if old_val == 0:
                        # Try adding edge
                        if not can_add_edge(adj, i, j):
                            continue
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        # Try removing edge
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    
                    score = largest_triangle_free(adj, deadline - 5.0)
                    
                    if score < best_score:
                        best_score = score
                        improved = True
                    else:
                        # Revert
                        if old_val == 0:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Try 2-edge swaps: remove one edge, add another
            if time.time() < deadline - 2.0:
                for i in range(N):
                    if time.time() > deadline:
                        return best_score
                    for j in range(i + 1, N):
                        if time.time() > deadline:
                            return best_score
                        if not ((adj[i] >> j) & 1):
                            continue  # Only consider removing existing edges
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        
                        # Try adding each non-edge
                        for a in range(N):
                            if time.time() > deadline:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                return best_score
                            for b in range(a + 1, N):
                                if time.time() > deadline:
                                    adj[i] |= (1 << j)
                                    adj[j] |= (1 << i)
                                    return best_score
                                if (adj[a] >> b) & 1:
                                    continue  # Already an edge
                                if not can_add_edge(adj, a, b):
                                    continue
                                adj[a] |= (1 << b)
                                adj[b] |= (1 << a)
                                
                                score = largest_triangle_free(adj, deadline - 5.0)
                                
                                # Revert the add
                                adj[a] &= ~(1 << b)
                                adj[b] &= ~(1 << a)
                                
                                if score < best_score:
                                    best_score = score
                                    improved = True
                                    # Keep the swap: re-add the new edge
                                    adj[a] |= (1 << b)
                                    adj[b] |= (1 << a)
                                    # Revert the removal
                                    adj[i] &= ~(1 << j)
                                    adj[j] &= ~(1 << i)
                                    break
                                # Revert the removal for next try
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
        
        return best_score
    
    # Run local search from C9 ∪ C9
    best_score = local_search(current_adj, best_score, deadline)
    if best_score < current_score:
        best_adj = current_adj[:]
    
    # Run simulated annealing from current best
    if time.time() < deadline - 20.0:
        best_score = simulated_annealing(best_adj, best_score, deadline - 10.0)
    
    # Random restarts with perturbation
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Generate a random K4-free graph by greedy edge addition
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Verify K4-free (should be, but just in case)
        if not is_k4_free(adj):
            continue
        
        score = largest_triangle_free(adj, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            # Local search from this point
            best_score = local_search(adj, best_score, deadline)
            if best_score < score:
                best_adj = adj[:]
        
        # Also try perturbing the C9 union
        if attempts % 3 == 0:
            adj2 = make_c9_union_adj()
            # Randomly add some edges while maintaining K4-free
            num_adds = random.randint(3, 15)
            for _ in range(num_adds):
                u, v = random.sample(range(N), 2)
                if can_add_edge(adj2, u, v):
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
            
            if is_k4_free(adj2):
                score2 = largest_triangle_free(adj2, deadline - 5.0)
                if score2 < best_score:
                    best_score = score2
                    best_adj = adj2[:]
                    best_score = local_search(adj2, best_score, deadline)
                    if best_score < score2:
                        best_adj = adj2[:]
        
        # Also try perturbing by removing edges from current best
        if attempts % 5 == 0 and time.time() < deadline - 15.0:
            adj3 = best_adj[:]
            num_removals = random.randint(1, 5)
            current_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (adj3[i] >> j) & 1]
            if current_edges:
                for _ in range(min(num_removals, len(current_edges))):
                    u, v = random.choice(current_edges)
                    adj3[u] &= ~(1 << v)
                    adj3[v] &= ~(1 << u)
                    current_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if (adj3[i] >> j) & 1]
                    if not current_edges:
                        break
                
                score3 = largest_triangle_free(adj3, deadline - 5.0)
                if score3 < best_score:
                    best_score = score3
                    best_adj = adj3[:]
        
        # Try SA from random restarts occasionally
        if attempts % 7 == 0 and time.time() < deadline - 20.0:
            adj4 = adj[:]  # Copy the random graph
            best_score = simulated_annealing(adj4, best_score, deadline - 10.0)
    
    return get_edge_list(best_adj)

# EVOLVE-BLOCK-END
