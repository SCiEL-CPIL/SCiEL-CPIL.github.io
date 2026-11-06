import time
import random
import math

N = 20
TOTAL_EDGES = N * (N - 1) // 2  # 190
DEADLINE_MARGIN = 5.0

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Precomputation
    # Map edge index to (u, v)
    edge_to_uv = []
    uv_to_edge = {}
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            edge_to_uv.append((u, v))
            uv_to_edge[(u, v)] = idx
            uv_to_edge[(v, u)] = idx
            idx += 1
            
    # Map vertex to list of incident edge indices
    vertex_to_edges = [[] for _ in range(N)]
    for e in range(TOTAL_EDGES):
        u, v = edge_to_uv[e]
        vertex_to_edges[u].append(e)
        vertex_to_edges[v].append(e)
        
    # Helper to get neighbors of u given a mask
    def get_neighbors(mask, u):
        neighbors = []
        for e in vertex_to_edges[u]:
            if (mask >> e) & 1:
                v, w = edge_to_uv[e]
                nb = v if w == u else w
                neighbors.append(nb)
        return neighbors
        
    # Helper to initialize common counts from a mask
    def init_common_counts(mask):
        counts = [[0] * N for _ in range(N)]
        # For each vertex w, it contributes to common counts of pairs (u, v) 
        # where u, v are neighbors of w.
        # This is O(N^3) roughly, which is fine for N=20.
        # Or simpler: for each pair (u,v), count common neighbors.
        # Let's do the direct way for initialization.
        adj = [set() for _ in range(N)]
        for e in range(TOTAL_EDGES):
            if (mask >> e) & 1:
                u, v = edge_to_uv[e]
                adj[u].add(v)
                adj[v].add(u)
                
        for u in range(N):
            for v in range(u + 1, N):
                counts[u][v] = len(adj[u] & adj[v])
                counts[v][u] = counts[u][v]
        return counts, adj
        
    # Helper to calculate penalty from counts
    def calc_penalty(counts):
        pen = 0
        for u in range(N):
            for v in range(u + 1, N):
                c = counts[u][v]
                if c > 2:
                    pen += c - 2
        return pen
        
    # Helper to count edges in mask
    def count_edges(mask):
        return bin(mask).count('1')
        
    # Greedy initialization
    def greedy_init():
        mask = 0
        adj = [set() for _ in range(N)]
        # Shuffle edges
        edges = list(range(TOTAL_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            u, v = edge_to_uv[e]
            # Check if adding (u,v) is valid
            # 1. |N(u) ∩ N(v)| <= 2
            if len(adj[u] & adj[v]) > 2:
                continue
            # 2. For all w in N(v), |N(u) ∩ N(w)| <= 1 (if v in N(w) is true, which it is if w in N(v))
            # Wait, the condition is: if we add (u,v), 
            # for any w in N(v), the common neighbors of u and w increase by 1 if v in N(w) (always true for w in N(v) except u).
            # So we need current |N(u) ∩ N(w)| <= 1 for all w in N(v) \ {u}.
            valid = True
            for w in adj[v]:
                if w == u:
                    continue
                if len(adj[u] & adj[w]) > 1:
                    valid = False
                    break
            if not valid:
                continue
            # 3. For all w in N(u), |N(v) ∩ N(w)| <= 1
            for w in adj[u]:
                if w == v:
                    continue
                if len(adj[v] & adj[w]) > 1:
                    valid = False
                    break
            if not valid:
                continue
            
            mask |= (1 << e)
            adj[u].add(v)
            adj[v].add(u)
            
        return mask
        
    # Simulated Annealing
    def simulated_annealing(initial_mask, time_limit):
        mask = initial_mask
        counts, adj = init_common_counts(mask)
        current_penalty = calc_penalty(counts)
        current_edges = count_edges(mask)
        
        best_mask = mask if current_penalty == 0 else 0
        best_edges = current_edges if current_penalty == 0 else 0
        
        # If initial is invalid, try to fix it or just start from scratch?
        # The greedy init should be valid.
        
        T_start = 10.0
        T_end = 0.01
        # Estimate number of steps
        # We can run until time limit.
        
        rng = random.Random(42)
        
        while time.time() < time_limit:
            # Decay temperature
            # Simple linear or exponential decay based on time?
            # Let's use a fixed number of steps per "epoch" and decay T.
            # Or just use a constant T and rely on the random walk?
            # Standard SA requires decay.
            # Let's do 10000 steps per epoch, decay T by 0.95.
            
            for _ in range(10000):
                if time.time() > time_limit:
                    break
                    
                e = rng.randint(0, TOTAL_EDGES - 1)
                u, v = edge_to_uv[e]
                is_present = (mask >> e) & 1
                
                # Identify affected pairs
                # If adding:
                #   Pairs (u, w) for w in N(v) \ {u}
                #   Pairs (v, w) for w in N(u) \ {v}
                # If removing:
                #   Same pairs, but effect is reversed.
                
                affected_pairs = []
                if is_present:
                    # Removing (u,v)
                    # w in N(v) \ {u}
                    for w in adj[v]:
                        if w != u:
                            affected_pairs.append((u, w))
                    for w in adj[u]:
                        if w != v:
                            affected_pairs.append((v, w))
                else:
                    # Adding (u,v)
                    for w in adj[v]:
                        if w != u:
                            affected_pairs.append((u, w))
                    for w in adj[u]:
                        if w != v:
                            affected_pairs.append((v, w))
                
                delta_penalty = 0
                for a, b in affected_pairs:
                    if a > b:
                        a, b = b, a
                    old_c = counts[a][b]
                    if is_present:
                        # Removing edge (u,v)
                        # Does this reduce common neighbors of a and b?
                        # a, b are (u, w) or (v, w).
                        # If pair is (u, w), common neighbors change if v is a common neighbor of u and w?
                        # v is in N(u). Is v in N(w)?
                        # w is in N(v). So (v,w) is an edge.
                        # So v is a neighbor of w.
                        # So v IS a common neighbor of u and w.
                        # Removing (u,v) means u and v are no longer connected? No, (u,v) is the edge being removed.
                        # Wait.
                        # If we remove (u,v):
                        # N(u) loses v. N(v) loses u.
                        # Pair (u, w):
                        # Old common neighbors: N(u) ∩ N(w).
                        # New common neighbors: (N(u) \ {v}) ∩ N(w).
                        # Difference: v was in N(u) and v is in N(w) (since w in N(v) implies (v,w) edge).
                        # So v was a common neighbor. Now it is not.
                        # So count decreases by 1.
                        # Delta c = -1.
                        # Penalty change:
                        # if old_c >= 3: delta_pen = -1
                        # else: delta_pen = 0
                        if old_c >= 3:
                            delta_penalty -= 1
                    else:
                        # Adding (u,v)
                        # v becomes a new neighbor of u.
                        # v is in N(w) (since w in N(v)).
                        # So v becomes a new common neighbor of u and w.
                        # Count increases by 1.
                        # Delta c = +1.
                        # Penalty change:
                        # if old_c >= 2: delta_pen = +1
                        # else: delta_pen = 0
                        if old_c >= 2:
                            delta_penalty += 1
                
                delta_edges = -1 if is_present else 1
                # Energy = -edges + lambda * penalty
                # We want to minimize Energy.
                # Let lambda = 10.
                lambda_pen = 10.0
                delta_energy = delta_edges * (-1) + delta_penalty * lambda_pen
                # Wait, if adding edge, delta_edges = +1, so delta_energy = -1 + ...
                # If removing edge, delta_edges = -1, so delta_energy = +1 + ...
                
                # Actually, let's define Energy = -|E| + lambda * P
                # Delta E = -Delta|E| + lambda * Delta P
                # If adding: Delta|E| = 1 -> -1.
                # If removing: Delta|E| = -1 -> +1.
                
                if is_present:
                    delta_E = 1 + delta_penalty * lambda_pen
                else:
                    delta_E = -1 + delta_penalty * lambda_pen
                
                # Acceptance probability
                if delta_E < 0:
                    accept = True
                else:
                    # T is current temperature
                    # We need to manage T.
                    # Let's use a global T variable that decays.
                    # This inner loop structure makes T management tricky.
                    # Let's restructure to have T decay per outer loop.
                    pass
            
            # This structure is getting messy. Let's rewrite SA loop more cleanly.
            break 
            
        # Proper SA Implementation
        return best_mask, best_edges

    # Clean SA Implementation
    def run_sa(initial_mask, time_limit):
        mask = initial_mask
        counts, adj = init_common_counts(mask)
        current_penalty = calc_penalty(counts)
        current_edges = count_edges(mask)
        
        best_mask = mask if current_penalty == 0 else 0
        best_edges = current_edges if current_penalty == 0 else 0
        
        T = 10.0
        cooling_rate = 0.999
        rng = random.Random(42)
        lambda_pen = 10.0
        
        while time.time() < time_limit and T > 0.01:
            # Inner loop: multiple moves per temperature step
            for _ in range(100):
                if time.time() > time_limit:
                    break
                
                e = rng.randint(0, TOTAL_EDGES - 1)
                u, v = edge_to_uv[e]
                is_present = (mask >> e) & 1
                
                affected_pairs = []
                if is_present:
                    for w in adj[v]:
                        if w != u:
                            affected_pairs.append((u, w))
                    for w in adj[u]:
                        if w != v:
                            affected_pairs.append((v, w))
                else:
                    for w in adj[v]:
                        if w != u:
                            affected_pairs.append((u, w))
                    for w in adj[u]:
                        if w != v:
                            affected_pairs.append((v, w))
                
                delta_penalty = 0
                for a, b in affected_pairs:
                    if a > b:
                        a, b = b, a
                    old_c = counts[a][b]
                    if is_present:
                        if old_c >= 3:
                            delta_penalty -= 1
                    else:
                        if old_c >= 2:
                            delta_penalty += 1
                
                if is_present:
                    delta_E = 1 + delta_penalty * lambda_pen
                else:
                    delta_E = -1 + delta_penalty * lambda_pen
                
                if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                    # Accept move
                    if is_present:
                        mask &= ~(1 << e)
                        adj[u].remove(v)
                        adj[v].remove(u)
                        current_edges -= 1
                    else:
                        mask |= (1 << e)
                        adj[u].add(v)
                        adj[v].add(u)
                        current_edges += 1
                    
                    # Update counts
                    for a, b in affected_pairs:
                        if a > b:
                            a, b = b, a
                        if is_present:
                            counts[a][b] -= 1
                            counts[b][a] -= 1
                        else:
                            counts[a][b] += 1
                            counts[b][a] += 1
                    
                    current_penalty += delta_penalty
                    
                    if current_penalty == 0 and current_edges > best_edges:
                        best_edges = current_edges
                        best_mask = mask
            
            T *= cooling_rate
        
        return best_mask, best_edges

    # Run multiple SA instances with different seeds
    best_overall_mask = 0
    best_overall_edges = 0
    
    # Get a good initial guess
    init_mask = greedy_init()
    
    # Run SA
    m, e = run_sa(init_mask, deadline)
    if e > best_overall_edges:
        best_overall_edges = e
        best_overall_mask = m
        
    # Try another seed
    init_mask2 = greedy_init()
    m2, e2 = run_sa(init_mask2, deadline)
    if e2 > best_overall_edges:
        best_overall_edges = e2
        best_overall_mask = m2

    # Convert mask to edge list
    edges = []
    for e_idx in range(TOTAL_EDGES):
        if (best_overall_mask >> e_idx) & 1:
            u, v = edge_to_uv[e_idx]
            edges.append([u, v])
            
    return edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Edges: {len(edges)}")
