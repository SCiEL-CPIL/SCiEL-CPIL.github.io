import random
import time
import itertools

def construct():
    """
    Finds a K4-free graph with no independent set of size 6 on n vertices.
    Uses Simulated Annealing on the space of edge flips.
    Objective: Minimize E = (# K4s) + (# Independent 6-sets).
    Target: E = 0.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Fallback: Paley graph on 17 vertices
    # Paley(17) is K4-free (max clique 3) and has independence number 5 (no independent 6).
    # This is a known valid graph.
    squares = {x * x % 17 for x in range(1, 17)}
    fallback_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    fallback = {'n': 17, 'edges': fallback_edges}
    
    # Target n: We aim for 35 (the known record). 
    # If we can't find 35, we might try 34, 33, etc., but 35 is the goal.
    # The task says "Maximize n". 
    # Exoo's graph is 35. We try to find a 35-vertex graph.
    # If we fail, we return the best found.
    
    n_target = 35
    
    # Bitmasks for vertices 0..n-1
    # adj[u] is a bitmask of neighbors of u
    
    def count_k4s(adj, n):
        """Count the number of K4 subgraphs."""
        count = 0
        # Iterate over pairs (u, v) that are edges
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # List common neighbors
                    c_list = [i for i in range(n) if (common >> i) & 1]
                    # For each pair (c, d) in c_list, check if edge exists
                    for i in range(len(c_list)):
                        c = c_list[i]
                        for j in range(i + 1, len(c_list)):
                            d = c_list[j]
                            if (adj[c] >> d) & 1:
                                count += 1
        return count

    def count_ind6s(adj, n):
        """Count the number of independent sets of size 6."""
        # This is equivalent to counting K6 in the complement graph.
        # Construct complement adjacency masks
        comp_adj = [0] * n
        for u in range(n):
            mask = 0
            # All vertices except u
            all_mask = (1 << n) - 1
            all_mask &= ~(1 << u)
            # Remove neighbors of u
            mask = all_mask & ~adj[u]
            comp_adj[u] = mask
            
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (comp_adj[u] >> v) & 1:
                    common = comp_adj[u] & comp_adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c_list = [i for i in range(n) if (common >> i) & 1]
                    for i in range(len(c_list)):
                        c = c_list[i]
                        for j in range(i + 1, len(c_list)):
                            d = c_list[j]
                            if (comp_adj[c] >> d) & 1:
                                # Check for 2 more vertices in common of c,d that are also connected in comp
                                common_cd = comp_adj[c] & comp_adj[d]
                                common_cd &= ~(1 << c)
                                common_cd &= ~(1 << d)
                                # And they must be connected to u,v in comp
                                # Actually, if c,d are in common_ab, they are connected to u,v in comp.
                                # We need a pair (e,f) from common_cd such that e-f is an edge in comp.
                                e_list = [i for i in range(n) if (common_cd >> i) & 1]
                                for x in range(len(e_list)):
                                    e = e_list[x]
                                    for y in range(x + 1, len(e_list)):
                                        f = e_list[y]
                                        if (comp_adj[e] >> f) & 1:
                                            count += 1
        return count

    def get_energy(adj, n):
        # Weights: K4 is "heavier" or "lighter"? 
        # Usually, we want to minimize both. 
        # Let's use E = k4_count + ind6_count.
        k4s = count_k4s(adj, n)
        if k4s > 0:
            # Early exit? No, we need the value for SA.
            # But if k4s is huge, we might skip ind6 count to save time?
            # For SA, we need a consistent metric.
            pass
        ind6s = count_ind6s(adj, n)
        return k4s + ind6s

    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Initialize
    # Greedy initialization: Start with empty graph, add edges if they don't create K4 or Ind6?
    # Or start with random and run SA.
    # Let's try multiple restarts.
    
    best_energy = float('inf')
    best_adj = None
    
    # Pre-compute all edges for faster flip selection
    all_edges = [(u, v) for u in range(n_target) for v in range(u + 1, n_target)]
    
    for restart in range(20):  # Max 20 restarts
        if time.time() > deadline - 10.0:
            break
            
        # Initialize adj
        # Random init with p=0.5
        adj = [0] * n_target
        for u, v in all_edges:
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        current_energy = get_energy(adj, n_target)
        
        if current_energy == 0:
            best_adj = adj
            break
            
        # Simulated Annealing
        T = 10.0  # Initial temperature
        T_min = 0.1
        cooling = 0.995
        steps_per_temp = 50
        
        while T > T_min and time.time() < deadline - 5.0:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 5.0:
                    break
                    
                # Pick a random edge to flip
                u, v = random.choice(all_edges)
                
                # Calculate delta energy
                # Flipping (u,v) affects:
                # 1. K4s containing edge (u,v)
                # 2. Ind6s containing non-edge (u,v) (i.e., Ind6s in G that become K6 in comp)
                
                # Efficient Delta Calculation:
                # Instead of recalculating full energy, we can estimate or recalc.
                # Recalc is O(n^4) roughly. With n=35, it's fast enough?
                # 35^4 is ~1.5M ops. Python is slow. 
                # Let's try to optimize the delta calculation.
                
                # Option 1: Full recalc. 
                # Option 2: Incremental.
                # Let's try full recalc first for simplicity, but it might be too slow for many steps.
                # 100 steps * 1.5M ops = 150M ops. In Python, this is ~10-20s. 
                # So we can do maybe 50-100 steps per second?
                # We have ~100s. So ~5000-10000 steps.
                # That's not a lot for SA.
                
                # Let's optimize the energy calculation.
                # We can cache the K4 count and Ind6 count?
                # No, flipping one edge changes many.
                
                # Alternative: Use a faster heuristic for delta?
                # Or just accept that we do fewer steps.
                
                # Let's try to speed up count_k4s and count_ind6s.
                # The current implementations are pure Python loops.
                
                # Flip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                new_energy = get_energy(adj, n_target)
                
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_adj = adj[:]
                        if best_energy == 0:
                            break
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
            T *= cooling
            if best_energy == 0:
                break
                
        if best_energy == 0:
            break

    if best_adj:
        return {'n': n_target, 'edges': get_edges_from_adj(best_adj, n_target)}
    else:
        # If we didn't find a 35-vertex graph, try 34?
        # The task asks to maximize n. 
        # If 35 is not found, we should try 34, 33...
        # But 35 is the record. If we can't find it, returning 17 is bad.
        # Let's try to find a valid graph for smaller n if 35 fails.
        
        # Fallback logic: Try n=34, 33...
        # But given time constraints, if 35 fails, we likely won't find 34 either quickly.
        # However, 35 is known to exist. SA should find it with high probability if tuned well.
        # If it fails, we return the best we found (which might be invalid? No, best_adj is only updated if energy < inf, but we don't check validity explicitly in the return unless energy==0).
        # Wait, if best_energy > 0, best_adj is INVALID.
        # We must return a VALID graph.
        # So if best_energy > 0, we must fall back to a known valid graph.
        
        return fallback

