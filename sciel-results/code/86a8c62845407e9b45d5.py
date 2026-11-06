import random
import time
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = n_vertices * d // 2
    
    # Time management
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Precompute edges
    # Edge id -> (u, v)
    # Vertex -> list of edge ids
    edges = []
    edge_map = {}
    vertex_edges = [[] for _ in range(n_vertices)]
    
    eid = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_map[eid] = (u, v)
                edges.append((u, v))
                vertex_edges[u].append(eid)
                vertex_edges[v].append(eid)
                eid += 1
                
    # 2. Precompute 4-cycles
    # Cycle id -> list of 4 edge ids
    # Edge id -> list of cycle ids containing it
    cycles = []
    edge_cycles = [[] for _ in range(n_edges)]
    
    # A 4-cycle is defined by a vertex u, and two distinct dimensions i, j
    # The vertices are u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j)
    # We need to map these to edge IDs.
    
    # To avoid duplicates, we can iterate u, i, j with u < ...?
    # Simpler: Iterate all u, i, j.
    # But we must ensure we don't count the same cycle 4 times.
    # A cycle has 4 vertices. We can define the cycle by its minimum vertex u_min.
    # And the two dimensions i, j.
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                # Check if u is the minimum vertex of this cycle
                if u < v1 and u < v2 and u < v3:
                    # Get edge IDs
                    # Edge (u, v1) - dim i
                    # Edge (v2, v3) - dim i
                    # Edge (u, v2) - dim j
                    # Edge (v1, v3) - dim j
                    
                    # We need a fast way to get edge ID from (u, v)
                    # Since we built edge_map, we can use a dict lookup, but that's slow.
                    # Better: precompute edge_id[u][i] ?
                    # Or just use the fact that edge ID is deterministic.
                    # Let's create a lookup dict for (min(u,v), max(u,v)) -> eid
                    # We already have edge_map, but it's eid -> (u,v).
                    # Let's create pair_to_eid.
                    
                    pass # Placeholder

    # Let's rebuild the precomputation more efficiently.
    pair_to_eid = {}
    for eid, (u, v) in enumerate(edges):
        pair_to_eid[(min(u, v), max(u, v))] = eid

    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                if u < v1 and u < v2 and u < v3:
                    e1 = pair_to_eid[(min(u, v1), max(u, v1))]
                    e2 = pair_to_eid[(min(v2, v3), max(v2, v3))]
                    e3 = pair_to_eid[(min(u, v2), max(u, v2))]
                    e4 = pair_to_eid[(min(v1, v3), max(v1, v3))]
                    
                    cycle = [e1, e2, e3, e4]
                    cycles.append(cycle)
                    for e in cycle:
                        edge_cycles[e].append(len(cycles) - 1)

    n_cycles = len(cycles)
    
    # State
    # present[eid] = 1 if edge is in the set
    # count[cid] = number of present edges in cycle cid
    
    # Initialize with a greedy random start
    random.seed(42)
    
    best_edges = []
    best_count = 0
    
    def evaluate(present):
        # Return number of edges
        return sum(present)
        
    def is_valid(present, counts):
        for c in range(n_cycles):
            if counts[c] == 4:
                return False
        return True

    def run_sa():
        # Start with random configuration
        present = [0] * n_edges
        counts = [0] * n_cycles
        
        # Random init: pick edges with prob 0.6
        for e in range(n_edges):
            if random.random() < 0.6:
                present[e] = 1
                for c in edge_cycles[e]:
                    counts[c] += 1
                    
        current_edges = sum(present)
        
        # Repair phase: remove edges from violating cycles
        # While there are violations, remove one edge from a violating cycle
        # Choose a random violating cycle, remove a random edge from it
        while True:
            violations = [c for c in range(n_cycles) if counts[c] == 4]
            if not violations:
                break
            c = random.choice(violations)
            e = random.choice(cycles[c])
            if present[e]:
                present[e] = 0
                current_edges -= 1
                for cc in edge_cycles[e]:
                    counts[cc] -= 1
                    
        best_local = current_edges
        best_local_present = present.copy()
        
        # SA Phase
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        steps = 0
        max_steps = 500000 # Will be limited by time
        
        while T > T_min:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Choose a move
            # 1. Add an edge
            # 2. Remove an edge
            # 3. Swap
            
            # Try adding a random missing edge
            e_add = random.randint(0, n_edges - 1)
            if not present[e_add]:
                # Check if it creates a violation
                # It creates a violation if any cycle containing it has count 3
                ok = True
                for c in edge_cycles[e_add]:
                    if counts[c] == 3:
                        ok = False
                        break
                if ok:
                    # Accept
                    present[e_add] = 1
                    current_edges += 1
                    for c in edge_cycles[e_add]:
                        counts[c] += 1
                    if current_edges > best_local:
                        best_local = current_edges
                        best_local_present = present.copy()
                else:
                    # Reject or accept with prob exp(-delta/T)
                    # Delta is the penalty for violation? 
                    # We are using a hard constraint repair approach initially.
                    # Let's switch to soft constraint: 
                    # Score = edges - lambda * violations
                    # This is complex to maintain incrementally with strict validity.
                    # Let's stick to: Only make moves that keep validity, or 
                    # allow violations but penalize heavily?
                    
                    # Simpler: Only add if valid.
                    # If we can't add, try to remove?
                    pass
            
            # Try removing a random present edge (to allow re-adding later? or just to explore?)
            # Usually we want to maximize, so removal is only good if it allows more additions later.
            # Let's try a "swap" or just "add if possible".
            
            # To escape local optima, we can randomly remove an edge, 
            # then try to add edges again.
            
            # Let's implement a simple "Random Restart Local Search" instead of full SA 
            # if SA is complex.
            
            # Actually, let's just do:
            # 1. Greedy Add: Scan all edges, add if possible.
            # 2. Random Shuffle Greedy Add.
            # 3. Repeat many times.
            
            T *= cooling
            steps += 1
            
        return best_local_present, best_local

    # Strategy: Multiple Greedy Random Restarts
    # Each restart:
    # 1. Randomize edge order
    # 2. Iterate, add if possible
    # 3. Record best
    
    best_result = []
    best_score = 0
    
    # Run greedy restarts
    for i in range(1000): # Will stop by time
        if time.time() - start_time > time_limit - 10:
            break
            
        # Random order
        order = list(range(n_edges))
        random.shuffle(order)
        
        present = [0] * n_edges
        counts = [0] * n_cycles
        current_edges = 0
        
        for e in order:
            # Check if adding e is valid
            ok = True
            for c in edge_cycles[e]:
                if counts[c] == 3:
                    ok = False
                    break
            if ok:
                present[e] = 1
                current_edges += 1
                for c in edge_cycles[e]:
                    counts[c] += 1
        
        if current_edges > best_score:
            best_score = current_edges
            best_result = present.copy()
            
    # Local Search on the best result found
    # Try to improve best_result
    present = best_result
    counts = [0] * n_cycles
    for e in range(n_edges):
        if present[e]:
            for c in edge_cycles[e]:
                counts[c] += 1
                
    improved = True
    while improved and (time.time() - start_time < time_limit - 5):
        improved = False
        # Try to add any edge that was missing
        for e in range(n_edges):
            if not present[e]:
                ok = True
                for c in edge_cycles[e]:
                    if counts[c] == 3:
                        ok = False
                        break
                if ok:
                    present[e] = 1
                    best_score += 1
                    for c in edge_cycles[e]:
                        counts[c] += 1
                    improved = True
        
        # Try swaps: Remove e1, add e2?
        # This is expensive. 
        # Let's try a random shuffle of the "add" phase again on the current state?
        # No, that would reset.
        
        # Let's try: Remove a random edge, then greedy add again?
        # This is "Iterated Local Search".
        
        # Pick a random edge to remove
        if best_score > 0:
            present_edges = [e for e in range(n_edges) if present[e]]
            if present_edges:
                e_remove = random.choice(present_edges)
                present[e_remove] = 0
                best_score -= 1
                for c in edge_cycles[e_remove]:
                    counts[c] -= 1
                    
                # Now greedy add
                order = list(range(n_edges))
                random.shuffle(order)
                for e in order:
                    if not present[e]:
                        ok = True
                        for c in edge_cycles[e]:
                            if counts[c] == 3:
                                ok = False
                                break
                        if ok:
                            present[e] = 1
                            best_score += 1
                            for c in edge_cycles[e]:
                                counts[c] += 1
                improved = True

    # Convert present to list of edges
    out = []
    for e in range(n_edges):
        if present[e]:
            u, v = edges[e]
            out.append([u, v])
            
    return out

construct()
