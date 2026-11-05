import time
import random
import numpy as np
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 20
    num_colours = 20
    
    # Precompute all 4-cycles in K_n
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d and a pairing.
    # There are 3 pairings for {a,b,c,d}: (ab,cd), (ac,bd), (ad,bc)
    # But we need the cycle order.
    # Cycles: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a (and reverses which are same edge sets)
    # Let's just list unique edge sets.
    # Edges: (a,b), (b,c), (c,d), (d,a)
    # Edges: (a,b), (b,d), (d,c), (c,a)
    # Edges: (a,c), (c,b), (b,d), (d,a)
    
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    num_cycles = len(cycles)
    
    # Precompute for each edge, which cycles it belongs to? 
    # Or just iterate cycles. 4845 cycles is small.
    
    def check_validity(adj):
        """Returns list of indices of invalid cycles"""
        invalid = []
        for i, cycle in enumerate(cycles):
            e1, e2, e3, e4 = cycle
            # Check if all edges exist
            if adj[e1[0], e1[1]] == -1 or adj[e2[0], e2[1]] == -1 or \
               adj[e3[0], e3[1]] == -1 or adj[e4[0], e4[1]] == -1:
                continue
            c1 = adj[e1[0], e1[1]]
            c2 = adj[e2[0], e2[1]]
            c3 = adj[e3[0], e3[1]]
            c4 = adj[e4[0], e4[1]]
            if len({c1, c2, c3, c4}) < 4:
                invalid.append(i)
        return invalid

    def get_edge_count(adj):
        return int(np.sum(adj != -1)) // 2

    best_adj = None
    best_edges = 0
    
    # Strategy: Randomized Local Search
    # 1. Init with K_20 and random colors
    # 2. Find all invalid cycles
    # 3. Pick a random invalid cycle, pick a random edge in it to remove
    # 4. Repeat until no invalid cycles or time limit
    
    max_attempts = 500 # We can do many quick restarts
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Initialize complete graph with random colors
        adj = np.zeros((n, n), dtype=int)
        # Fill upper triangle with random colors
        for u in range(n):
            for v in range(u + 1, n):
                adj[u, v] = random.randint(0, num_colours - 1)
                adj[v, u] = adj[u, v]
        
        # Mark diagonal as -1 (no self loops)
        np.fill_diagonal(adj, -1)
        
        # Local Search Loop
        # We maintain a list of invalid cycles? Or just recompute?
        # Recomputing 4845 cycles is fast (~0.1ms in numpy/python if optimized, maybe 1ms pure python).
        # Let's try to be efficient.
        
        current_edges = 190
        
        # Optimization: Keep a set of "active" invalid cycles?
        # Removing an edge only affects cycles containing that edge.
        # Let's precompute edge -> cycles mapping.
        
        # Since this is inside the loop, let's precompute outside? 
        # No, let's just do a simple loop first. If too slow, optimize.
        
        # Actually, let's precompute edge_to_cycles once.
        if attempt == 0:
            edge_to_cycles = {e: [] for e in combinations(range(n), 2)}
            for i, cycle in enumerate(cycles):
                for e in cycle:
                    # Normalize edge to (min, max)
                    eu = min(e[0], e[1])
                    ev = max(e[0], e[1])
                    edge_to_cycles[(eu, ev)].append(i)
        
        # In the main loop, we can track invalid cycles.
        # Start by finding all invalid cycles.
        invalid_cycles = check_validity(adj)
        
        # If we start with random colors, there will be many invalid cycles.
        # We remove edges until none are left.
        
        # To avoid O(N^2) scanning of all cycles every step, we can use a queue or set.
        # But removing an edge makes cycles containing it "satisfied" (they don't count anymore).
        # It doesn't make other cycles invalid.
        # So the set of invalid cycles only shrinks.
        
        # So we can just keep a list of invalid cycle indices.
        # When we remove an edge, we filter the list.
        
        # However, filtering a list of size ~4000 every step might be slow if we do 1000 steps.
        # 1000 * 4000 = 4M ops. Acceptable.
        
        # Let's refine:
        # invalid_indices: list of indices into 'cycles'
        
        while invalid_cycles and time.time() < deadline:
            # Pick a random invalid cycle
            idx = random.choice(invalid_cycles)
            cycle = cycles[idx]
            
            # Pick a random edge in this cycle to remove
            # We want to maximize edges, so maybe we should pick the edge that is "most problematic"?
            # Or just random.
            # Let's try random first.
            e_idx = random.randint(0, 3)
            u, v = cycle[e_idx]
            
            # Remove edge (u, v)
            adj[u, v] = -1
            adj[v, u] = -1
            
            current_edges -= 1
            
            # Update invalid_cycles:
            # Any cycle that contains (u, v) is now no longer a "4-cycle in the graph" 
            # (it has a missing edge), so it is no longer invalid.
            # We need to remove all indices in invalid_cycles that correspond to cycles containing (u, v).
            
            # To do this efficiently, we need to know which cycles in invalid_cycles contain (u, v).
            # We can precompute for each edge, the list of cycle indices.
            # But we need to intersect with the current invalid_cycles set.
            
            # Let's use a set for invalid_cycles for fast removal? 
            # Or just filter the list.
            
            # Filter list:
            # new_invalid = [i for i in invalid_cycles if not (u,v) in cycles[i]]
            # This is O(L * 4).
            
            # Let's optimize by precomputing a "contains edge" boolean matrix or similar?
            # Or just rely on the small size.
            
            # Optimization: Precompute for each edge, a bitmask of cycles? 
            # 4845 cycles > 64 bits. Use Python int as bitmask? 
            # Python ints are arbitrary precision. 
            # Let's try that. It might be faster than list filtering.
            
            # Actually, list filtering is simple and likely fast enough for ~1000 removals.
            # 1000 removals * 4845 checks * 4 edges check = 19M ops. 
            # Python might take ~2-5 seconds. 
            # If we do 500 attempts, that's too slow.
            
            # We need to be smarter.
            
            # Alternative: 
            # Don't restart 500 times. Do one long run?
            # Or, make the removal process faster.
            
            # Let's try to make the filter faster.
            # Instead of checking all 4 edges of the cycle, we can precompute 
            # for each cycle index, the set of edges it has.
            
            # Precompute: cycle_edges[i] = set of normalized edges in cycles[i]
            
            if attempt == 0:
                cycle_edges = []
                for c in cycles:
                    es = set()
                    for e in c:
                        eu = min(e[0], e[1])
                        ev = max(e[0], e[1])
                        es.add((eu, ev))
                    cycle_edges.append(es)
                
                # Also precompute edge -> list of cycle indices (already done)
                # And maybe edge -> bitmask?
                # Let's stick to sets for now, but use a different removal strategy.
                
                # Strategy:
                # Keep a list of invalid cycle indices.
                # When removing edge e, we know which cycles contain e (from edge_to_cycles).
                # We can create a set of "to_remove" indices: 
                # to_remove = set(edge_to_cycles[e])
                # new_invalid = [i for i in invalid_cycles if i not in to_remove]
                
                # This is O(L + K) where K is number of cycles containing e.
                # K is roughly 4 * (n-2) * ... ?
                # In K20, an edge is in (n-2)(n-3)/2 * 2? No.
                # Number of 4-cycles containing a fixed edge (u,v):
                # Pick w from remaining 18, x from remaining 17. 
                # w and x must form an edge? No, in K20 all edges exist.
                # But we need a 4-cycle u-v-w-x-u.
                # w != u,v. x != u,v,w.
                # Edges (v,w), (w,x), (x,u) must exist. In K20 they do.
                # So for each w (18 choices), x can be any of the remaining 17?
                # Wait, x must be connected to u and w.
                # In K20, yes.
                # So 18 * 17 = 306 cycles per edge?
                # Plus the other direction? No, the cycle is defined by vertices.
                # Let's verify:
                # Total cycles = 4845.
                # Total edge-cycle incidences = 4845 * 4 = 19380.
                # Total edges = 190.
                # Avg cycles per edge = 19380 / 190 = 102.
                # So K ~ 102.
                
                # So the filter is O(L + 102).
                # L starts at 4845 and decreases.
                # Total work ~ sum(L_i). 
                # If we remove 100 edges, L drops to ~3800.
                # Work ~ 100 * 4000 = 400k ops. Very fast.
                
                pass # Precomputation done in first attempt
            
            # Execute filter
            e_norm = (min(u, v), max(u, v))
            to_remove = set(edge_to_cycles[e_norm])
            # List comprehension is fast
            invalid_cycles = [i for i in invalid_cycles if i not in to_remove]
            
        # End while loop
        
        # If we exited because invalid_cycles is empty, we have a valid graph.
        # If we exited because of time, we might have an invalid graph.
        # We should only update best if valid.
        
        if not invalid_cycles:
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
                
        # Early exit if we reached a very high edge count?
        # Max possible is 190.
        # If we get > 180, that's good.
        
        # Optional: Try to add edges back?
        # If we have time, we can try to add edges back to increase density.
        # But "add edge" check is expensive (needs to check all cycles containing it).
        # Given the budget, let's stick to removal.
        
    # Fallback
    if best_adj is None:
        # Return a valid small graph
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Final verification just in case
    valid = True
    for cycle in cycles:
        e1, e2, e3, e4 = cycle
        c1 = best_adj[e1[0], e1[1]]
        c2 = best_adj[e2[0], e2[1]]
        c3 = best_adj[e3[0], e3[1]]
        c4 = best_adj[e4[0], e4[1]]
        
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            continue
            
        if len({c1, c2, c3, c4}) < 4:
            valid = False
            break
            
    if not valid:
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
