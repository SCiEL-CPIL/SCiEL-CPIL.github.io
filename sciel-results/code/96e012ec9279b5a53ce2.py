import time
import sys
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute cycle edge indices for a given n
    # We need to know, for any edge (u,v), which C6 cycles contain it.
    # And for each such cycle, what are the other 5 edges?
    
    def get_cycle_data(n):
        """
        Returns:
        edge_map: dict (u,v) -> idx
        edge_list: list of (u,v)
        edge_to_cycles: list of lists of cycle_ids
        cycle_edges: list of lists of edge_ids (length 6)
        """
        edge_map = {}
        edge_list = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                edge_list.append((i, j))
                idx += 1
        
        num_edges = len(edge_list)
        
        # Generate all C6 cycles
        # A cycle is defined by 6 distinct vertices.
        # To avoid duplicates, we fix the smallest vertex as v0.
        # Then we permute the other 5? 
        # Actually, it's easier to just iterate all combinations of 6 vertices 
        # and generate the 30 unique cycles for that set.
        
        cycles = []
        seen_cycle_keys = set()
        
        # For efficiency, we can precompute the "other 5 edges" for each edge in the cycle
        # But first, let's just get the list of edge indices for each cycle.
        
        # Optimization: 
        # For each set of 6 vertices, there are 30 cycles.
        # Total cycles for n=18: C(18,6) * 30 = 18564 * 30 = 556,920.
        # This is a lot to precompute in pure Python loops.
        # Let's use a smarter representation.
        
        # Alternative: 
        # Instead of precomputing ALL cycles, we can compute violations on the fly?
        # No, we need fast updates.
        
        # Let's try to precompute but efficiently.
        # We will store for each edge, a list of (cycle_id, other_5_edges).
        
        edge_to_cycles = [[] for _ in range(num_edges)]
        cycle_edges_list = [] # List of 6-tuples of edge indices
        
        # To speed up generation, we iterate over combinations of 6 vertices.
        # For each combination, we generate the 30 cycles.
        
        # Helper to get edge index
        def get_ei(u, v):
            if u > v: u, v = v, u
            return edge_map[(u, v)]
        
        # Generate cycles
        # We will use itertools.combinations for vertices
        # And a hardcoded list of the 30 permutations of indices 0..5 
        # that represent unique cycles on 6 vertices.
        
        # The 30 unique cycles on vertices 0,1,2,3,4,5:
        # Fix 0. Permute 1..5.
        # Condition: perm[0] < perm[4] to break reversal symmetry?
        # Let's just generate all 120 perms of 1..5, and use a set to dedup.
        
        # Actually, 30 is small. Let's just hardcode the structure or generate once.
        # For n=18, 18564 sets. 18564 * 120 = 2.2M iterations.
        # This might take 5-10 seconds in Python. 
        # We have 115s. This is acceptable.
        
        # Let's optimize the inner loop.
        
        # Precompute the 30 unique edge-sets for a canonical 6-vertex set {0,1,2,3,4,5}
        # The edges are (0,1), (1,2), (2,3), (3,4), (4,5), (5,0) for the cycle 0-1-2-3-4-5-0.
        # We need the 30 distinct edge sets.
        
        canonical_cycles = set()
        v0, v1, v2, v3, v4, v5 = 0, 1, 2, 3, 4, 5
        # Fix v0=0.
        # Permute v1..v5.
        for perm in __import__('itertools').permutations([1, 2, 3, 4, 5]):
            # Break reversal: perm[0] < perm[4]
            if perm[0] > perm[4]:
                continue
            
            verts = [0] + list(perm)
            edges = []
            for k in range(6):
                u = verts[k]
                v = verts[(k+1) % 6]
                if u > v: u, v = v, u
                edges.append((u, v))
            
            # Canonicalize edge set
            edge_set = tuple(sorted(edges))
            canonical_cycles.add(edge_set)
            
        # canonical_cycles should have size 30.
        
        # Now, for each combination of 6 vertices in 0..n-1:
        # Map the canonical edges to the actual vertices.
        
        # This is still slow if we do it for 18k sets.
        # Let's try to do it.
        
        for verts in __import__('itertools').combinations(range(n), 6):
            # Map 0..5 to verts[0]..verts[5]
            # The canonical cycles are defined on 0..5.
            # For each canonical cycle (which is a set of 6 pairs from 0..5):
            # Replace each vertex x in the pair with verts[x].
            
            for can_edge_set in canonical_cycles:
                # Transform edges
                actual_edges = []
                for u, v in can_edge_set:
                    au = verts[u]
                    av = verts[v]
                    if au > av: au, av = av, au
                    actual_edges.append(edge_map[(au, av)])
                
                # Sort to create a key for this specific cycle in the graph
                # Since the vertices are distinct and from a combination, 
                # different canonical cycles will produce different edge sets?
                # Yes, because the structure is preserved.
                
                cycle_key = tuple(sorted(actual_edges))
                
                # We don't strictly need to dedup again if canonical_cycles are unique 
                # and verts are unique, but let's be safe.
                # Actually, we need to add this to edge_to_cycles.
                
                cycle_id = len(cycle_edges_list)
                cycle_edges_list.append(actual_edges)
                
                for e_idx in actual_edges:
                    edge_to_cycles[e_idx].append((cycle_id, actual_edges))
        
        return edge_map, edge_list, edge_to_cycles, cycle_edges_list

    def solve_local_search(n, time_limit):
        """
        Uses local search to find a 4-coloring of K_n without mono-C6.
        """
        # Precomputation
        edge_map, edge_list, edge_to_cycles, cycle_edges_list = get_cycle_data(n)
        num_edges = len(edge_list)
        num_cycles = len(cycle_edges_list)
        
        if num_edges == 0:
            return []
        
        # Initialize colors randomly
        colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
        
        # Calculate initial violation count
        # A violation is a cycle where all 6 edges have the same color.
        
        # To update efficiently, we maintain:
        # cycle_mono_count[cycle_id] = 1 if the cycle is monochromatic, 0 otherwise.
        # total_violations = sum(cycle_mono_count)
        
        # However, checking all cycles initially is O(num_cycles * 6).
        # For n=18, 556k cycles. 3.3M ops. Fast.
        
        cycle_is_mono = np.zeros(num_cycles, dtype=np.int8)
        
        for c_idx in range(num_cycles):
            edges = cycle_edges_list[c_idx]
            # Check if all 6 edges have same color
            c0 = colors[edges[0]]
            if c0 == colors[edges[1]] == colors[edges[2]] == colors[edges[3]] == colors[edges[4]] == colors[edges[5]]:
                cycle_is_mono[c_idx] = 1
        
        total_violations = np.sum(cycle_is_mono)
        
        if total_violations == 0:
            return colors.tolist()
        
        # Local Search Loop
        # We pick an edge, change its color, and update violations.
        
        # To speed up updates:
        # When we change color of edge e from c_old to c_new:
        # 1. For each cycle containing e:
        #    - It was mono if cycle_is_mono[c_idx] == 1.
        #    - After change, is it still mono?
        #    - We need to check the other 5 edges.
        
        # This is O(deg(e)) per step. deg(e) is ~1600 for n=18.
        # 1600 * 6 ops = ~10k ops per step.
        # 100,000 steps = 1M seconds? No.
        # 10k ops is very fast in Python? Maybe 1ms.
        # 100,000 steps = 100 seconds.
        # This is tight.
        
        # Optimization:
        # We can precompute for each edge e, the list of (cycle_id, other_5_edges).
        # This is already in edge_to_cycles.
        
        # Let's refine the update.
        
        # We will use a "greedy" strategy:
        # Pick a random edge involved in a violation.
        # Try all 3 other colors.
        # Pick the one that reduces violations the most.
        # If no reduction, pick best (least increase) or random.
        
        # To pick an edge involved in a violation:
        # We can maintain a list of violating cycles.
        # If total_violations > 0, pick a random violating cycle, pick a random edge from it.
        
        violating_cycles = np.where(cycle_is_mono == 1)[0]
        
        while time.time() < time_limit:
            if total_violations == 0:
                return colors.tolist()
            
            # Pick a random violating cycle
            c_idx = violating_cycles[random.randint(0, len(violating_cycles)-1)]
            edges = cycle_edges_list[c_idx]
            
            # Pick a random edge from this cycle
            e_idx = edges[random.randint(0, 5)]
            
            old_color = colors[e_idx]
            
            # Try other 3 colors
            best_color = -1
            best_delta = 1000000
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                # Simulate change
                # We need to check all cycles containing e_idx.
                delta = 0
                for (cid, other_edges) in edge_to_cycles[e_idx]:
                    # Check if cycle cid was mono before
                    was_mono = cycle_is_mono[cid]
                    
                    # Check if it will be mono after
                    # The cycle contains e_idx.
                    # Other 5 edges are in other_edges.
                    # Note: other_edges is the full list of 6 edges including e_idx?
                    # In get_cycle_data, we appended actual_edges (all 6) to edge_to_cycles.
                    # So other_edges is the list of 6 edge indices.
                    
                    # Let's find the color of the other 5 edges.
                    # If any of the other 5 has a color != new_color, it's not mono.
                    # If all 5 are new_color, and e_idx becomes new_color, it is mono.
                    
                    is_mono_after = True
                    for oe in other_edges:
                        if oe == e_idx:
                            continue
                        if colors[oe] != new_color:
                            is_mono_after = False
                            break
                    
                    if was_mono:
                        if not is_mono_after:
                            delta -= 1
                    else:
                        if is_mono_after:
                            delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_color = new_color
            
            # Apply the best color
            if best_color != -1:
                # Update total_violations
                # We calculated delta for best_color.
                # But we need to update cycle_is_mono for all cycles containing e_idx.
                
                for (cid, other_edges) in edge_to_cycles[e_idx]:
                    # Re-evaluate mono status
                    is_mono_now = True
                    for oe in other_edges:
                        if oe == e_idx:
                            continue
                        if colors[oe] != best_color:
                            is_mono_now = False
                            break
                    # Note: e_idx is now best_color.
                    
                    if is_mono_now:
                        if cycle_is_mono[cid] == 0:
                            total_violations += 1
                        cycle_is_mono[cid] = 1
                    else:
                        if cycle_is_mono[cid] == 1:
                            total_violations -= 1
                        cycle_is_mono[cid] = 0
                
                colors[e_idx] = best_color
                
                # Update violating_cycles list
                # This is O(N) to rebuild.
                # For 100k steps, this might be slow.
                # We can lazily update or just use a set.
                # Let's just rebuild if time permits, or use a simpler heuristic:
                # Just pick a random edge if we can't find a violating one easily.
                # But we need to know if total_violations == 0.
                
                # Rebuild violating_cycles every 100 steps?
                # Or just keep it updated.
                # Rebuilding is 556k checks. ~0.05s.
                # If we do it every step, 100k steps = 5000s. Too slow.
                
                # Alternative: 
                # We don't need the exact list of violating cycles to pick an edge.
                # We can just pick a random edge from the whole graph?
                # No, that's inefficient.
                
                # Let's maintain a "dirty" flag or just rebuild every 1000 steps?
                # Or, we can pick a random edge, and if it's not in a violation, 
                # we might not reduce violations.
                
                # Better: 
                # We know total_violations.
                # If total_violations > 0, we *want* to pick an edge in a violation.
                # We can maintain a list of violating cycles, but update it incrementally.
                # When a cycle becomes mono, add to list. When it stops, remove.
                # Removal from list is O(N).
                
                # Let's just use a random edge from the graph, but bias towards edges 
                # that are likely in violations?
                
                # Actually, for n=17/18, the search space is small enough 
                # that random restarts + greedy hill climbing on random edges 
                # might work well.
                
                # Let's try: Pick a random edge.
                # If we get stuck, restart.
                
                # But we need to exit if total_violations == 0.
                
                # Let's rebuild violating_cycles every 500 steps.
                # And in between, pick a random edge.
                
                pass # Placeholder for logic below
            
        # Fallback if loop finishes without finding 0 violations
        return None

    # Strategy:
    # 1. Try n=18 with local search for 60s.
    # 2. If fails, try n=17 with local search for 50s.
    # 3. If fails, return n=16 or 17 hardcoded if possible?
    #    Actually, n=17 is known to exist. Local search should find it quickly.
    
    # To make local search faster, we can use a better initialization.
    # Random is often enough for Ramsey graphs.
    
    # Let's optimize the `solve_local_search` to be more robust.
    # The rebuild of violating_cycles is the bottleneck.
    # We can avoid it by just picking random edges.
    # If we pick a random edge, the probability it is in a violation is 
    # total_violations * 6 / num_edges? No.
    # Number of edges in violations / total edges.
    
    # Let's try a simpler local search that doesn't track violating cycles explicitly,
    # but just picks random edges and tries to reduce violations.
    
    def solve_local_search_fast(n, time_limit):
        edge_map, edge_list, edge_to_cycles, cycle_edges_list = get_cycle_data(n)
        num_edges = len(edge_list)
        num_cycles = len(cycle_edges_list)
        
        if num_edges == 0:
            return []
        
        colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
        
        # Precompute cycle mono status
        cycle_is_mono = np.zeros(num_cycles, dtype=np.int8)
        for c_idx in range(num_cycles):
            edges = cycle_edges_list[c_idx]
            c0 = colors[edges[0]]
            if c0 == colors[edges[1]] == colors[edges[2]] == colors[edges[3]] == colors[edges[4]] == colors[edges[5]]:
                cycle_is_mono[c_idx] = 1
        
        total_violations = np.sum(cycle_is_mono)
        
        if total_violations == 0:
            return colors.tolist()
        
        # Local Search
        # We will pick random edges.
        # To improve efficiency, we can use a "tabu" list or just restarts.
        
        steps = 0
        while time.time() < time_limit:
            if total_violations == 0:
                return colors.tolist()
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Try other colors
            best_color = -1
            best_delta = 1000000
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                delta = 0
                for (cid, other_edges) in edge_to_cycles[e_idx]:
                    was_mono = cycle_is_mono[cid]
                    
                    is_mono_after = True
                    for oe in other_edges:
                        if oe == e_idx:
                            continue
                        if colors[oe] != new_color:
                            is_mono_after = False
                            break
                    
                    if was_mono:
                        if not is_mono_after:
                            delta -= 1
                    else:
                        if is_mono_after:
                            delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_color = new_color
            
            # Apply best color
            if best_color != -1:
                # Update state
                for (cid, other_edges) in edge_to_cycles[e_idx]:
                    is_mono_now = True
                    for oe in other_edges:
                        if oe == e_idx:
                            continue
                        if colors[oe] != best_color:
                            is_mono_now = False
                            break
                    
                    if is_mono_now:
                        if cycle_is_mono[cid] == 0:
                            total_violations += 1
                        cycle_is_mono[cid] = 1
                    else:
                        if cycle_is_mono[cid] == 1:
                            total_violations -= 1
                        cycle_is_mono[cid] = 0
                
                colors[e_idx] = best_color
            
            steps += 1
            # If we are stuck (no improvement in last 1000 steps), restart?
            # For now, just continue.
        
        return None

    best_n = 0
    best_colors = None
    
    # Try n=18
    t_start = time.time()
    res_18 = solve_local_search_fast(18, t_start + 50)
    if res_18 is not None:
        best_n = 18
        best_colors = res_18
    
    # If n=18 failed, try n=17
    if best_n < 17:
        t_start = time.time()
        res_17 = solve_local_search_fast(17, t_start + 50)
        if res_17 is not None:
            best_n = 17
            best_colors = res_17
    
    # Fallback to n=16 if 17 failed (should not happen)
    if best_n < 16:
        t_start = time.time()
        res_16 = solve_local_search_fast(16, t_start + 10)
        if res_16 is not None:
            best_n = 16
            best_colors = res_16
        else:
            # Hardcoded fallback for n=15?
            # Just return n=15 random?
            # Let's return a small valid one.
            n = 9
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    c = min(j - i, n - j + i) - 1
                    if c >= 4: c = 3
                    colors.append(c)
            return {'n': n, 'colors': colors}
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}, colors length={len(res['colors'])}")
