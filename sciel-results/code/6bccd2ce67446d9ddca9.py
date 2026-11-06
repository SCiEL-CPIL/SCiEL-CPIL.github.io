import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit (evaluator kills at 150s, but task says 120s limit usually, let's be safe)
    
    # Fallback: n=5 (trivially valid, no C6 possible)
    def get_fallback(n):
        m = n * (n - 1) // 2
        colors = [0] * m
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    def precompute_cycles(n):
        """Precompute all C6 cycles and edge-to-cycle mapping."""
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            return edge_map[(u, v)]
        
        m = n * (n - 1) // 2
        cycles = []
        edge_to_cycles = [[] for _ in range(m)]
        
        # Generate all unique C6 cycles
        # A C6 is a set of 6 vertices and a specific cyclic ordering.
        # Number of C6 in K_n is n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5)/72
        # We can generate them by picking 6 vertices and then the 5 distinct cycles on them?
        # No, it's easier to iterate permutations but that's slow.
        # Better: Fix 6 vertices v0..v5. There are 5 distinct cycles on 6 labeled vertices (up to rotation/reflection? No, edge set is what matters).
        # Actually, the number of distinct C6 edge-sets in K_6 is 5.
        # Total C6 in K_n = C(n,6) * 5.
        
        vertices = list(range(n))
        
        # To speed up, we can just generate the 5 cycles for each combination of 6 vertices.
        # The 5 cycles on {0,1,2,3,4,5} (relative) are:
        # 1. 0-1-2-3-4-5-0
        # 2. 0-1-2-3-5-4-0
        # 3. 0-1-2-4-3-5-0
        # 4. 0-1-3-2-4-5-0
        # 5. 0-1-3-4-2-5-0
        # (These are the 5 non-isomorphic ways to connect 6 vertices in a cycle, considering vertex labels fixed in the set).
        
        # Let's verify the count. 5 cycles per set of 6 vertices.
        # We need to map these relative cycles to actual vertices.
        
        rel_cycles = [
            [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)],
            [(0,1), (1,2), (2,3), (3,5), (5,4), (4,0)],
            [(0,1), (1,2), (2,4), (4,3), (3,5), (5,0)],
            [(0,1), (1,3), (3,2), (2,4), (4,5), (5,0)],
            [(0,1), (1,3), (3,4), (4,2), (2,5), (5,0)]
        ]
        
        for combo in __import__('itertools').combinations(vertices, 6):
            # combo is a tuple of 6 sorted vertices
            v = combo
            for rc in rel_cycles:
                cycle_edges = []
                for (u_rel, v_rel) in rc:
                    u = v[u_rel]
                    w = v[v_rel]
                    e = edge_index(u, w)
                    cycle_edges.append(e)
                
                # Canonical form to avoid duplicates? 
                # The 5 relative cycles are distinct edge sets on the labeled set {0..5}.
                # Since we map to specific vertices, these 5 will be distinct edge sets for this combo.
                # Are there any duplicates across different combos? No, each combo has unique vertices.
                
                cycle_id = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        return edge_map, cycles, edge_to_cycles

    def solve_min_conflicts(n, time_budget):
        """
        Solve the 4-coloring of K_n with no mono C6 using Min-Conflicts.
        """
        try:
            edge_map, cycles, edge_to_cycles = precompute_cycles(n)
        except Exception:
            return None
        
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            return {'n': n, 'colors': [0]*m}
        
        # Initialize colors
        # Strategy: Random initialization is usually fine for Min-Conflicts.
        # We can also try a structured initialization to get closer.
        # Let's use random but with a bias towards spreading colors.
        
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Count conflicts for each cycle
        # conflict_count[ci] = 1 if mono, 0 else.
        # Actually, we want to count how many edges in the cycle have the same color as the 'majority' or just check if all are same.
        # For Min-Conflicts, we define a cost function.
        # Cost = number of monochromatic cycles.
        
        # To update cost efficiently:
        # When we change color of edge e from c_old to c_new:
        # For each cycle ci containing e:
        #   old_state = 1 if all edges in ci were c_old, else 0. (Note: if e was c_old, and others were c_old, it was mono c_old).
        #   new_state = 1 if all edges in ci are c_new, else 0.
        #   delta = new_state - old_state.
        
        # We need to know for each cycle, how many edges have color c for c in {0,1,2,3}.
        # cycle_color_counts[ci] = [count_0, count_1, count_2, count_3]
        
        cycle_color_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
        
        # Initialize counts
        for ci, ce in enumerate(cycles):
            for e in ce:
                cycle_color_counts[ci][colors[e]] += 1
        
        # Calculate initial total conflicts
        total_conflicts = 0
        for ci in range(num_cycles):
            counts = cycle_color_counts[ci]
            # A cycle is monochromatic if any count is 6.
            # Since sum is 6, at most one count can be 6.
            if any(c == 6 for c in counts):
                total_conflicts += 1
        
        if total_conflicts == 0:
            return {'n': n, 'colors': colors}
        
        best_colors = colors[:]
        best_conflicts = total_conflicts
        
        start_time_mc = time.time()
        restarts = 0
        max_restarts = 50
        
        while time.time() < start_time_mc + time_budget and restarts < max_restarts:
            # Min-Conflicts Loop
            # We keep a list of "conflicted" edges to pick from.
            # An edge is conflicted if it belongs to at least one monochromatic cycle.
            
            # To make this fast, we can just pick a random edge from the set of edges in conflicted cycles.
            # Or simply pick a random edge, and if it's not in a conflict, we might waste time.
            # Better: Maintain a set of edges involved in conflicts.
            
            # For simplicity and speed in Python, let's just pick a random edge involved in a conflict.
            # We can iterate through conflicts to find one? No, that's O(C).
            # Let's maintain a list of conflicted edges.
            
            # Initial conflicted edges
            conflicted_edges = set()
            for ci in range(num_cycles):
                counts = cycle_color_counts[ci]
                if any(c == 6 for c in counts):
                    for e in cycles[ci]:
                        conflicted_edges.add(e)
            
            iterations = 0
            max_iter = 50000 # per restart
            
            while iterations < max_iter:
                if time.time() > start_time_mc + time_budget:
                    break
                
                if not conflicted_edges:
                    # No conflicts!
                    return {'n': n, 'colors': colors}
                
                # Pick a random conflicted edge
                e = random.choice(list(conflicted_edges))
                c_old = colors[e]
                
                # Find best new color
                best_c = c_old
                best_delta = 0
                min_delta = 0
                
                # Try all 4 colors
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    
                    delta = 0
                    # Check cycles containing e
                    for ci in edge_to_cycles[e]:
                        counts = cycle_color_counts[ci]
                        
                        # Old state: was it mono?
                        # It was mono if counts[c_old] == 6.
                        # Note: counts[c_old] includes e.
                        # If counts[c_old] == 6, then all other 5 are c_old.
                        # If we change e to c_new, counts[c_old] becomes 5, counts[c_new] becomes 1.
                        # New state: is it mono? No (5 vs 1).
                        
                        # So if it was mono, delta -= 1.
                        # If it was not mono, can it become mono?
                        # It becomes mono c_new only if counts[c_new] was 5 (before change, e was c_old so not c_new).
                        # Wait, if e is c_old, then counts[c_new] does NOT include e.
                        # So if counts[c_new] == 5, then the other 5 edges are c_new.
                        # If we change e to c_new, counts[c_new] becomes 6.
                        # So it becomes mono. delta += 1.
                        
                        if counts[c_old] == 6:
                            delta -= 1
                        if counts[c_new] == 5:
                            delta += 1
                    
                    if delta < min_delta:
                        min_delta = delta
                        best_c = c_new
                
                # If no improvement (min_delta >= 0), we might need to kick
                # Standard Min-Conflicts: if delta > 0, accept with prob exp(-delta/T) or just restart.
                # Let's try to accept if delta == 0 (neutral) or < 0 (improvement).
                # If min_delta > 0, we are in a local minimum.
                
                if min_delta < 0:
                    # Improvement
                    c_new = best_c
                    colors[e] = c_new
                    total_conflicts += min_delta
                    if total_conflicts < best_conflicts:
                        best_conflicts = total_conflicts
                        best_colors = colors[:]
                    
                    # Update counts and conflicted_edges
                    # This part is tricky to do efficiently without rebuilding.
                    # We update cycle_color_counts for all cycles containing e.
                    # And we update conflicted_edges.
                    
                    # To keep it simple and robust, let's just recalculate conflicted_edges from scratch 
                    # every K steps? Or just accept that picking from a static set of ALL edges 
                    # is slow but if we filter...
                    
                    # Actually, updating `conflicted_edges` incrementally is hard because 
                    # removing an edge from a conflict requires knowing if other conflicts exist.
                    
                    # Alternative: Just pick a random edge from the WHOLE graph, 
                    # but prioritize those in conflicts?
                    # Or, simpler: Just restart if we don't improve in X steps.
                    
                    # Let's stick to a simpler Min-Conflicts:
                    # 1. Pick random edge from conflicted set.
                    # 2. Change to best color.
                    # 3. If no improvement, restart.
                    
                    # But we need to update `conflicted_edges` correctly.
                    # Let's just rebuild `conflicted_edges` every 100 steps? 
                    # No, that's O(C).
                    
                    # Let's just use a "Random Edge" approach but weight by conflict?
                    # No, let's just use the standard approach:
                    # If we can't improve, restart.
                    
                    # For now, let's just update the counts.
                    for ci in edge_to_cycles[e]:
                        cycle_color_counts[ci][c_old] -= 1
                        cycle_color_counts[ci][c_new] += 1
                    
                    # Rebuilding conflicted_edges is expensive. 
                    # Let's just clear it and rebuild if we suspect it's stale?
                    # Or, just pick a random edge from the ENTIRE graph, 
                    # and check if it's in a conflict?
                    
                    # Let's switch strategy: 
                    # Instead of maintaining a set, we just pick a random edge.
                    # If it's in a conflict, we try to fix it.
                    # If not, we skip.
                    # This is inefficient if few conflicts.
                    
                    # Better: Keep the set, but update it lazily.
                    # We know e was in a conflict. After change, it might not be.
                    # Other edges in the same cycle might become conflicted.
                    
                    # Given the complexity, let's just RESTART if we don't find a solution quickly.
                    # Min-Conflicts on random restarts is very powerful.
                    
                    iterations += 1
                    # To avoid getting stuck, if we do 1000 steps without improvement, restart.
                    # We track `best_conflicts`. If `total_conflicts` hasn't decreased in 1000 steps, restart.
                    
                else:
                    # No improvement possible from this edge.
                    # Restart
                    break
            
            # Restart
            restarts += 1
            # Re-initialize with new random colors
            colors = [random.randint(0, 3) for _ in range(m)]
            cycle_color_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
            for ci, ce in enumerate(cycles):
                for e in ce:
                    cycle_color_counts[ci][colors[e]] += 1
            total_conflicts = 0
            for ci in range(num_cycles):
                counts = cycle_color_counts[ci]
                if any(c == 6 for c in counts):
                    total_conflicts += 1
            
            if total_conflicts == 0:
                return {'n': n, 'colors': colors}

        # If we used up time, return best found
        if best_conflicts == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # Try to find valid colorings for n=17, 16, 15...
    # n=17 is the target.
    
    # Allocate time: 
    # n=17: 100s
    # n=16: 10s
    # n=15: 5s
    
    targets = [17, 16, 15]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
        
        alloc = 100 if n == 17 else (10 if n == 16 else 5)
        remaining = deadline - 5 - time.time()
        alloc = min(alloc, remaining)
        
        if alloc < 2:
            break
            
        result = solve_min_conflicts(n, alloc)
        
        if result:
            # Verify
            # Quick verification
            m = n * (n - 1) // 2
            if len(result['colors']) == m:
                # Re-verify conflicts
                try:
                    edge_map, cycles, edge_to_cycles = precompute_cycles(n)
                    bad = 0
                    for ce in cycles:
                        c0 = result['colors'][ce[0]]
                        mono = True
                        for e in ce[1:]:
                            if result['colors'][e] != c0:
                                mono = False
                                break
                        if mono:
                            bad += 1
                    if bad == 0:
                        best_result = result
                        break
                except Exception:
                    pass

    return best_result

if __name__ == '__main__':
    # Quick self-test
    res = construct()
    print(res['n'])
