"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # We want to maximize n. Known best is 17. Try 19, 18, 17, ...
    
    def solve_n(n):
        if n < 2:
            return None
        
        num_edges = n * (n - 1) // 2
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        # Generate all C6 cycles
        cycles = []
        cycle_edges = []
        
        for vertices in itertools.combinations(range(n), 6):
            v0 = vertices[0]
            others = vertices[1:]
            for perm in itertools.permutations(others):
                v1, v2, v3, v4, v5 = perm
                if v1 > v5:
                    continue
                cycle_verts = [v0, v1, v2, v3, v4, v5]
                c_edges = []
                for k in range(6):
                    u = cycle_verts[k]
                    v = cycle_verts[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    c_edges.append(edge_map[(u, v)])
                cycles.append(cycle_verts)
                cycle_edges.append(c_edges)
        
        num_cycles = len(cycles)
        
        # Build edge_to_cycles mapping
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, c_edges in enumerate(cycle_edges):
            for e_idx in c_edges:
                edge_to_cycles[e_idx].append(c_idx)
        
        best_mono_count = float('inf')
        best_colors = None
        
        # Multiple restarts with local search
        max_restarts = 500
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Initialize cycle color counts
            cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
            for c_idx, c_edges in enumerate(cycle_edges):
                for e_idx in c_edges:
                    cycle_counts[c_idx][colors[e_idx]] += 1
            
            mono_count = 0
            mono_cycles = []
            for c_idx in range(num_cycles):
                counts = cycle_counts[c_idx]
                if any(c == 6 for c in counts):
                    mono_count += 1
                    mono_cycles.append(c_idx)
            
            if mono_count == 0:
                return colors
            
            if mono_count < best_mono_count:
                best_mono_count = mono_count
                best_colors = colors[:]
            
            # Local search phase
            stuck_count = 0
            steps = 0
            while time.time() < deadline - 5:
                if mono_count == 0:
                    return colors
                
                # Periodic full rescan to refresh mono_cycles list
                if steps % 200 == 0:
                    mono_cycles = []
                    for c_idx in range(num_cycles):
                        counts = cycle_counts[c_idx]
                        if any(c == 6 for c in counts):
                            mono_cycles.append(c_idx)
                
                if not mono_cycles:
                    break
                
                # Pick a random mono-cycle
                c_idx = random.choice(mono_cycles)
                c_edges = cycle_edges[c_idx]
                
                # Pick a random edge in this cycle
                e_idx = random.choice(c_edges)
                old_color = colors[e_idx]
                
                # Evaluate all 3 other colors
                best_new_color = -1
                best_delta = float('inf')
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    delta = 0
                    for cc_idx in edge_to_cycles[e_idx]:
                        counts = cycle_counts[cc_idx]
                        if counts[old_color] == 6:
                            delta -= 1
                        if counts[new_color] == 5:
                            delta += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                
                # Acceptance criteria
                accept = False
                if best_delta < 0:
                    accept = True
                    stuck_count = 0
                elif best_delta == 0:
                    if random.random() < 0.3:
                        accept = True
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    # Accept if stuck for a while (simulated annealing style)
                    if stuck_count > 50 and random.random() < 0.02:
                        accept = True
                        stuck_count = 0
                    else:
                        stuck_count += 1
                
                if accept:
                    new_color = best_new_color
                    colors[e_idx] = new_color
                    mono_count += best_delta
                    
                    # Update cycle counts
                    for cc_idx in edge_to_cycles[e_idx]:
                        cycle_counts[cc_idx][old_color] -= 1
                        cycle_counts[cc_idx][new_color] += 1
                    
                    steps += 1
                    
                    # If mono_count increased significantly, consider restart
                    if mono_count > 50:
                        break
                
                # If stuck too long, break and restart
                if stuck_count > 500:
                    break
            
            # If this restart was good, continue; otherwise try again
            if mono_count < best_mono_count:
                best_mono_count = mono_count
                best_colors = colors[:]
        
        return best_colors
    
    # Try n from 19 down to 9
    for n in range(19, 8, -1):
        if time.time() > deadline - 10:
            break
        
        result = solve_n(n)
        if result is not None:
            return {'n': n, 'colors': result}
    
    # Fallback to the known n=9 solution
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
