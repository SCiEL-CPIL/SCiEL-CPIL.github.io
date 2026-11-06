import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def generate_cycles(n):
        """Generate all C6 cycles in K_n as lists of edge indices."""
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            # Generate unique cycles on these 6 vertices
            # Fix v0 at position 0. Permute the other 5.
            # To avoid duplicates from reversal, require perm[0] < perm[-1]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    edges.append(get_edge_index(u, v, n))
                cycles.append(tuple(edges))
        return cycles

    def try_solve(n):
        num_edges = n * (n - 1) // 2
        
        # Precompute cycles
        cycles = generate_cycles(n)
        num_cycles = len(cycles)
        if num_cycles == 0:
            return None
            
        # Map edge index to list of cycle indices containing it
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_cycles[e_idx].append(c_idx)
        
        # Store cycles as numpy array for fast access if needed, 
        # but for incremental updates, we need to check specific cycles.
        # We will keep cycles as list of tuples for easy iteration in update logic.
        
        # State
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
        
        # Initial conflict calculation
        # mono_status[c_idx] = True if cycle c_idx is monochromatic
        mono_status = np.zeros(num_cycles, dtype=bool)
        total_conflicts = 0
        
        for c_idx, cycle in enumerate(cycles):
            c0 = colors[cycle[0]]
            if all(colors[e] == c0 for e in cycle[1:]):
                mono_status[c_idx] = True
                total_conflicts += 1
                
        if total_conflicts == 0:
            return colors.tolist()
            
        best_conflicts = total_conflicts
        best_colors = colors.copy()
        
        # Local Search with Incremental Updates
        max_restarts = 500
        steps_per_restart = 50000 # High, but incremental update is fast
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            if r == 0:
                # Use initial random
                pass
            else:
                colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
                mono_status = np.zeros(num_cycles, dtype=bool)
                total_conflicts = 0
                for c_idx, cycle in enumerate(cycles):
                    c0 = colors[cycle[0]]
                    if all(colors[e] == c0 for e in cycle[1:]):
                        mono_status[c_idx] = True
                        total_conflicts += 1
                if total_conflicts < best_conflicts:
                    best_conflicts = total_conflicts
                    best_colors = colors.copy()
                if total_conflicts == 0:
                    return colors.tolist()
                    
            current_conflicts = total_conflicts
            
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    break
                    
                # Pick a random edge
                e_idx = np.random.randint(num_edges)
                old_color = colors[e_idx]
                
                # Find best new color
                best_new_color = -1
                best_delta = 0 # We want to minimize conflicts, so delta < 0 is good
                
                # Try all other 3 colors
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    # Calculate delta in conflicts
                    # A cycle changes status if it contains e_idx.
                    # It can only become mono or stop being mono.
                    delta = 0
                    for c_idx in edge_to_cycles[e_idx]:
                        cycle = cycles[c_idx]
                        # Check if this cycle is affected
                        # We need to know if it was mono before and if it is mono after.
                        
                        # Optimization: 
                        # If it was NOT mono before:
                        #   It can only become mono if all other 5 edges are new_color.
                        # If it WAS mono before:
                        #   It was mono with old_color. Changing e_idx to new_color != old_color breaks it.
                        #   So it becomes NOT mono.
                        
                        was_mono = mono_status[c_idx]
                        
                        if was_mono:
                            # It was mono with old_color.
                            # Changing e_idx to new_color != old_color breaks mono status.
                            # So it becomes NOT mono.
                            # Conflict count decreases by 1.
                            delta -= 1
                        else:
                            # It was NOT mono.
                            # It can become mono only if all other 5 edges are new_color.
                            # Check other 5 edges.
                            # cycle is tuple of 6 edges.
                            # Check if all e in cycle, e != e_idx, have colors[e] == new_color
                            # This is O(5) per cycle.
                            is_potential_mono = True
                            for e in cycle:
                                if e == e_idx:
                                    continue
                                if colors[e] != new_color:
                                    is_potential_mono = False
                                    break
                            if is_potential_mono:
                                # It becomes mono.
                                # Conflict count increases by 1.
                                delta += 1
                                                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                        
                # If best_delta < 0, we have an improving move
                if best_new_color != -1 and best_delta < 0:
                    # Apply move
                    colors[e_idx] = best_new_color
                    current_conflicts += best_delta
                    
                    # Update mono_status for affected cycles
                    for c_idx in edge_to_cycles[e_idx]:
                        cycle = cycles[c_idx]
                        # Re-evaluate this cycle
                        c0 = colors[cycle[0]]
                        is_mono = all(colors[e] == c0 for e in cycle[1:])
                        mono_status[c_idx] = is_mono
                        
                    if current_conflicts == 0:
                        return colors.tolist()
                        
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        best_colors = colors.copy()
                        
                else:
                    # No improving move found for this edge.
                    # To escape local minima, we might accept a neutral or worsening move occasionally?
                    # Or just restart if we get stuck?
                    # Simple hill climbing gets stuck easily. 
                    # Let's add a small probability of accepting a neutral move (delta == 0) to wander.
                    if best_delta == 0 and random.random() < 0.1:
                         colors[e_idx] = best_new_color
                         current_conflicts += best_delta
                         for c_idx in edge_to_cycles[e_idx]:
                             cycle = cycles[c_idx]
                             c0 = colors[cycle[0]]
                             is_mono = all(colors[e] == c0 for e in cycle[1:])
                             mono_status[c_idx] = is_mono
                             if current_conflicts == 0:
                                 return colors.tolist()
                    else:
                        # Stuck? Let's try a random jump or restart sooner.
                        # For simplicity, if we fail to find improvement in a large chunk of steps, restart.
                        # But we don't track "stuck" easily here without a counter.
                        # Let's just continue. The random edge selection will eventually find a way out or we restart.
                        pass
        
        if best_conflicts == 0:
            return best_colors.tolist()
        return None

    # Try n=17
    n_target = 17
    result = try_solve(n_target)
    if result is not None:
        return {'n': n_target, 'colors': result}
        
    # Fallback to n=16 or lower if 17 fails (unlikely with this method)
    # But 17 is the known max. If it fails, return seed n=9 to be safe?
    # No, better to try 16.
    n_target = 16
    result = try_solve(n_target)
    if result is not None:
        return {'n': n_target, 'colors': result}
        
    # Fallback to seed solution n=9
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}

