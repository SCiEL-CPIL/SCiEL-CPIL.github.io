"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks for speed
    adj_mask = [0] * n
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    def popcount(x):
        """Fast popcount for small integers."""
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def find_hamiltonian_cycle_fast(adj_masks, start_node=0, deadline=None, max_steps=5000000):
        """
        Finds a Hamiltonian cycle using bitmask-based DFS with aggressive pruning.
        """
        if deadline is None:
            deadline = start_time + 85.0
        
        n_local = len(adj_masks)
        full_mask = (1 << n_local) - 1
        
        # Pre-check: all vertices must have degree >= 2
        for u in range(n_local):
            if popcount(adj_masks[u]) < 2:
                return None
        
        steps = [0]
        path = [start_node]
        visited_mask = [1 << start_node]
        
        # Precompute neighbor lists for faster access
        neighbors = [0] * n_local
        for i in range(n_local):
            neighbors[i] = adj_masks[i]
        
        def backtrack():
            if time.time() > deadline or steps[0] > max_steps:
                return False
            
            steps[0] += 1
            current = path[-1]
            vm = visited_mask[-1]
            
            if len(path) == n_local:
                if (adj_masks[current] >> start_node) & 1:
                    return True
                return False
            
            # Unvisited neighbors of current
            candidates = adj_masks[current] & ~vm
            
            if candidates == 0:
                return False
            
            # Build list of candidate vertices
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cand_list.append(v)
                temp ^= lsb
            
            # Most constrained first (fewest unvisited neighbors)
            def priority(v):
                deg = popcount(adj_masks[v] & ~vm)
                # Penalize if v is the last step and doesn't connect to start
                if len(path) + 1 == n_local and not ((adj_masks[v] >> start_node) & 1):
                    deg = 1000
                return deg
            cand_list.sort(key=priority)
            
            for next_node in cand_list:
                # Pruning: if this is the last vertex, it must connect to start
                if len(path) + 1 == n_local:
                    if not ((adj_masks[next_node] >> start_node) & 1):
                        continue
                
                new_vm = vm | (1 << next_node)
                
                # Connectivity pruning: check unvisited vertices aren't isolated
                if len(path) > 2:
                    unvisited = full_mask & ~new_vm
                    if unvisited:
                        stuck = False
                        t = unvisited
                        while t:
                            lsb = t & (-t)
                            u = lsb.bit_length() - 1
                            t ^= lsb
                            # u must have a neighbor in visited or in unvisited
                            if (adj_masks[u] & (new_vm | unvisited)) == 0:
                                stuck = True
                                break
                        if stuck:
                            continue
                
                path.append(next_node)
                visited_mask.append(new_vm)
                if backtrack():
                    return True
                path.pop()
                visited_mask.pop()
            
            return False
        
        if backtrack():
            return path[:]
        return None
    
    def try_find_cycles(adj_masks, max_cycles=3, deadline=None, random_seed=None, start_order=None):
        """Try to find max_cycles edge-disjoint Hamiltonian cycles."""
        if deadline is None:
            deadline = start_time + 85.0
        
        cycles = []
        current_masks = adj_masks[:]
        
        for i in range(max_cycles):
            if time.time() > deadline:
                break
            
            # Check viability: every vertex must have degree >= 2
            valid = True
            for u in range(n):
                if popcount(current_masks[u]) < 2:
                    valid = False
                    break
            if not valid:
                break
            
            # Try multiple starting nodes
            cycle = None
            if start_order is not None:
                starts = start_order
            elif random_seed is not None:
                rng = random.Random(random_seed + i * 7919)
                starts = list(range(n))
                rng.shuffle(starts)
            else:
                starts = list(range(n))
                if i > 0:
                    starts = starts[i % n:] + starts[:i % n]
            
            for start in starts:
                if time.time() > deadline:
                    break
                cycle = find_hamiltonian_cycle_fast(current_masks, start_node=start, 
                                                   deadline=deadline, max_steps=3000000)
                if cycle is not None:
                    break
            
            if cycle is None:
                break
            
            cycles.append(cycle)
            
            # Remove edges of this cycle
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_masks[u] &= ~(1 << v)
                current_masks[v] &= ~(1 << u)
        
        return cycles
    
    # Strategy: Try to find 3 edge-disjoint Hamiltonian cycles
    best_cycles = []
    
    # Phase 1: Deterministic approach with natural start order
    cycles = try_find_cycles(adj_mask, max_cycles=3, deadline=deadline)
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
    
    # Phase 2: Try different start orderings
    if len(best_cycles) < 3:
        # Try starting from different vertices
        for base_start in range(0, n, 3):
            if time.time() > deadline - 5.0:
                break
            start_order = list(range(base_start, n)) + list(range(0, base_start))
            cycles = try_find_cycles(adj_mask, max_cycles=3, deadline=deadline, start_order=start_order)
            if len(cycles) > len(best_cycles):
                best_cycles = cycles
                if len(best_cycles) >= 3:
                    break
    
    # Phase 3: Randomized restarts
    if len(best_cycles) < 3:
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 3.0:
                break
            
            seed = 42 + attempt * 131
            cycles = try_find_cycles(adj_mask, max_cycles=3, deadline=deadline, random_seed=seed)
            
            if len(cycles) > len(best_cycles):
                best_cycles = cycles
                if len(best_cycles) >= 3:
                    break
    
    # Fallback: ensure we return at least one valid cycle
    if not best_cycles:
        for start in range(n):
            if time.time() > deadline - 1.0:
                break
            cycle = find_hamiltonian_cycle_fast(adj_mask, start_node=start, 
                                               deadline=deadline, max_steps=10000000)
            if cycle is not None:
                best_cycles = [cycle]
                break
        
        if not best_cycles:
            # Minimal fallback - find any valid cycle
            best_cycles = [[0, 1, 2]]
            valid = True
            for i in range(3):
                u = best_cycles[0][i]
                v = best_cycles[0][(i + 1) % 3]
                if not ((adj_mask[u] >> v) & 1):
                    valid = False
                    break
            if not valid:
                best_cycles = []
    
    # Convert to list of lists of Python ints
    result = [list(c) for c in best_cycles]
    
    return result
# EVOLVE-BLOCK-END
