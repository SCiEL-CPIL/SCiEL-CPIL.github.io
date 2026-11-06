import time
import random
import sys
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
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
        # Fix vertex 0 in each 6-combination to reduce symmetry
        vertices = list(range(n))
        
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = combo[1:]
            seen_cycles = set()
            
            for perm in itertools.permutations(others):
                # Avoid reverse duplicates: enforce perm[0] < perm[-1]
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    e = edge_index(u, v)
                    cycle_edges.append(e)
                
                # Canonical form: sort the edge indices
                key = tuple(sorted(cycle_edges))
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycle_id = len(cycles)
                    cycles.append(cycle_edges)
                    for e in cycle_edges:
                        edge_to_cycles[e].append(cycle_id)
        
        return edge_map, cycles, edge_to_cycles
    
    def initial_coloring(n, strategy='structured'):
        """Generate an initial coloring using structured patterns."""
        m = n * (n - 1) // 2
        colors = [0] * m
        idx = 0
        
        if strategy == 'cyclic':
            # Cyclic coloring: color(i,j) = (j - i) % 4
            for i in range(n):
                for j in range(i + 1, n):
                    colors[idx] = (j - i) % 4
                    idx += 1
        elif strategy == 'random':
            for i in range(m):
                colors[i] = random.randint(0, 3)
        else:
            # Mixed: use modular arithmetic with some randomness
            for i in range(n):
                for j in range(i + 1, n):
                    base = (j - i) % 4
                    if random.random() < 0.2:
                        base = (base + 1) % 4
                    colors[idx] = base
                    idx += 1
        
        return colors
    
    def count_bad_cycles(colors, cycles):
        """Count monochromatic C6 cycles."""
        bad = 0
        for ce in cycles:
            c0 = colors[ce[0]]
            mono = True
            for e in ce[1:]:
                if colors[e] != c0:
                    mono = False
                    break
            if mono:
                bad += 1
        return bad
    
    def simulated_annealing(n, time_budget, initial_colors=None):
        """Run simulated annealing to minimize monochromatic C6 count."""
        if time.time() > deadline - 5:
            return None
        
        try:
            edge_map, cycles, edge_to_cycles = precompute_cycles(n)
        except Exception:
            return None
        
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        # Initialize
        if initial_colors is None:
            colors = initial_coloring(n, 'cyclic')
        else:
            colors = initial_colors[:]
        
        # Count initial bad cycles
        bad_count = count_bad_cycles(colors, cycles)
        if bad_count == 0:
            return {'n': n, 'colors': colors}
        
        best_colors = colors[:]
        best_bad = bad_count
        
        # Simulated annealing parameters
        T = 5.0
        cool_rate = 0.9995
        max_iter = 200000
        start_time_sa = time.time()
        
        for i in range(max_iter):
            if time.time() > start_time_sa + time_budget:
                break
            if time.time() > deadline - 5:
                break
            
            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]
            
            # Pick new color (different from old)
            other_colors = [c for c in range(4) if c != c_old]
            c_new = random.choice(other_colors)
            
            # Calculate delta in bad cycle count
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                
                # Check if was monochromatic before
                was_mono = True
                for ed in ce:
                    if ed != e and colors[ed] != c_old:
                        was_mono = False
                        break
                
                # Check if will be monochromatic after
                will_be_mono = True
                for ed in ce:
                    if ed != e and colors[ed] != c_new:
                        will_be_mono = False
                        break
                
                if was_mono:
                    delta -= 1
                if will_be_mono:
                    delta += 1
            
            # Acceptance criterion
            accept = False
            if delta <= 0:
                accept = True
            else:
                if delta / T < 50:  # Avoid underflow
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                colors[e] = c_new
                bad_count += delta
                
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors}
            
            T *= cool_rate
        
        if best_bad == 0:
            return {'n': n, 'colors': best_colors}
        return None
    
    def verify_coloring(n, colors):
        """Verify that no color class contains a C6."""
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        
        # Build adjacency for each color
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 3:
                    return False
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for C6 in each color class using BFS
        for c in range(4):
            for start in range(n):
                # BFS to find path of length 5 from start back to start
                # Actually, we need to check if there's a cycle of length 6
                # Use DFS with depth limit
                
                def has_c6_from(start, color):
                    # Check if there's a C6 containing start in color class
                    # DFS: find path of length 5 from start to a neighbor of start
                    stack = [(start, 1 << start, 0)]
                    while stack:
                        node, visited, depth = stack.pop()
                        if depth == 5:
                            # Check if node is connected to start
                            if adj[color][node] & (1 << start):
                                return True
                            continue
                        
                        neighbors = adj[color][node]
                        while neighbors:
                            lsb = neighbors & -neighbors
                            nb = lsb.bit_length() - 1
                            if not (visited & lsb):
                                stack.append((nb, visited | lsb, depth + 1))
                            neighbors ^= lsb
                    return False
                
                if has_c6_from(start, c):
                    return False
        
        return True
    
    # Try to find valid colorings for decreasing n
    # Allocate time based on difficulty
    targets = [17, 16, 15, 14, 13, 12, 11, 10]
    
    # Time allocation: give more time to higher n
    time_allocations = {
        17: 60,
        16: 30,
        15: 15,
        14: 10,
        13: 5,
        12: 3,
        11: 2,
        10: 1,
    }
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        alloc = time_allocations.get(n, 5)
        # Don't exceed remaining time
        remaining = deadline - 10 - time.time()
        alloc = min(alloc, remaining)
        
        if alloc < 1:
            break
        
        # Try multiple strategies
        strategies = ['cyclic', 'random', 'mixed']
        
        found = False
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            
            init_colors = initial_coloring(n, strat)
            result = simulated_annealing(n, alloc, init_colors)
            
            if result and result['n'] == n:
                # Verify the coloring
                if verify_coloring(n, result['colors']):
                    best_result = result
                    found = True
                    break
                else:
                    # Invalid, discard
                    result = None
        
        if found:
            break  # Found valid coloring for this n, stop
    
    # If we didn't find anything better, keep the fallback
    if best_result['n'] <= 5:
        # Try a quick random search for small n
        for n in [8, 7, 6]:
            if time.time() > deadline - 5:
                break
            for _ in range(10):
                colors = initial_coloring(n, 'random')
                if verify_coloring(n, colors):
                    best_result = {'n': n, 'colors': colors}
                    break
            if best_result['n'] >= n:
                break
    
    return best_result
