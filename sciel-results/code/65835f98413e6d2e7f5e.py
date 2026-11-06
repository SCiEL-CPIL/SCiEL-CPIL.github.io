import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def generate_c6s(n):
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (2 * n - i - 1) // 2 + (j - i - 1)

        cycles = []
        for verts in itertools.combinations(range(n), 6):
            for perm in itertools.permutations(verts[1:]):
                if perm[0] < perm[4]:
                    cycle_verts = [verts[0]] + list(perm)
                    edges = []
                    for k in range(6):
                        u = cycle_verts[k]
                        v = cycle_verts[(k + 1) % 6]
                        edges.append(edge_idx(u, v))
                    cycles.append(tuple(edges))
        return cycles

    def solve_n(n):
        if n < 6:
            return {'n': n, 'colors': [0] * (n * (n - 1) // 2)}
        
        num_edges = n * (n - 1) // 2
        cycles = generate_c6s(n)
        
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, cycle in enumerate(cycles):
            for e in cycle:
                edge_to_cycles[e].append(c_idx)
        
        # Pre-compute "other edges" for each (cycle, edge_in_cycle) pair?
        # No, just store cycle edges.
        
        colors = random.choices([0, 1, 2, 3], k=num_edges)
        
        def count_bad():
            count = 0
            for cycle in cycles:
                c0 = colors[cycle[0]]
                if all(colors[e] == c0 for e in cycle[1:]):
                    count += 1
            return count
        
        current_bad = count_bad()
        if current_bad == 0:
            return {'n': n, 'colors': colors}
        
        best_colors = colors[:]
        best_bad = current_bad
        
        temp = 1.0
        cool = 0.9999
        
        while time.time() < deadline - 2:
            if current_bad == 0:
                return {'n': n, 'colors': colors}
            
            e = random.randint(0, num_edges - 1)
            old_c = colors[e]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            delta = 0
            for c_idx in edge_to_cycles[e]:
                cycle = cycles[c_idx]
                
                # Check if mono before
                # All other edges must be old_c
                mono_before = True
                for ed in cycle:
                    if ed == e:
                        continue
                    if colors[ed] != old_c:
                        mono_before = False
                        break
                
                # Check if mono after
                # All other edges must be new_c
                mono_after = True
                for ed in cycle:
                    if ed == e:
                        continue
                    if colors[ed] != new_c:
                        mono_after = False
                        break
                
                if mono_before:
                    delta -= 1
                if mono_after:
                    delta += 1
            
            if delta < 0 or random.random() < pow(temp, delta):
                colors[e] = new_c
                current_bad += delta
                
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = colors[:]
            
            temp *= cool
            if temp < 0.01:
                temp = 0.01
        
        if best_bad == 0:
            return {'n': n, 'colors': best_colors}
        return None

    for n in range(17, 9, -1):
        if time.time() > deadline:
            break
        result = solve_n(n)
        if result:
            return result
    
    n = 9
    return {'n': n, 'colors': [0] * (n * (n - 1) // 2)}
