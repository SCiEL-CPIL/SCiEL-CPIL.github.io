import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def build_edge_map(n):
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        return edge_map

    def generate_cycles(n):
        """Generate all distinct C6 cycles and map edges to cycles."""
        edge_map = build_edge_map(n)
        m = n * (n - 1) // 2
        cycles = []
        edge_to_cycles = [[] for _ in range(m)]
        
        vertices = list(range(n))
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = list(combo[1:])
            seen = set()
            for perm in itertools.permutations(others):
                path = [v0] + list(perm)
                # Normalize: sort the cycle to avoid duplicates
                # A cycle is determined by the set of edges
                edge_set = frozenset()
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    edge_set.add((u, v))
                if len(edge_set) != 6:
                    continue
                key = tuple(sorted(edge_set))
                if key in seen:
                    continue
                seen.add(key)
                
                # Store cycle as list of edge indices
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    cycle_edges.append(edge_map[(u, v)])
                
                ci = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(ci)
        
        return cycles, edge_to_cycles, edge_map

    def spectral_init(n, edge_map):
        """Initialize colors using spectral embedding of a random regular graph."""
        m = n * (n - 1) // 2
        colors = [0] * m
        
        # Create a random 3-regular graph (or close to it)
        # Assign each edge a "weight" based on random regular graph structure
        # Use this to create a 2D embedding and assign colors by quadrant
        
        # Build adjacency for a random regular-like graph
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            # Connect to 3 random neighbors
            candidates = [j for j in range(n) if j != i]
            random.shuffle(candidates)
            for j in candidates[:3]:
                adj[i][j] = 1.0
                adj[j][i] = 1.0
        
        # Simple power iteration for top 2 eigenvectors
        # Start with random vectors
        v1 = [random.random() for _ in range(n)]
        v2 = [random.random() for _ in range(n)]
        
        # Orthonormalize
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(x * x for x in v1) ** 0.5
        norm2 = sum(x * x for x in v2) ** 0.5
        if norm1 > 0 and norm2 > 0:
            v1 = [x / norm1 for x in v1]
            v2 = [x / norm2 for x in v2]
            dot = sum(a * b for a, b in zip(v1, v2))
            v2 = [a - dot * b for a, b in zip(v2, v1)]
            norm2 = sum(x * x for x in v2) ** 0.5
            if norm2 > 0:
                v2 = [x / norm2 for x in v2]
        
        # Power iteration
        for _ in range(20):
            # Multiply by A
            new_v1 = [sum(adj[i][j] * v1[j] for j in range(n)) for i in range(n)]
            new_v2 = [sum(adj[i][j] * v2[j] for j in range(n)) for i in range(n)]
            
            # Orthonormalize
            dot = sum(a * b for a, b in zip(new_v1, new_v2))
            norm1 = sum(x * x for x in new_v1) ** 0.5
            norm2 = sum(x * x for x in new_v2) ** 0.5
            if norm1 > 0:
                new_v1 = [x / norm1 for x in new_v1]
            if norm2 > 0:
                new_v2 = [x / norm2 for x in new_v2]
            dot = sum(a * b for a, b in zip(new_v1, new_v2))
            new_v2 = [a - dot * b for a, b in zip(new_v2, new_v1)]
            norm2 = sum(x * x for x in new_v2) ** 0.5
            if norm2 > 0:
                new_v2 = [x / norm2 for x in new_v2]
            
            v1, v2 = new_v1, new_v2
        
        # Assign colors by quadrant in the 2D embedding
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_map[(i, j)]
                # Midpoint of vertices i and j in the embedding
                mx = (v1[i] + v1[j]) / 2.0
                my = (v2[i] + v2[j]) / 2.0
                # Assign color based on quadrant
                if mx >= 0 and my >= 0:
                    colors[idx] = 0
                elif mx < 0 and my >= 0:
                    colors[idx] = 1
                elif mx < 0 and my < 0:
                    colors[idx] = 2
                else:
                    colors[idx] = 3
        
        return colors

    def count_bad(cycles, colors):
        """Count monochromatic C6 cycles."""
        bad = 0
        for ce in cycles:
            c0 = colors[ce[0]]
            if all(colors[e] == c0 for e in ce[1:]):
                bad += 1
        return bad

    def local_search(n, colors, cycles, edge_to_cycles, edge_map, time_limit):
        """Targeted local search to eliminate C6 violations."""
        m = n * (n - 1) // 2
        current_colors = colors[:]
        current_bad = count_bad(cycles, current_colors)
        
        if current_bad == 0:
            return current_colors, 0
        
        best_colors = current_colors[:]
        best_bad = current_bad
        
        # Precompute for each cycle which edges have which colors
        # This helps with fast delta computation
        
        T = 3.0
        cool_rate = 0.995
        max_iter = 50000
        
        for i in range(max_iter):
            if time.time() > time_limit:
                break
            
            # Pick a random bad cycle and try to fix it
            if current_bad > 0:
                # Find a bad cycle
                bad_cycles = []
                for ci, ce in enumerate(cycles):
                    c0 = current_colors[ce[0]]
                    if all(current_colors[e] == c0 for e in ce[1:]):
                        bad_cycles.append(ci)
                        if len(bad_cycles) >= 10:
                            break
                
                if bad_cycles:
                    ci = random.choice(bad_cycles)
                    ce = cycles[ci]
                    c_mono = current_colors[ce[0]]
                    
                    # Pick a random edge in this cycle to recolor
                    e = random.choice(ce)
                    c_old = current_colors[e]
                    # Pick a new color different from c_mono to break the cycle
                    c_new = random.choice([c for c in range(4) if c != c_mono])
                else:
                    # No bad cycles found (shouldn't happen if current_bad > 0)
                    e = random.randint(0, m - 1)
                    c_old = current_colors[e]
                    c_new = random.choice([c for c in range(4) if c != c_old])
            else:
                # Random move to explore
                e = random.randint(0, m - 1)
                c_old = current_colors[e]
                c_new = random.choice([c for c in range(4) if c != c_old])
            
            # Calculate delta
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                # Check if cycle was mono with c_old
                was_mono = True
                for ed in ce:
                    if ed != e and current_colors[ed] != c_old:
                        was_mono = False
                        break
                
                # Check if cycle will be mono with c_new
                will_be_mono = True
                for ed in ce:
                    if ed != e and current_colors[ed] != c_new:
                        will_be_mono = False
                        break
                
                if was_mono:
                    delta -= 1
                if will_be_mono:
                    delta += 1
            
            # Acceptance
            if delta < 0:
                current_colors[e] = c_new
                current_bad += delta
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = current_colors[:]
                    if best_bad == 0:
                        return best_colors, 0
            else:
                if random.random() < pow(2.718281828, -delta / T):
                    current_colors[e] = c_new
                    current_bad += delta
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_colors = current_colors[:]
                        if best_bad == 0:
                            return best_colors, 0
            
            T *= cool_rate
        
        return best_colors, best_bad

    # Try different n values in descending order
    targets = [17, 16, 15, 14, 13, 12, 11]
    
    for n in targets:
        if time.time() > deadline - 15:
            break
        
        # Generate cycles and edge map
        cycles, edge_to_cycles, edge_map = generate_cycles(n)
        m = n * (n - 1) // 2
        
        # Try multiple random restarts with spectral init
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Spectral initialization
            colors = spectral_init(n, edge_map)
            
            # Local search with time budget
            time_per_search = min(15.0, (deadline - 10 - time.time()) / max(1, 5 - restart))
            result_colors, result_bad = local_search(n, colors, cycles, edge_to_cycles, edge_map, time.time() + time_per_search)
            
            if result_bad == 0:
                return {'n': n, 'colors': result_colors}
        
        # If we found a good but not perfect coloring, save it as fallback
        # (but we only return valid colorings, so this doesn't help directly)
    
    return best_result
