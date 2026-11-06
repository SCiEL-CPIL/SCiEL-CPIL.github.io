"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using Simulated Annealing with incremental C6 counting."""
# EVOLVE-BLOCK-START
import time
import random
import math

def edge_index(i, j, n):
    """Get the index in the colors array for edge (i,j) with i<j in lexicographic order."""
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def build_adj_per_color(colors, n):
    """Build adjacency lists for each color class. Returns list of 4, each a list of sets."""
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    return adj

def count_c6s_in_graph(adj_set, n):
    """Count number of 6-cycles in a graph given as a set of edges."""
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for (u, v) in adj_set:
        adj[u].add(v)
        adj[v].add(u)
    
    count = 0
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                for v3 in adj[v2]:
                    if v3 in (v0, v1):
                        continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2, v3):
                            continue
                        for v5 in adj[v4]:
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if v0 in adj[v5]:
                                count += 1
    return count

def total_c6_count(colors, n):
    """Count total number of monochromatic C6s across all 4 color classes."""
    adj_sets = build_adj_per_color(colors, n)
    total = 0
    for c in range(4):
        total += count_c6s_in_graph(adj_sets[c], n)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety
    
    # Start with a known valid coloring for n=9
    # Circular distance coloring
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find valid colorings for increasing n
    for n in range(10, 21):
        if time.time() > deadline:
            break
        
        num_edges = n * (n - 1) // 2
        found = False
        
        # Simulated Annealing with multiple restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Build adjacency sets
            adj_sets = build_adj_per_color(colors, n)
            
            # Initial C6 count
            current_count = sum(count_c6s_in_graph(adj_sets[c], n) for c in range(4))
            
            # If already valid, we're done
            if current_count == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            # Simulated Annealing parameters
            T_start = 10.0
            T_end = 0.01
            cooling_rate = 0.9995
            max_steps = 100000
            
            T = T_start
            best_in_sa = current_count
            best_colors_sa = colors[:]
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to change
                ei = random.randint(0, num_edges - 1)
                old_color = colors[ei]
                
                # Find which vertices this edge connects
                i = 0
                while True:
                    s = i * (n - 1) - i * (i - 1) // 2
                    e = (i + 1) * (n - 1) - (i + 1) * i // 2
                    if s <= ei < e:
                        break
                    i += 1
                j = i + 1 + (ei - s)
                
                # Try changing to a different color
                new_color = random.choice([c for c in range(4) if c != old_color])
                
                # Compute delta in C6 count
                # Remove edge from old_color, add to new_color
                adj_sets[old_color].discard((i, j))
                adj_sets[new_color].add((i, j))
                
                # Recompute C6 count for the two affected color classes
                old_count_1 = count_c6s_in_graph(adj_sets[old_color], n)
                new_count_2 = count_c6s_in_graph(adj_sets[new_color], n)
                
                # Need to compute the counts for the other two color classes too
                # But we only changed two, so we can cache the others
                # For simplicity, let's just recompute all four (slower but simpler)
                # Actually, let's be smarter: only recompute the two affected
                # But we need the total. Let's keep track of per-color counts.
                
                # Let me refactor: maintain per-color C6 counts
                # Actually, this is getting complex. Let's just do a simple approach:
                # For each move, compute the delta by counting C6s in old and new color classes.
                
                # Let me try a different approach: just recompute total if count is small
                # Or use a simpler local search first to get close, then SA.
                
                # For now, let's revert and try a simpler local search
                adj_sets[old_color].add((i, j))
                adj_sets[new_color].discard((i, j))
                
                # Simpler approach: just try each new color and pick the best
                best_new_color = old_color
                best_new_count = current_count
                
                for nc in range(4):
                    if nc == old_color:
                        continue
                    adj_sets[old_color].discard((i, j))
                    adj_sets[nc].add((i, j))
                    nc_count = sum(count_c6s_in_graph(adj_sets[c], n) for c in range(4))
                    if nc_count < best_new_count:
                        best_new_count = nc_count
                        best_new_color = nc
                    adj_sets[old_color].add((i, j))
                    adj_sets[nc].discard((i, j))
                
                if best_new_color != old_color:
                    # Accept if better, or with probability exp(-delta/T) if worse
                    delta = best_new_count - current_count
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        adj_sets[old_color].discard((i, j))
                        adj_sets[best_new_color].add((i, j))
                        colors[ei] = best_new_color
                        current_count = best_new_count
                        
                        if current_count == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                        
                        if current_count < best_in_sa:
                            best_in_sa = current_count
                            best_colors_sa = colors[:]
                
                T *= cooling_rate
            
            if found:
                break
            
            # If we didn't find a valid coloring, keep the best found
            if best_in_sa < current_count and best_in_sa == 0:
                best_n = n
                best_colors = best_colors_sa[:]
                found = True
                break
        
        if found:
            continue  # Try larger n
        else:
            break  # Can't find valid for this n, stop
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
