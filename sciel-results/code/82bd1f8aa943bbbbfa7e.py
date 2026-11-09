import time
import random
import numpy as np
from itertools import combinations

def _color_index(i, j):
    """Index in the flat list for edge (i, j) with i < j. Note: n is implicit."""
    return i * (i - 1) // 2 + (j - i - 1)

def _build_paley(p):
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _build_adj_matrices(colors, n):
    """Build adjacency matrices for both colors."""
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(2)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _color_index(i, j)
            c = colors[idx]
            adj[c][i][j] = 1
            adj[c][j][i] = 1
    return adj

def _count_violations_fast(colors, n, v):
    """
    Count monochromatic K2,2,2 subgraphs involving vertex v.
    """
    violations = 0
    adj = _build_adj_matrices(colors, n)
    
    for c in range(2):
        neigh = np.where(adj[c][v] == 1)[0]
        L = len(neigh)
        if L < 6:
            continue
        
        # Iterate over all 6-combinations of neighbors
        for combo in combinations(neigh, 6):
            a, b, c1, d, e, f = combo
            
            pairings = [
                ((a,b),(c1,d),(e,f)),
                ((a,b),(c1,e),(d,f)),
                ((a,b),(c1,f),(d,e)),
                ((a,c1),(b,d),(e,f)),
                ((a,c1),(b,e),(d,f)),
                ((a,c1),(b,f),(d,e)),
                ((a,d),(b,c1),(e,f)),
                ((a,d),(b,e),(c1,f)),
                ((a,d),(b,f),(c1,e)),
                ((a,e),(b,c1),(d,f)),
                ((a,e),(b,d),(c1,f)),
                ((a,e),(b,f),(c,d)),
                ((a,f),(b,c1),(d,e)),
                ((a,f),(b,d),(c1,e)),
                ((a,f),(b,e),(c1,d))
            ]
            
            for p1, p2, p3 in pairings:
                valid = True
                for u in p1:
                    for w in p2:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                    for w in p3:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in p2:
                        for w in p3:
                            if adj[c][u][w] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    violations += 1
    
    return violations

def _total_violations(colors, n):
    """Count total monochromatic K2,2,2 subgraphs in the entire graph."""
    total = 0
    adj = _build_adj_matrices(colors, n)
    
    for c in range(2):
        # For each 6-subset, check if it contains K2,2,2 in color c
        for combo in combinations(range(n), 6):
            a, b, c1, d, e, f = combo
            
            pairings = [
                ((a,b),(c1,d),(e,f)),
                ((a,b),(c1,e),(d,f)),
                ((a,b),(c1,f),(d,e)),
                ((a,c1),(b,d),(e,f)),
                ((a,c1),(b,e),(d,f)),
                ((a,c1),(b,f),(d,e)),
                ((a,d),(b,c1),(e,f)),
                ((a,d),(b,e),(c1,f)),
                ((a,d),(b,f),(c1,e)),
                ((a,e),(b,c1),(d,f)),
                ((a,e),(b,d),(c1,f)),
                ((a,e),(b,f),(c,d)),
                ((a,f),(b,c1),(d,e)),
                ((a,f),(b,d),(c1,e)),
                ((a,f),(b,e),(c1,d))
            ]
            
            for p1, p2, p3 in pairings:
                valid = True
                for u in p1:
                    for w in p2:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                    for w in p3:
                        if adj[c][u][w] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in p2:
                        for w in p3:
                            if adj[c][u][w] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    total += 1
    
    return total

def _is_valid_full(colors, n):
    """Check if the coloring is valid (no mono K2,2,2)."""
    return _total_violations(colors, n) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Verify the starting coloring is valid
    if not _is_valid_full(colors, n):
        return {'n': n, 'colors': colors}
    
    # Try to add vertex 30 using simulated annealing
    new_n = n + 1  # 30
    best_colors = None
    best_energy = float('inf')
    
    # Simulated annealing to find a valid 30-vertex coloring
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Initialize: copy 29-vertex coloring, add random edges for vertex 29
        cur_colors = colors[:]
        v_new = new_n - 1
        for i in range(v_new):
            cur_colors.append(random.randint(0, 1))
        
        # Initial energy
        cur_energy = _total_violations(cur_colors, new_n)
        
        if cur_energy == 0:
            best_colors = cur_colors[:]
            best_energy = 0
            break
        
        # Simulated annealing parameters
        T = 2.0
        T_min = 0.01
        cooling_rate = 0.995
        max_steps = 2000
        
        best_cur_colors = cur_colors[:]
        best_cur_energy = cur_energy
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            
            # Random move: flip a random edge
            idx = random.randint(0, len(cur_colors) - 1)
            cur_colors[idx] = 1 - cur_colors[idx]
            
            # Compute new energy
            new_energy = _total_violations(cur_colors, new_n)
            
            # Acceptance criterion
            if new_energy < cur_energy:
                cur_energy = new_energy
                if cur_energy < best_cur_energy:
                    best_cur_energy = cur_energy
                    best_cur_colors = cur_colors[:]
                    if cur_energy == 0:
                        break
            else:
                # Accept with probability exp(-(new_energy - cur_energy) / T)
                delta = new_energy - cur_energy
                if delta > 0 and random.random() < np.exp(-delta / T):
                    cur_energy = new_energy
                else:
                    # Reject: undo the flip
                    cur_colors[idx] = 1 - cur_colors[idx]
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Track best solution
        if best_cur_energy < best_energy:
            best_energy = best_cur_energy
            best_colors = best_cur_colors[:]
        
        if best_energy == 0:
            break
    
    # If we found a valid 30-vertex coloring, use it
    if best_colors is not None and best_energy == 0:
        if _is_valid_full(best_colors, new_n):
            colors = best_colors
            n = new_n
    
    # Final validation
    if not _is_valid_full(colors, n):
        # Fallback: return the valid Paley 29
        colors = _build_paley(29)
        n = 29
    
    return {'n': n, 'colors': colors}
