import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for all K4s and edge-to-K4 mapping
    def get_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)
        for e in edges:
            edge_to_k4s[e].append(len(k4_edges) - 1)
            
    num_k4 = len(k4_edges)

    def count_distinct_colors(coloring, edges):
        """
        Returns the number of distinct colors in the given edges.
        """
        # Using a set is fast enough for 6 elements
        return len(set(coloring[e] for e in edges))

    def calculate_violations(coloring):
        """
        Returns a list of indices of K4s that violate the condition (< 5 colors).
        """
        violations = []
        for i, edges in enumerate(k4_edges):
            if count_distinct_colors(coloring, edges) < 5:
                violations.append(i)
        return violations

    def is_valid(coloring):
        return len(calculate_violations(coloring)) == 0

    def get_max_color(coloring):
        return max(coloring) if coloring else 0

    # 1. Initial Structured Construction
    # The bound f(16) is between 13 and 15.
    # A known construction for edge-coloring K_n such that every K4 has many colors 
    # often relies on modular arithmetic or graph decompositions.
    # Let's try a modular construction: color(i, j) = (i + j) % k.
    # For K4 with vertices a,b,c,d, edges are (a+b, a+c, a+d, b+c, b+d, c+d) mod k.
    # We want at least 5 distinct values.
    # If k is large enough, this works. 
    # The seed used k=17 (17 colors). 
    # Can we do better with a different structure?
    # Let's try a "Latin Square" style or similar.
    # Actually, the modular sum (i+j)%k is quite strong.
    # Let's initialize with k=15 and see if it's valid. If not, we repair.
    
    # Initialize with a random valid-ish structure or just random with k=15
    # To be safe, let's start with the known valid (i+j)%17 and then try to reduce.
    # Or start with random k=15 and repair.
    # Repairing from random k=15 might be hard if many violations.
    # Let's start with (i+j)%15 and check.
    
    coloring = [((i + j) % 15) for i in range(n) for j in range(i + 1, n)]
    
    # Check if valid
    violations = calculate_violations(coloring)
    
    # If not valid, try a few other simple structures or just start SA from random
    if violations:
        # Try (i*j) % 15? No, many zeros.
        # Try random restart with k=15 until we find one with few violations?
        # Or just start SA from this invalid state, but SA can handle invalid states.
        # However, starting valid is often better for "minimize colors" phase.
        
        # Let's try to repair the initial (i+j)%15 state quickly
        # If it's too broken, we might just use random.
        
        # Actually, (i+j)%15 for n=16:
        # Vertices 0..15.
        # K4 {0,1,2,3}: edges (1,2,3, 3,4,5) mod 15 -> {1,2,3,4,5} -> 5 colors. OK.
        # K4 {0,1,2,14}: edges (1,2,14, 3,15=0, 16=1) mod 15 -> {1,2,14,3,0} -> 5 colors. OK.
        # It seems (i+j)%15 might actually be valid or close to it.
        # Let's assume we can repair it if needed.
        
        pass # We will use the repair loop below regardless.

    # 2. Local Search / Repair Loop
    # Strategy:
    # While time remains:
    #   1. If valid, try to reduce palette size.
    #      - Find max color used.
    #      - Try to remap all edges of that color to other colors randomly.
    #      - If valid, keep it.
    #      - If invalid, revert and try a different subset or different target colors.
    #   2. If invalid, repair violations.
    #      - Pick a random violating K4.
    #      - Pick a random edge in that K4.
    #      - Change its color to a random color in the current palette (or slightly larger).
    #      - Check if this reduces total violations. If yes, keep. If no, revert.
    
    current_max_colors = get_max_color(coloring)
    best_coloring = coloring.copy()
    best_max_colors = current_max_colors
    
    # If initial is invalid, we need a repair strategy that can escape deep holes.
    # Simple hill climbing on violations might get stuck.
    # Let's use a "simulated annealing" style move for violations, but strictly local.
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline - 2.0:
        
        violations = calculate_violations(coloring)
        
        if not violations:
            # Valid state! Try to minimize colors.
            current_max = get_max_color(coloring)
            
            if current_max < best_max_colors:
                best_max_colors = current_max
                best_coloring = coloring.copy()
                
            # Try to remove the highest color
            if current_max > 0:
                color_to_remove = current_max
                edges_with_color = [i for i, c in enumerate(coloring) if c == color_to_remove]
                
                if edges_with_color:
                    # Try to recolor these edges to lower colors
                    # We do this in a batch to avoid intermediate invalid states blocking us
                    # Or one by one? One by one is safer for validity but slower.
                    # Let's try a "batch" random recoloring of a subset.
                    
                    success = False
                    # Try a few times
                    for attempt in range(20):
                        trial = coloring.copy()
                        # Pick a random subset of edges to change? Or all?
                        # If we change all, it might break many K4s.
                        # Let's try changing all edges of this color to random lower colors.
                        for e in edges_with_color:
                            trial[e] = random.randint(0, current_max - 1)
                        
                        # Check validity
                        if is_valid(trial):
                            coloring = trial
                            success = True
                            break
                        else:
                            # Maybe try changing just one edge at a time?
                            # For now, if batch fails, we might try single edge swaps later.
                            pass
                    
                    if success:
                        # Reset temperature for a bit?
                        T = 5.0
                        continue
                    else:
                        # If we can't remove the highest color easily, 
                        # maybe we should perturb the coloring to find a new valid state
                        # that allows color removal.
                        # Pick a random edge and change it, accepting if it doesn't increase violations (which is 0, so it must stay 0).
                        # So we are doing local search in the space of valid colorings.
                        e_idx = random.randint(0, num_edges - 1)
                        old_c = coloring[e_idx]
                        new_c = random.randint(0, current_max - 1)
                        if old_c != new_c:
                            coloring[e_idx] = new_c
                            if is_valid(coloring):
                                pass # Keep it
                            else:
                                coloring[e_idx] = old_c # Revert
                        T *= cooling
                        continue

        else:
            # Invalid state. Repair.
            # Pick a random violation
            v_idx = random.choice(violations)
            edges = k4_edges[v_idx]
            
            # Pick a random edge in the violating K4
            e_idx = random.choice(edges)
            old_c = coloring[e_idx]
            
            # Try a new color
            # To increase distinct colors, we might need a new color or an unused one in this K4.
            # Let's just pick a random color from 0 to current_max (or slightly higher)
            new_c = random.randint(0, current_max)
            
            if old_c == new_c:
                continue
                
            coloring[e_idx] = new_c
            
            # Calculate new violations count
            # Optimization: Only check K4s affected by e_idx
            # affected_k4s = edge_to_k4s[e_idx]
            # old_violation_count = len(violations)
            # new_violation_count = sum(1 for k in affected_k4s if count_distinct_colors(coloring, k4_edges[k]) < 5)
            # ... plus violations not affected (unchanged)
            
            # Simpler: Just count all violations if the number of edges is small (120) and K4s (1820).
            # 1820 * 6 lookups is ~10k ops. Very fast.
            new_violations = calculate_violations(coloring)
            
            delta = len(new_violations) - len(violations)
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < pow(2.71828, -delta / T)):
                # Keep the change
                # Update violations list for next iteration? 
                # We recalculate at the start of the loop anyway.
                pass
            else:
                coloring[e_idx] = old_c
                
            T *= cooling
            if T < 0.01:
                T = 0.01

    # Return the best valid coloring found
    # Ensure best_coloring is valid
    if not is_valid(best_coloring):
        # Fallback: use the current coloring if it's valid, or a known valid one
        if is_valid(coloring):
            best_coloring = coloring
        else:
            # Last resort: (i+j)%17 is always valid
            best_coloring = [((i + j) % 17) for i in range(n) for j in range(i + 1, n)]
            
    return best_coloring
