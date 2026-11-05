import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit
    
    # Fallback solution (seed)
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_result = {'n': n_seed, 'colors': seed_colors}
    
    # Target n values to try, in descending order of difficulty/rarity
    # We know 17 is possible. 18 is the frontier.
    # We will try 17 first, then 18 if time permits.
    target_ns = [17, 18]
    
    for n in target_ns:
        if time.time() > deadline - 5.0:
            break
            
        try:
            result = solve_n(n, deadline)
            if result is not None:
                # Found a valid coloring for this n
                # Since we iterate descending, the first found is the best
                return result
        except Exception as e:
            # If solving fails for any reason, continue to next n or fallback
            pass
            
    return best_result

def solve_n(n, deadline):
    # Precompute all C6 cycles
    # This might take a few seconds for n=18
    if time.time() > deadline - 10.0:
        return None
        
    vertices = list(range(n))
    cycles = []
    edge_to_cycles = {}
    
    # Generate all C6s
    # A C6 is a set of 6 edges forming a cycle.
    # We iterate over all 6-subsets of vertices.
    # For each subset, we generate all 60 distinct cycles.
    
    # Helper to get edge index
    def edge_idx(u, v):
        if u > v:
            u, v = v, u
        # Map (u,v) to index
        # We can use a dictionary or a formula
        # Formula: sum_{k=0}^{u-1} (n-1-k) + (v-u-1) ?
        # Simpler: use a dict for mapping during precomp, or just store tuples
        # For speed, let's use a dict mapping (u,v) to index
        pass
    
    # We need a fast way to map edges to indices for the color array
    # Let's create the mapping
    edge_map = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_map[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
            
    num_edges = len(edge_list)
    
    # Initialize edge_to_cycles
    for e in range(num_edges):
        edge_to_cycles[e] = []
        
    # Generate cycles
    # To avoid duplicates, we can enforce that the smallest vertex in the cycle is the first one,
    # and the second smallest is either the second or the last?
    # Standard canonical form for cycles:
    # Rotate so min vertex is at pos 0.
    # If the neighbor of min vertex in clockwise direction is larger than the neighbor in counter-clockwise, 
    # keep it, else reverse.
    
    for subset in itertools.combinations(vertices, 6):
        # Generate all 60 permutations that form a cycle
        # Fix subset[0] as start? No, subset is sorted.
        # Let's just generate all permutations of the 6 vertices
        # and check if they form a simple cycle (they always do in K_n)
        # and canonicalize them.
        
        seen_canonical = set()
        for perm in itertools.permutations(subset):
            # Canonicalize:
            # 1. Rotate so min element is at index 0
            min_idx = perm.index(min(perm))
            rot = perm[min_idx:] + perm[:min_idx]
            # 2. Check direction
            # If rot[1] > rot[-1], reverse
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            # Now rot is canonical
            if rot in seen_canonical:
                continue
            seen_canonical.add(rot)
            
            # Extract edges
            edges = []
            for k in range(6):
                u = rot[k]
                v = rot[(k+1)%6]
                if u > v:
                    u, v = v, u
                edges.append(edge_map[(u, v)])
            
            # Store cycle
            cycle_id = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cycle_id)
                
    num_cycles = len(cycles)
    
    # If no cycles found (n < 6), return trivial solution
    if num_cycles == 0:
        return {'n': n, 'colors': [0]*num_edges}
        
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Initialize cycle counts
    # cycle_counts[c][color] = number of edges in cycle c with that color
    cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
    total_violations = 0
    
    for c_id in range(num_cycles):
        edges = cycles[c_id]
        for e in edges:
            c = colors[e]
            cycle_counts[c_id][c] += 1
        # Check violation
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_violations += 1
                break # Only count once per cycle
        
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We want to run until deadline
    # Estimate steps per second
    # Each step involves ~5000 updates
    # Let's just run until time limit
    
    while time.time() < deadline - 2.0:
        if total_violations == 0:
            return {'n': n, 'colors': colors}
            
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        # Calculate delta
        delta = 0
        affected_cycles = edge_to_cycles[e]
        
        for c_id in affected_cycles:
            cc = cycle_counts[c_id]
            old_count = cc[old_color]
            new_count = cc[new_color]
            
            # Before change
            old_violation = 1 if old_count == 6 else 0
            # Note: It's possible that another color was 6? No, sum is 6.
            # So only one color can be 6.
            
            # After change
            cc[old_color] -= 1
            cc[new_color] += 1
            new_count = cc[new_color]
            
            new_violation = 1 if new_count == 6 else 0
            
            delta += new_violation - old_violation
            
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            colors[e] = new_color
            total_violations += delta
            T *= cooling
            if T < min_T:
                T = min_T
        else:
            # Reject: revert changes
            for c_id in affected_cycles:
                cc = cycle_counts[c_id]
                cc[old_color] += 1
                cc[new_color] -= 1
            T *= cooling
            if T < min_T:
                T = min_T

    # If we exit loop, check if we have a solution
    if total_violations == 0:
        return {'n': n, 'colors': colors}
        
    return None
# EVOLVE-BLOCK-END
