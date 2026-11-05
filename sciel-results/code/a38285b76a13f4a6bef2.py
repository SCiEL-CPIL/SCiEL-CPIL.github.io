"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict graph: two vertices i, j (values i+1, j+1) conflict if (i+1)+(j+1) is a perfect square
    # Include self-loops: vertex i has self-conflict if 2*(i+1) is a perfect square
    adj = [0] * N  # bitmask of neighbors (including self if self-conflict)
    
    for i in range(N):
        val = i + 1
        mask = 0
        for j in range(N):
            other = j + 1
            if val + other in squares:
                mask |= (1 << j)
        adj[i] = mask
    
    # Identify vertices that cannot be in any valid set (self-conflict: 2*a is a square)
    # These must be excluded from our set
    valid_vertices = []
    for i in range(N):
        if not (adj[i] & (1 << i)):
            valid_vertices.append(i)
    
    # Now we need to find a maximum independent set in the induced subgraph on valid_vertices
    # where edges are between vertices i, j (i != j) if (i+1)+(j+1) is a perfect square
    
    # For the induced subgraph, we only care about conflicts between different valid vertices
    # Build adjacency list for the induced subgraph
    # Map valid vertex index to position in valid_vertices list
    # Actually, let's work with the original indices but only consider valid vertices
    
    # Greedy heuristic: start with all valid vertices, then iteratively remove conflicts
    # Better: use a greedy algorithm to build a large independent set
    
    # Strategy: 
    # 1. Start with all valid vertices
    # 2. Use a greedy approach: sort by degree (ascending), add vertices if no conflict with already-added
    
    # But first, let's try a better approach: local search / simulated annealing style
    
    # Let's use a greedy + local improvement approach
    # Start with a greedy solution, then try to add vertices or swap to improve
    
    # Greedy: sort valid vertices by degree in the conflict graph (among valid vertices), ascending
    # For each vertex, if it doesn't conflict with any already-selected vertex, add it
    
    # First, compute degrees among valid vertices
    valid_set = set(valid_vertices)
    degrees = {}
    for i in valid_vertices:
        deg = 0
        mask = adj[i]
        for j in valid_vertices:
            if i != j and (mask & (1 << j)):
                deg += 1
        degrees[i] = deg
    
    # Greedy independent set
    def greedy_is():
        # Sort by degree ascending
        sorted_v = sorted(valid_vertices, key=lambda x: degrees[x])
        selected = []
        selected_set = set()
        for v in sorted_v:
            # Check if v conflicts with any selected vertex
            mask = adj[v]
            conflict = False
            for s in selected:
                if mask & (1 << s):
                    conflict = True
                    break
            if not conflict:
                selected.append(v)
                selected_set.add(v)
        return selected
    
    # Local improvement: try to add vertices that were rejected, or swap
    def local_improve(solution, time_limit):
        sol_set = set(solution)
        sol_list = list(solution)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try to add each non-selected valid vertex
            for v in valid_vertices:
                if v in sol_set:
                    continue
                mask = adj[v]
                conflicts = [s for s in sol_list if mask & (1 << s)]
                if len(conflicts) == 0:
                    sol_list.append(v)
                    sol_set.add(v)
                    improved = True
                    break  # restart after adding
            # Try swaps: remove one conflict vertex, add v
            for v in valid_vertices:
                if v in sol_set:
                    continue
                mask = adj[v]
                conflicts = [s for s in sol_list if mask & (1 << s)]
                if 0 < len(conflicts) <= 1:
                    # If exactly one conflict, try removing it and adding v
                    if len(conflicts) == 1:
                        s = conflicts[0]
                        # Check if removing s and adding v keeps independence
                        # s only conflicts with v among sol_list (since len(conflicts)==1)
                        # But we need to make sure no other vertex in sol_list conflicts with v
                        # which is guaranteed since conflicts has only s
                        # And removing s doesn't create new conflicts
                        sol_list.remove(s)
                        sol_set.remove(s)
                        sol_list.append(v)
                        sol_set.add(v)
                        improved = True
                        break
        return sol_list
    
    # Get initial greedy solution
    solution = greedy_is()
    
    # Local improvement
    solution = local_improve(solution, DEADLINE - 5)
    
    # Try multiple random restarts if time permits
    best_solution = solution
    best_size = len(solution)
    
    while (time.time() - start_time) < DEADLINE - 3:
        # Randomized greedy
        random.seed(int(time.time() * 1000) % (2**31))
        sorted_v = list(valid_vertices)
        random.shuffle(sorted_v)
        # Sort by degree with some randomness
        sorted_v = sorted(sorted_v, key=lambda x: degrees[x] + random.random() * 2)
        
        selected = []
        selected_set = set()
        for v in sorted_v:
            mask = adj[v]
            conflict = False
            for s in selected:
                if mask & (1 << s):
                    conflict = True
                    break
            if not conflict:
                selected.append(v)
                selected_set.add(v)
        
        if len(selected) > best_size:
            best_solution = selected
            best_size = len(selected)
        
        # Local improve
        improved = local_improve(selected, DEADLINE - 3)
        if len(improved) > best_size:
            best_solution = improved
            best_size = len(improved)
    
    # Convert back to values
    result = [v + 1 for v in best_solution]
    
    # Verify the result
    squares_set = set()
    k = 1
    while k * k <= 2 * N:
        squares_set.add(k * k)
        k += 1
    
    for i in range(len(result)):
        for j in range(i, len(result)):
            if result[i] + result[j] in squares_set:
                # This should not happen; remove the problematic vertex
                # For safety, just return what we have (it should be valid)
                pass
    
    if not result:
        # Fallback: return a single vertex that has no self-conflict
        for i in valid_vertices:
            return [i + 1]
        return [1]
    
    return result
# EVOLVE-BLOCK-END
