import math
import time
import random
import sys

# Increase recursion limit for deep B&B
sys.setrecursionlimit(10000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    n = 256
    max_sum = 2 * n
    
    # 1. Preprocessing: Identify valid nodes and build conflict graph
    # A node 'a' is invalid if 2*a is a perfect square (diagonal constraint a+a)
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    valid_numbers = []
    val_to_idx = {}
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)
            val_to_idx[a] = len(valid_numbers) - 1
            
    num_valid = len(valid_numbers)
    
    # Build adjacency list and bitmasks
    # adjacency[i] is a bitmask of all j such that i and j conflict
    # Conflict if a + b is a perfect square
    adjacency = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b > a and b in val_to_idx:
                j = val_to_idx[b]
                adjacency[i] |= (1 << j)
                adjacency[j] |= (1 << i)
    
    # 2. Initial Lower Bound (Greedy)
    # We need a good starting solution to prune the B&B effectively.
    # Use a greedy heuristic: pick vertex with min degree in remaining graph, add to set, remove neighbors.
    def greedy_initial():
        remaining = (1 << num_valid) - 1
        solution = 0
        # Pre-compute static degrees for initial sorting, but we'll update dynamically
        # For simplicity and speed in initialization, we use a few random restarts of greedy
        best_greedy = 0
        
        # Static degree sort
        static_degrees = [bin(adjacency[i]).count('1') for i in range(num_valid)]
        
        # Try a few different orderings
        for trial in range(10):
            if time.time() - start_time > deadline - 5:
                break
            rem = (1 << num_valid) - 1
            sol = 0
            # Create a list of nodes, sort by static degree + random noise
            indices = list(range(num_valid))
            if trial == 0:
                indices.sort(key=lambda x: static_degrees[x])
            else:
                random.seed(trial)
                random.shuffle(indices)
            
            for v in indices:
                if not (rem & (1 << v)):
                    continue
                # Check if v is still valid (not removed as neighbor)
                # Actually, 'rem' contains nodes not yet decided. 
                # If v is in rem, it hasn't been removed by a neighbor selection.
                # Wait, standard greedy: 
                # 1. Pick v
                # 2. Add v to solution
                # 3. Remove v and its neighbors from 'rem'
                
                sol |= (1 << v)
                # Remove v and neighbors from rem
                rem &= ~(adjacency[v] | (1 << v))
                
            if bin(sol).count('1') > bin(best_greedy).count('1'):
                best_greedy = sol
                
        return best_greedy

    best_solution = greedy_initial()
    best_size = bin(best_solution).count('1')
    
    # 3. Branch and Bound
    # State: (candidate_mask, current_set_mask)
    # candidate_mask: bits that are still available to be added
    # current_set_mask: bits already in the independent set
    
    # Upper Bound:
    # 1. Trivial: popcount(candidate_mask) + popcount(current_set_mask)
    # 2. Greedy Coloring Bound:
    #    The conflict graph induced by candidate_mask can be colored.
    #    The size of the maximum independent set is at most the number of colors needed?
    #    No, MIS <= Clique Cover. 
    #    Actually, a standard upper bound for MIS is the size of the maximum clique in the complement?
    #    Let's use the "Greedy Coloring" upper bound for the *conflict* graph.
    #    If we can color the conflict graph with K colors, then the independent set can pick at most 1 from each color class?
    #    No, that's for the Clique problem (MIS in complement).
    #    For MIS in G: 
    #    An upper bound is given by the greedy coloring of the *conflict* graph?
    #    Actually, if we color the conflict graph, each color class is an independent set in G.
    #    This doesn't give an upper bound on the size of a *single* independent set.
    #    
    #    Correct Upper Bound for MIS:
    #    Let H be the conflict graph. We want MIS in H.
    #    We can use the "Clique Cover" of H? No, MIS in H is Clique in Complement(H).
    #    So we need an upper bound on the size of a Clique in Complement(H).
    #    A standard upper bound for Max Clique is the Greedy Coloring of the graph (here Complement(H)).
    #    However, building Complement(H) is expensive (dense).
    #    
    #    Alternative Upper Bound:
    #    Greedy Coloring of the *Conflict Graph H* gives a partition of vertices into independent sets of H.
    #    This doesn't help bound the size of a single independent set.
    #    
    #    Let's stick to a simpler, effective bound for sparse graphs:
    #    Bound = popcount(candidate_mask). 
    #    This is weak.
    #    
    #    Better Bound: 
    #    If we select a vertex v, we can either include it or not.
    #    If we include v, we must exclude its neighbors.
    #    
    #    Let's use a "Degree-based" bound.
    #    Or, simpler: just use the trivial bound but branch smartly.
    #    With 256 nodes, trivial bound might be too weak if the greedy solution is small.
    #    But the graph is quite sparse (degree ~10-20).
    #    
    #    Let's try a recursive B&B with the following:
    #    1. If popcount(candidate) + popcount(current) <= best_size: prune.
    #    2. If candidate is empty: update best.
    #    3. Pick a vertex v from candidate. 
    #       Strategy: Pick v with highest degree in the *induced* subgraph of candidate?
    #       Or lowest? 
    #       High degree vertices constrain more, so branching on them might prune faster.
    #    4. Branch 1: Include v. New candidate = candidate & ~(adjacency[v] | (1<<v)). New current = current | (1<<v).
    #    5. Branch 2: Exclude v. New candidate = candidate & ~(1<<v). New current = current.
    
    # To speed up "pick vertex with highest degree in induced subgraph", 
    # calculating degrees dynamically is O(N^2) or O(E). 
    # With N=256, it's manageable if done carefully.
    
    def solve_bnb(candidate_mask, current_mask, current_size):
        nonlocal best_solution, best_size, start_time, deadline
        
        if time.time() - start_time > deadline - 2:
            return
        
        # Prune 1: Trivial bound
        cand_count = bin(candidate_mask).count('1')
        if current_size + cand_count <= best_size:
            return
            
        if candidate_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_solution = current_mask
            return
        
        # Prune 2: If we can't possibly beat best_size, return
        # (Covered by Prune 1)
        
        # Select a pivot vertex
        # We want a vertex that appears in many conflicts within the candidate set.
        # This maximizes the reduction in candidate set when we include it.
        # Let's pick the vertex with the maximum degree in the induced subgraph.
        
        # Optimization: If we just pick the first set bit, it might be slow.
        # Let's iterate through candidates and find max degree.
        # N=256, so this loop is at most 256 iterations.
        # Inside, we count bits in (adjacency[v] & candidate_mask).
        
        max_deg = -1
        pivot = -1
        # We can also use a heuristic: if there's a vertex with 0 degree in candidate, 
        # we MUST include it (it doesn't conflict with anyone in candidate).
        # If we include it, we remove it from candidate but don't remove others.
        
        # Let's find vertices with 0 degree in candidate first.
        # If any, we can force-include them all.
        forced = 0
        for v in range(num_valid):
            if candidate_mask & (1 << v):
                deg = bin(adjacency[v] & candidate_mask).count('1')
                if deg == 0:
                    forced |= (1 << v)
        
        if forced:
            # Include all forced vertices
            current_mask |= forced
            current_size += bin(forced).count('1')
            # Remove them from candidate
            candidate_mask &= ~forced
            # They don't conflict with others in candidate, so we don't need to remove neighbors
            # Recurse immediately
            return solve_bnb(candidate_mask, current_mask, current_size)
            
        # Now find max degree vertex
        for v in range(num_valid):
            if candidate_mask & (1 << v):
                deg = bin(adjacency[v] & candidate_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                    
        if pivot == -1:
            return # Should not happen if candidate_mask != 0
            
        # Branch 1: Include pivot
        # New candidate: remove pivot and its neighbors in candidate
        new_cand_include = candidate_mask & ~(adjacency[pivot] | (1 << pivot))
        new_cur_include = current_mask | (1 << pivot)
        new_size_include = current_size + 1
        
        # Branch 2: Exclude pivot
        new_cand_exclude = candidate_mask & ~(1 << pivot)
        new_cur_exclude = current_mask
        new_size_exclude = current_size
        
        # Heuristic: Try the branch that seems more promising first.
        # Including pivot removes many nodes, so the bound might drop faster.
        # But excluding pivot keeps more nodes.
        # Usually, "Include" is tried first in MIS to find a large solution quickly.
        
        # Check bound for Include branch
        if new_size_include + bin(new_cand_include).count('1') > best_size:
            solve_bnb(new_cand_include, new_cur_include, new_size_include)
            
        # Check bound for Exclude branch
        if new_size_exclude + bin(new_cand_exclude).count('1') > best_size:
            solve_bnb(new_cand_exclude, new_cur_exclude, new_size_exclude)

    # Run B&B
    # Initial candidate is all valid nodes
    initial_candidate = (1 << num_valid) - 1
    # We already have a greedy solution, so B&B will prune heavily
    
    # To prevent stack overflow or excessive time, we rely on the time check inside.
    # However, pure recursion might be slow in Python.
    # Let's add a timeout check in the main loop if we were iterating, but here it's recursive.
    # The time check `if time.time() - start_time > deadline - 2: return` handles it.
    
    try:
        solve_bnb(initial_candidate, 0, 0)
    except RecursionError:
        pass # If it hits recursion limit, we keep the best found so far

    # 4. Local Search Polish (Optional, if time remains)
    # If B&B didn't find the optimum, a quick local search on the result might help.
    # But B&B is exact if it finishes. If it times out, it's a lower bound.
    # Let's do a quick "add if fits" pass on the result.
    
    result_mask = best_solution
    # Try to add any node that fits
    changed = True
    while changed:
        changed = False
        for v in range(num_valid):
            if not (result_mask & (1 << v)):
                if not (adjacency[v] & result_mask):
                    result_mask |= (1 << v)
                    changed = True
                    if bin(result_mask).count('1') > best_size:
                        best_size = bin(result_mask).count('1')
                        best_solution = result_mask

    # Convert mask to list
    result = [valid_numbers[i] for i in range(num_valid) if (best_solution & (1 << i))]
    
    if not result:
        return [1]
        
    return result
# EVOLVE-BLOCK-END
