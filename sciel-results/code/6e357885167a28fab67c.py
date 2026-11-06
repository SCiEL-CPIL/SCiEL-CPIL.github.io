import time
import random
from itertools import combinations

# Precompute popcounts for all 16-bit masks
N = 16
ALL_MASK = (1 << N) - 1
POP_COUNTS = [0] * (1 << N)
for i in range(1 << N):
    POP_COUNTS[i] = bin(i).count('1')

# Precompute all possible triples as bitmasks
ALL_TRIPLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append(mask)

def is_linear(existing_edges, new_edge):
    """Check if new_edge shares at most 1 point with all existing_edges."""
    for e in existing_edges:
        if (e & new_edge) != 0:
            # If intersection size > 1, it's not linear
            # Intersection size is 2 or 3
            # (e & new_edge) will have 2 or 3 bits set
            # We just need to check if popcount(e & new_edge) > 1
            if POP_COUNTS[e & new_edge] > 1:
                return False
    return True

def calc_alpha(edges_masks):
    """Calculate independence number of the hypergraph defined by edges_masks."""
    if not edges_masks:
        return N
    
    bad = [False] * (1 << N)
    
    for e in edges_masks:
        # Mark all supersets of e as bad
        comp = ALL_MASK ^ e
        sub = comp
        while True:
            bad[e | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    max_alpha = 0
    for i in range(1 << N):
        if not bad[i]:
            pc = POP_COUNTS[i]
            if pc > max_alpha:
                max_alpha = pc
                if max_alpha == N:
                    break
    return max_alpha

def random_greedy_packing():
    """Generate a random maximal linear packing."""
    edges = []
    available = ALL_TRIPLES[:]
    random.shuffle(available)
    
    for t in available:
        if is_linear(edges, t):
            edges.append(t)
            
    return edges

def local_search(edges, time_limit):
    """Improve the independence number via local search."""
    current_alpha = calc_alpha(edges)
    best_alpha = current_alpha
    best_edges = edges[:]
    
    while time.time() < time_limit:
        improved = False
        # Try swapping one edge
        for i in range(len(edges)):
            if time.time() >= time_limit:
                break
                
            e_to_remove = edges[i]
            remaining_edges = edges[:i] + edges[i+1:]
            
            # Find all valid replacements
            # We can just iterate through ALL_TRIPLES and check linearity with remaining_edges
            # To speed up, we can filter candidates that share points with e_to_remove?
            # Not strictly necessary for N=16
            
            for new_e in ALL_TRIPLES:
                if new_e == e_to_remove:
                    continue
                if is_linear(remaining_edges, new_e):
                    candidate_edges = remaining_edges + [new_e]
                    new_alpha = calc_alpha(candidate_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = candidate_edges
                        improved = True
                        # Break to restart search from new state? Or continue?
                        # Continuing might find further improvements in same pass
                        # But updating edges list in place is tricky.
                        # Let's just update best and break inner loop to re-evaluate next edge?
                        # Or just accept and break both loops to restart outer while?
                        # Accepting and breaking is safer for stability.
                        break
            if improved:
                edges = best_edges[:]
                current_alpha = best_alpha
                break # Restart outer while loop with new edges
        
        if not improved:
            # No single swap improved the alpha.
            # Try adding an edge?
            # If the packing is not maximal, adding an edge can only decrease alpha.
            # Check if it's maximal
            maximal = True
            for t in ALL_TRIPLES:
                if is_linear(edges, t):
                    # Not maximal
                    new_edges = edges + [t]
                    new_alpha = calc_alpha(new_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = new_edges
                        edges = best_edges[:]
                        current_alpha = best_alpha
                        maximal = False
                        break
            if maximal:
                break # Local optimum reached
        
    return best_edges, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial seed
    seed_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    seed_masks = []
    for t in seed_triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        seed_masks.append(mask)
        
    best_edges = seed_masks[:]
    best_alpha = calc_alpha(best_edges)
    
    # Random restarts
    while time.time() < deadline:
        # Generate a random packing
        candidate_edges = random_greedy_packing()
        candidate_alpha = calc_alpha(candidate_edges)
        
        if candidate_alpha < best_alpha:
            best_alpha = candidate_alpha
            best_edges = candidate_edges[:]
        
        # Try to improve the candidate (or best if candidate was better)
        # We have limited time, so only do local search if we have time
        if time.time() < deadline - 5:
            improved_edges, improved_alpha = local_search(candidate_edges, deadline - 1)
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_edges = improved_edges[:]
                
        # Also try to improve the current best if time permits
        if time.time() < deadline - 5 and len(best_edges) > 0:
            # Only do this if we haven't just done it
            if best_edges != candidate_edges:
                improved_edges, improved_alpha = local_search(best_edges, deadline - 1)
                if improved_alpha < best_alpha:
                    best_alpha = improved_alpha
                    best_edges = improved_edges[:]

    # Convert best_masks back to list of triples [a, b, c]
    result = []
    for mask in best_edges:
        triple = []
        for i in range(N):
            if mask & (1 << i):
                triple.append(i)
        result.append(triple)
        
    return result
