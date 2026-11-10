"""Find a small 5-uniform hypergraph without Property B (no proper 2-colouring)."""
import time
import random
import math
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Fast check for 2-colorability using bitmask backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True

    # Precompute bitmasks for edges
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    # Precompute vertex lists for edges for propagation
    edge_verts = [list(e) for e in edges]

    # We use a recursive search.
    # State: mask0, mask1
    # If a vertex is in mask0, it's color 0.
    # If a vertex is in mask1, it's color 1.
    # Unassigned vertices are in neither.

    # To speed up, we can order vertices by degree or just use 0..n-1.
    # Let's use a simple recursion with strong pruning.

    # Precompute for each edge, the list of vertices
    # And maybe a dictionary from vertex to list of edges containing it?
    # That would help propagation.

    # For now, simple backtracking with pruning.
    # Pruning 1: If an edge is monochromatic, fail.
    # Pruning 2: If an edge has 4 vertices of one color and 1 unassigned, force the 5th.

    # We'll implement a helper to check if a state is valid and propagate.
    
    def check_state(mask0, mask1):
        # Check for monochromatic edges
        for em in edge_masks:
            # If all vertices in edge are in mask0
            if (em & ~mask0) == 0:
                return False
            # If all vertices in edge are in mask1
            if (em & ~mask1) == 0:
                return False
        return True

    def propagate(mask0, mask1):
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for em, verts in zip(edge_masks, edge_verts):
                # Count colors in this edge
                c0 = bin(em & mask0).count('1')
                c1 = bin(em & mask1).count('1')
                uncolored = [v for v in verts if (mask0 & (1<<v)) == 0 and (mask1 & (1<<v)) == 0]
                
                if not uncolored:
                    # Fully colored
                    if c0 == 0 or c1 == 0:
                        return False, False, False
                    continue
                
                # If c0 == 4 and c1 == 0, the uncolored one MUST be 1
                if c0 == 4 and c1 == 0:
                    # There must be exactly 1 uncolored vertex
                    if len(uncolored) == 1:
                        v = uncolored[0]
                        if (mask1 & (1<<v)) != 0:
                            return False, False, False # Already colored 1? No, we said uncolored.
                        # Wait, if it's uncolored, it's not in mask1.
                        # So we force it to be 1.
                        mask1 |= (1 << v)
                        changed = True
                    elif len(uncolored) > 1:
                        # This shouldn't happen if c0=4 in a 5-edge, unless c1>0?
                        # If c0=4, c1=0, sum=4. 1 vertex left.
                        pass
                elif c1 == 4 and c0 == 0:
                    if len(uncolored) == 1:
                        v = uncolored[0]
                        mask0 |= (1 << v)
                        changed = True
        
        # Check for monochromatic again after propagation
        if not check_state(mask0, mask1):
            return False, False, False
            
        return True, mask0, mask1

    # Recursive backtracking
    # We pick a vertex to assign.
    # Heuristic: Pick vertex involved in most "constrained" edges?
    # Simpler: Pick first unassigned vertex.
    
    def backtrack(mask0, mask1):
        # Find an unassigned vertex
        unassigned_mask = ((1 << n) - 1) & ~mask0 & ~mask1
        if unassigned_mask == 0:
            # All assigned
            return check_state(mask0, mask1)
        
        # Pick a vertex. Let's pick the one with highest degree in remaining unassigned?
        # Or just the lowest index for determinism.
        # For speed, let's pick one.
        v = (unassigned_mask & -unassigned_mask).bit_length() - 1
        
        # Try color 0
        new_mask0, new_mask1 = mask0 | (1 << v), mask1
        ok, m0, m1 = propagate(new_mask0, new_mask1)
        if ok:
            if backtrack(m0, m1):
                return True
        
        # Try color 1
        new_mask0, new_mask1 = mask0, mask1 | (1 << v)
        ok, m0, m1 = propagate(new_mask0, new_mask1)
        if ok:
            if backtrack(m0, m1):
                return True
        
        return False

    return backtrack(0, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # We will use Simulated Annealing to minimize edges.
    # Start with a known non-2-colorable hypergraph.
    # Complete hypergraph on 9 vertices is non-2-colorable.
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Verify it's non-2-colorable
    if _is_2_colorable(n, current_edges):
        # Fallback, though this shouldn't happen
        n = 10
        all_edges = [list(e) for e in combinations(range(n), 5)]
        current_edges = all_edges[:]

    best_edges = current_edges[:]
    best_n = n
    best_cost = len(best_edges)

    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.995
    min_T = 0.01
    
    # Candidate pool: all possible edges on n vertices
    # We can also try adding vertices? 
    # The task allows n up to 26.
    # Adding vertices increases the space of edges, but might allow removing more edges?
    # Usually, fewer vertices is better for compactness, but the constraint is on edge count.
    # Let's stick to n=9, 10, 11 for now.
    
    # Precompute all possible edges for n=9, 10, 11
    # To allow "add edge" moves, we need a pool.
    
    # Let's run SA for a fixed amount of time.
    
    while time.time() < deadline - 10:
        if T < min_T:
            break
            
        # Generate a move
        # 1. Remove a random edge
        # 2. Add a random edge (from a random vertex set if we change n, or from pool)
        
        # For simplicity, let's just remove and add within the current n.
        # Pool of edges not in current_edges
        current_edge_set = set(tuple(e) for e in current_edges)
        all_possible = [tuple(e) for e in combinations(range(n), 5)]
        
        move_type = random.random()
        
        if move_type < 0.7 and len(current_edges) > 5:
            # Remove an edge
            idx = random.randint(0, len(current_edges) - 1)
            new_edges = current_edges[:idx] + current_edges[idx+1:]
        else:
            # Add an edge
            # Pick a random edge not in current set
            # This is slow if we do it naively.
            # Let's pick a random candidate from all_possible
            # If it's already there, skip (or pick another)
            added = False
            for _ in range(10): # Try a few times
                candidate = random.choice(all_possible)
                if candidate not in current_edge_set:
                    new_edges = current_edges + [list(candidate)]
                    added = True
                    break
            if not added:
                # Continue if we can't add
                T *= cooling_rate
                continue

        # Check validity
        # If new_edges is 2-colorable, it's invalid.
        # We want to minimize edges, so valid states have cost = len(edges).
        # Invalid states have cost = infinity.
        
        if _is_2_colorable(n, new_edges):
            # Invalid
            new_cost = float('inf')
        else:
            new_cost = len(new_edges)

        old_cost = len(current_edges)
        
        # Acceptance
        if new_cost < old_cost:
            current_edges = new_edges
            if new_cost < best_cost:
                best_cost = new_cost
                best_edges = new_edges[:]
                best_n = n
        else:
            # Calculate acceptance probability
            # delta_cost = new_cost - old_cost
            # If new_cost is inf, delta is inf, exp(-inf/T) = 0.
            if new_cost == float('inf'):
                accept = False
            else:
                delta = new_cost - old_cost
                if delta == 0:
                    accept = True
                else:
                    accept_prob = math.exp(-delta / T)
                    accept = random.random() < accept_prob
            
            if accept:
                current_edges = new_edges

        T *= cooling_rate

    # Final check and return
    if _is_2_colorable(best_n, best_edges):
        # Fallback to a known valid one if somehow we lost it
        # But we only update best if valid.
        # Just in case, return complete K_9
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': best_edges}
