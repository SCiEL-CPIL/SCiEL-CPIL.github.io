import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    # Returns True if there exists a 2-coloring such that no edge is monochromatic.
    # "No proper 2-colouring" in the task means "No coloring leaves every edge with both colours".
    # So if is_2_colorable returns False, the hypergraph has NO proper 2-colouring (it is non-2-colorable).
    
    num_masks = 1 << n
    # Precompute vertex bits for edges?
    # For speed, represent edges as tuples of vertices.
    
    for mask in range(num_masks):
        valid = True
        for e in edges:
            # Check if e is monochromatic
            c0 = (mask >> e[0]) & 1
            # If any vertex has different color, it's not monochromatic
            # If all have same color, it is monochromatic -> invalid coloring
            is_mono = True
            for v in e[1:]:
                if ((mask >> v) & 1) != c0:
                    is_mono = False
                    break
            if is_mono:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, leaving 10s margin for safety
    
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Shuffle to avoid bias
    random.seed(42)
    
    best_edges = all_edges[:]
    best_count = len(all_edges)
    
    while time.time() < deadline:
        trial_edges = all_edges[:]
        random.shuffle(trial_edges)
        
        # Maintain the set of active edges
        active = set(tuple(e) for e in all_edges)
        
        for e in trial_edges:
            # Time check at the start of the iteration
            if time.time() > deadline - 5:
                break
            
            e_tuple = tuple(e)
            if e_tuple not in active:
                continue
            
            # Remove e
            active.remove(e_tuple)
            
            # Check if the remaining graph is non-2-colorable
            # Convert to list for the checker
            active_list = [list(t) for t in active]
            
            if not is_2_colorable(n, active_list):
                # Still non-2-colorable, so e was redundant. Keep it removed.
                pass
            else:
                # Became 2-colorable, so e was necessary. Restore it.
                active.add(e_tuple)
        
        # End of pass
        # Verify the result of this pass is non-2-colorable
        # (This should be guaranteed by the logic, but safety first)
        if active:
            final_list = [list(t) for t in active]
            if not is_2_colorable(n, final_list):
                if len(active) < best_count:
                    best_count = len(active)
                    best_edges = final_list
                    
                    # Early exit if we hit the known best or lower
                    if best_count <= 51:
                        break
            else:
                # If it is 2-colorable (should not happen), revert to previous best
                pass
        else:
            # If active is empty, it's 2-colorable (vacuously or not, but empty is colorable)
            # Actually empty hypergraph is 2-colorable.
            # So this state is invalid for our goal.
            pass

    # Final safety check: if best_edges is 2-colorable, we have a problem.
    # This can only happen if the initial all_edges was 2-colorable (it's not)
    # or if a logic error occurred.
    # If best_edges is 2-colorable, we should try to return a valid one.
    # But we don't have a hardcoded one.
    # However, the logic above ensures best_edges is only updated if it's non-2-colorable.
    # So best_edges should be valid.
    
    return {'n': n, 'edges': best_edges}
