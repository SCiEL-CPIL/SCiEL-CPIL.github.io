import time
import random
import sys

N = 128
DEADLINE_MARGIN = 5.0  # seconds to leave before 60s limit

def _build_edges():
    """
    Build the set of forbidden configurations.
    A configuration is a set of values {a, b, c, a+b, a+c, b+c} 
    where a,b,c are distinct and all values are in 1..N.
    We store these as frozensets of the involved vertices.
    Note: The constraint is that NOT ALL of these vertices can be in the set.
    """
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 > N or s2 > N or s3 > N:
                    continue
                
                # The forbidden set of vertices is {a, b, c, s1, s2, s3}
                # If all these are in our selected set, it's a violation.
                # We can deduplicate the values in the edge itself, 
                # but the condition is "all values in the edge are selected".
                # If s1 == a, then the edge is {a, b, c, s2, s3}. 
                # If all are selected, it's still a violation.
                edge = frozenset([a, b, c, s1, s2, s3])
                edges.append(edge)
    return edges

def construct():
    t_start = time.time()
    deadline = t_start + 60.0 - DEADLINE_MARGIN

    edges = _build_edges()
    
    # Map vertex -> list of edge indices containing it
    vert_to_edges = {v: [] for v in range(1, N + 1)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)

    # Helper to check if adding v is valid given current edge_counts
    # edge_counts[ei] = number of vertices in edges[ei] that are currently selected
    def _can_add(v, edge_counts):
        for ei in vert_to_edges[v]:
            # If all other vertices in this edge are already selected, 
            # adding v completes the edge -> violation.
            # len(edges[ei]) is the size of the unique vertex set in the edge.
            # We need to check if edge_counts[ei] == len(edges[ei]) - 1.
            if edge_counts[ei] == len(edges[ei]) - 1:
                return False
        return True

    def _update_counts(v, delta, edge_counts):
        for ei in vert_to_edges[v]:
            edge_counts[ei] += delta

    # Initial solution: Upper half is always safe (sums > N)
    # Let's start with a random mix or just the upper half.
    # Upper half size is 64.
    # We can try to fill the lower half greedily.
    
    best_set = set(range(65, N + 1))
    best_size = len(best_set)
    
    # Initialize edge counts for the initial set
    edge_counts = [0] * len(edges)
    for v in best_set:
        _update_counts(v, 1, edge_counts)

    # List of vertices to try adding (lower half)
    candidates = list(range(1, 65))
    
    # Greedy Phase 1: Add as many as possible
    # Sort candidates by some heuristic? 
    # Smaller numbers are more likely to be part of sums, so maybe add larger lower-half numbers first?
    # Or just randomize to avoid bias.
    random.seed(42)
    
    # Let's try a few different greedy orderings
    orderings = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        candidates[:], # random
    ]
    random.shuffle(orderings[2])

    for ordering in orderings:
        # Reset to base state
        cur_set = set(range(65, N + 1))
        cur_edge_counts = [0] * len(edges)
        for v in cur_set:
            _update_counts(v, 1, cur_edge_counts)
            
        cur_size = len(cur_set)
        
        for v in ordering:
            if v in cur_set:
                continue
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                cur_size += 1

        if cur_size > best_size:
            best_size = cur_size
            best_set = cur_set.copy()
            # Update best edge counts? We will recalculate in local search anyway.

    # Local Search Phase
    # We will operate on the best set found so far.
    # We need to be careful with time.
    
    # Recalculate edge counts for best_set
    best_edge_counts = [0] * len(edges)
    for v in best_set:
        _update_counts(v, 1, best_edge_counts)
        
    cur_set = best_set
    cur_edge_counts = best_edge_counts
    cur_size = len(cur_set)
    
    # Pre-compute "potential" or just use random swaps/removals
    # Strategy: 
    # 1. Try to add any missing vertex.
    # 2. If stuck, remove a vertex that is "blocking" many others, then try to add others.
    
    # Identify missing vertices
    all_vertices = set(range(1, N + 1))
    
    # To speed up, we can precompute for each missing vertex v, 
    # which edges are "almost full" (count == len-1) and contain v.
    # But checking _can_add is O(deg(v)). deg(v) is smallish.
    
    # Let's do a simple hill climbing with restarts or perturbations.
    
    for iteration in range(1000): # Max iterations, but time-limited
        if time.time() > deadline:
            break
            
        improved = False
        
        # Try to add any valid vertex
        # Randomize the order of missing vertices to avoid local minima bias
        missing = [v for v in all_vertices if v not in cur_set]
        random.shuffle(missing)
        
        for v in missing:
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                cur_size += 1
                improved = True
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = cur_set.copy()
                    # best_edge_counts = cur_edge_counts.copy() # Can't just copy if we mutate later, but we don't need best_edge_counts unless we restore.
        
        if improved:
            continue # Try to add more
            
        # If no improvement, try to remove one vertex and see if we can add more.
        # Which vertex to remove?
        # Heuristic: Remove a vertex that is in many "tight" edges.
        # Or just pick a random vertex from the current set.
        
        # Let's pick a random vertex from the current set to remove.
        # To be smarter, we can pick one from the lower half, as upper half is "safe" base.
        # Actually, removing upper half might allow adding many lower half.
        
        # Let's try removing a random element from the current set.
        if not cur_set:
            break
            
        rem_v = random.choice(list(cur_set))
        
        # Save state for rollback if it doesn't help
        # Since we want to maximize, we only commit if we find a better set.
        # But we are doing in-place. Let's snapshot.
        old_set = cur_set.copy()
        old_edge_counts = cur_edge_counts.copy()
        old_size = cur_size
        
        cur_set.remove(rem_v)
        _update_counts(rem_v, -1, cur_edge_counts)
        cur_size -= 1
        
        # Now try to add as many as possible again
        missing_after_rem = [v for v in all_vertices if v not in cur_set]
        random.shuffle(missing_after_rem)
        
        local_improved = False
        for v in missing_after_rem:
            if v == rem_v:
                continue
            if _can_add(v, cur_edge_counts):
                cur_set.add(v)
                _update_counts(v, 1, cur_edge_counts)
                cur_size += 1
                local_improved = True
                
        if cur_size > old_size:
            # We found a better configuration
            best_size = cur_size
            best_set = cur_set.copy()
            # best_edge_counts = cur_edge_counts.copy()
        else:
            # Rollback
            cur_set = old_set
            cur_edge_counts = old_edge_counts
            cur_size = old_size
            
            # If we are stuck, maybe do a random restart with a different seed?
            # Or just continue with random removals.
            
    # Final Polish: Try to add any remaining valid vertices in a deterministic pass
    # This ensures we don't miss easy additions due to random shuffling stopping early
    # (though the loop above shuffles and tries all missing, so it should be exhaustive for additions)
    # Wait, the loop above breaks if `improved` is True and restarts the loop.
    # If `improved` is False, it does a removal.
    # So it effectively keeps adding until no more can be added.
    
    # One more pass to be sure
    missing_final = [v for v in all_vertices if v not in best_set]
    # We need edge counts for best_set
    be = [0] * len(edges)
    for v in best_set:
        _update_counts(v, 1, be)
        
    for v in missing_final:
        if _can_add(v, be):
            best_set.add(v)
            _update_counts(v, 1, be)
            
    result = [int(v) for v in sorted(best_set)]
    return result

if __name__ == "__main__":
    # Quick self-test
    print("Starting...")
    res = construct()
    print(f"Size: {len(res)}")
