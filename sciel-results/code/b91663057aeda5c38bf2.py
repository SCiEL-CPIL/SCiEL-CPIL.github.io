import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Optimized 2-colorability check using backtracking and unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    # Map vertices to indices 0..n-1
    # Pre-compute for each vertex, the list of edges containing it
    # This helps in propagation? Maybe not directly for simple backtracking.
    
    # For speed, we represent the state as a tuple of colors for 0..n-1.
    # But we do recursive search.
    
    # Heuristic: Order vertices by degree?
    # Let's stick to a simple but effective backtracking.
    
    # Convert edges to sets for faster lookup? Or just lists.
    # Let's use a bitmask approach for the current assignment?
    # n <= 26, so we can fit in a 32-bit int.
    # 0: unassigned, 1: color 0, 2: color 1?
    # Actually, let's use two bitmasks: mask0, mask1.
    
    # However, Python recursion with bitmasks might be slower than lists for small n.
    # Let's use a list of -1, 0, 1.
    
    assignment = [-1] * n
    
    # Sort edges to help with pruning?
    # We want to detect conflicts early.
    
    def propagate(assignment):
        """
        Perform unit propagation.
        Returns False if contradiction found.
        Returns True if no contradiction (and assignment is updated).
        """
        changed = True
        while changed:
            changed = False
            for e in edges:
                # Count colors in e
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if assignment[v] == 0:
                        c0 += 1
                    elif assignment[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # If 4 are 0, the remaining must be 1
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if assignment[v] == 1:
                        return False
                    if assignment[v] == -1:
                        assignment[v] = 1
                        changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if assignment[v] == 0:
                        return False
                    if assignment[v] == -1:
                        assignment[v] = 0
                        changed = True
                elif c0 == 5:
                    return False
                elif c1 == 5:
                    return False
        return True

    def backtrack(idx):
        if idx == n:
            return True
        
        # Try 0
        assignment[idx] = 0
        if propagate(assignment):
            if backtrack(idx + 1):
                return True
        
        # Try 1
        assignment[idx] = 1
        if propagate(assignment):
            if backtrack(idx + 1):
                return True
        
        assignment[idx] = -1
        return False

    return backtrack(0)

def construct():
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try different n and random seeds
    # We know m(5) <= 51. We aim to find < 51.
    # Smaller n might have smaller minimal edges?
    # For n=9, max edges 126. Minimal non-colorable is likely small.
    # For n=10, max edges 252.
    
    # Let's search n from 9 to 12.
    for n in range(9, 13):
        all_possible = list(combinations(range(n), 5))
        
        # Try several random instances
        for seed in range(50):
            random.seed(seed)
            # Start with a random subset of edges
            # Density: start with 30-50 edges
            k = random.randint(30, 60)
            if k > len(all_possible):
                k = len(all_possible)
            
            current_edges = random.sample(all_possible, k)
            
            # Check if non-colorable
            if not is_2_colorable(n, current_edges):
                # It is non-colorable. Now minimize.
                # Greedy removal
                edges_to_remove = list(current_edges)
                random.shuffle(edges_to_remove)
                
                current_set = set(current_edges)
                
                for e in edges_to_remove:
                    if e not in current_set:
                        continue
                    # Try removing e
                    current_set.remove(e)
                    if not is_2_colorable(n, list(current_set)):
                        # Still non-colorable, keep removed
                        pass
                    else:
                        # Became colorable, add back
                        current_set.add(e)
                
                count = len(current_set)
                if count < best_count:
                    best_count = count
                    best_edges = list(current_set)
                    best_n = n
                    # If we find something very small, we can stop early?
                    # But let's run a few more to be sure.
                    if best_count <= 30:
                        # Very good, likely optimal or close
                        break
            
            # Optional: If current random set was colorable, we could try adding edges,
            # but the sample approach above is simpler.
            
        if best_count <= 30:
            break

    if best_edges is None:
        # Fallback: Return a known valid construction if search fails
        # The seed program returned something. We should ensure we return *something* valid.
        # If the random search fails to find a non-colorable set (unlikely for n>=9),
        # we can fall back to the complete graph K_9^(5) and prune it (slow but works).
        n = 9
        all_edges = list(combinations(range(n), 5))
        current_edges = all_edges
        random.seed(42)
        edges_to_try = all_edges.copy()
        random.shuffle(edges_to_try)
        
        current_set = set(current_edges)
        for e in edges_to_try:
            if e not in current_set:
                continue
            current_set.remove(e)
            if not is_2_colorable(n, list(current_set)):
                pass
            else:
                current_set.add(e)
        
        best_edges = list(current_set)
        best_n = n

    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
