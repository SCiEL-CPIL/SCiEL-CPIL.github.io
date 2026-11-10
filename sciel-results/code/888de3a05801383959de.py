# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses branch-and-bound with local search improvements.
    """
    m = 205
    
    # Build forbidden set
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build conflict adjacency: conflict[i][j] = True if i and j cannot both be in the set
    # i.e., (i-j) mod m or (j-i) mod m is in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Precompute neighbor sets for each vertex
    neighbors = [set(np.where(conflict[i])[0]) for i in range(m)]
    
    def is_valid_independent_set(S):
        """Check if S is an independent set."""
        S_set = set(S)
        for i in S_set:
            for j in S_set:
                if i != j and conflict[i, j]:
                    return False
        return True
    
    def local_search(S, max_iters=50000):
        """Try to improve S by adding, removing, or swapping elements."""
        S_set = set(S)
        best_size = len(S_set)
        best_set = set(S_set)
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding a vertex
            for v in range(m):
                if v in S_set:
                    continue
                # Check if v can be added
                can_add = True
                for u in S_set:
                    if conflict[v, u]:
                        can_add = False
                        break
                if can_add:
                    S_set.add(v)
                    improved = True
                    if len(S_set) > best_size:
                        best_size = len(S_set)
                        best_set = set(S_set)
                    break
            
            if not improved:
                # Try swapping: remove one, add another
                swapped = False
                for u in list(S_set):
                    S_set.remove(u)
                    for v in range(m):
                        if v in S_set:
                            continue
                        can_add = True
                        for w in S_set:
                            if conflict[v, w]:
                                can_add = False
                                break
                        if can_add:
                            S_set.add(v)
                            swapped = True
                            if len(S_set) > best_size:
                                best_size = len(S_set)
                                best_set = set(S_set)
                            break
                    if swapped:
                        break
                    S_set.add(u)  # restore
                
                if not swapped:
                    # Try removing one and adding two (if possible)
                    for u in list(S_set):
                        S_set.remove(u)
                        added = 0
                        for v in range(m):
                            if v in S_set or v == u:
                                continue
                            can_add = True
                            for w in S_set:
                                if conflict[v, w]:
                                    can_add = False
                                    break
                            if can_add:
                                S_set.add(v)
                                added += 1
                                if added >= 2:
                                    break
                        if added > 1:
                            if len(S_set) > best_size:
                                best_size = len(S_set)
                                best_set = set(S_set)
                        else:
                            S_set.add(u)  # restore
                            # remove the one we added if any
                            for v in list(S_set):
                                if v != u and v not in best_set:
                                    if v not in S_set:
                                        break
                                    S_set.discard(v)
                            # Actually, let's just restore properly
                            S_set = set(best_set) if len(S_set) < best_size else S_set
                            break
                    else:
                        # No improvement possible, try random perturbation
                        if len(S_set) > 0:
                            # Randomly remove a random element and try to rebuild
                            u = random.choice(list(S_set))
                            S_set.remove(u)
                            for v in range(m):
                                if v in S_set:
                                    continue
                                can_add = True
                                for w in S_set:
                                    if conflict[v, w]:
                                        can_add = False
                                        break
                                if can_add:
                                    S_set.add(v)
                                    break
                            if len(S_set) < best_size:
                                S_set = set(best_set)
            
            if len(S_set) == m:
                break
        
        return list(best_set)
    
    def branch_and_bound(vertices, current_set, best_set):
        """
        Branch and bound to find a large independent set.
        vertices: list of candidate vertices to consider
        current_set: set of vertices already chosen
        best_set: best independent set found so far
        Returns: best independent set found
        """
        if len(current_set) + len(vertices) <= len(best_set):
            return best_set
        
        if not vertices:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            return best_set
        
        # Choose vertex with minimum degree in the remaining subgraph
        # (to minimize branching)
        v = min(vertices, key=lambda x: len(neighbors[x] & set(vertices)))
        
        # Option 1: Include v
        new_set = set(current_set)
        new_set.add(v)
        # Remove all neighbors of v from candidates
        remaining = [u for u in vertices if u != v and not conflict[v, u]]
        best_set = branch_and_bound(remaining, new_set, best_set)
        
        # Option 2: Exclude v
        remaining = [u for u in vertices if u != v]
        best_set = branch_and_bound(remaining, current_set, best_set)
        
        return best_set
    
    # Use a smarter approach: greedy initialization + local search with multiple restarts
    best_result = set()
    
    # Multiple random restarts with different orderings
    for trial in range(50):
        random.seed(trial * 12345 + 67890)
        
        # Random ordering of vertices
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy construction with the given order
        S = set()
        for v in order:
            can_add = True
            for u in S:
                if conflict[v, u]:
                    can_add = False
                    break
            if can_add:
                S.add(v)
        
        # Local search improvement
        S_list = local_search(list(S), max_iters=20000)
        S_set = set(S_list)
        
        if len(S_set) > len(best_result):
            best_result = S_set
    
    # Also try a systematic approach: start from all vertices, remove conflicts
    # Greedy by degree
    for trial in range(20):
        random.seed(trial * 54321 + 11111)
        
        # Order by degree (ascending) with some randomization
        degrees = [len(neighbors[i]) for i in range(m)]
        order = list(range(m))
        random.shuffle(order)
        order.sort(key=lambda x: (degrees[x] + random.random(), x))
        
        S = set()
        for v in order:
            can_add = True
            for u in S:
                if conflict[v, u]:
                    can_add = False
                    break
            if can_add:
                S.add(v)
        
        S_list = local_search(list(S), max_iters=20000)
        S_set = set(S_list)
        
        if len(S_set) > len(best_result):
            best_result = S_set
    
    # Final local search polish
    if best_result:
        best_list = local_search(list(best_result), max_iters=50000)
        best_result = set(best_list)
    
    # Verify the result
    if not is_valid_independent_set(list(best_result)):
        # Fallback: use a simple valid set
        best_result = {0}
    
    return np.array(sorted(best_result), dtype=np.int64)


# EVOLVE-BLOCK-END
