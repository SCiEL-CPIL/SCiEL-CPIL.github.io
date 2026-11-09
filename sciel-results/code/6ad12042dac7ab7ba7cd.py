"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _is_c4_free(adj, n):
    """Check if graph with adjacency list adj on n vertices is C4-free.
    C4-free means no two vertices share more than one common neighbor."""
    for i in range(n):
        for j in range(i + 1, n):
            # Count common neighbors
            common = 0
            for nb in adj[i]:
                if nb > j:  # only count neighbors with index > j to avoid double counting
                    if nb in adj[j]:
                        common += 1
                        if common > 1:
                            return False
    return True

def _has_independent_set_k(adj, n, k, max_sets=None):
    """Check if there is an independent set of size k.
    Uses a simple backtracking with pruning.
    Returns True if found, False otherwise."""
    if max_sets is not None:
        found_count = [0]
    
    def backtrack(candidates, current_size):
        if max_sets is not None:
            if found_count[0] >= max_sets:
                return True
        if current_size == k:
            if max_sets is not None:
                found_count[0] += 1
            return True
        # Pruning: not enough candidates left
        if len(candidates) < k - current_size:
            return False
        for idx, v in enumerate(candidates):
            # Try including v
            new_candidates = candidates[idx+1:]
            # Filter out neighbors of v
            filtered = [u for u in new_candidates if u not in adj[v]]
            if backtrack(filtered, current_size + 1):
                return True
        return False
    
    all_vertices = list(range(n))
    return backtrack(all_vertices, 0)

def _independence_number(adj, n, max_check=12):
    """Estimate independence number by checking if alpha >= k for k from max_check down to 1.
    Returns the largest k such that we confirmed an independent set of size k exists (or a lower bound)."""
    # We want to check if alpha >= 11. If yes, return at least 11.
    # For our purpose, we just need to know if alpha < 11.
    # Check k=11 first. If no independent set of size 11, alpha <= 10.
    # But we need to maximize n, so we want alpha <= 10.
    # Actually the requirement is: no independent set of 11 vertices, i.e., alpha <= 10.
    return _has_independent_set_k(adj, n, 11)

def _construct_paley_like(n, prime_p):
    """
    Construct a graph on n vertices using a quadratic residue construction.
    This is inspired by Paley graphs but modified for C4-free property.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_result = {'n': 30, 'edges': []}
    # Initialize with the trivial 30-vertex construction
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_result['edges'] += [[a, b], [a, c], [b, c]]
    
    random.seed(42)
    
    # Strategy: Build C4-free graphs by adding edges carefully,
    # then check independence number.
    # 
    # Key insight: A graph is C4-free iff for every pair of vertices,
    # they have at most one common neighbor.
    #
    # Approach: Use a greedy/randomized approach to build dense C4-free graphs,
    # then verify alpha <= 10.
    
    # Try various n values from high to low
    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
        
        # Multiple attempts for each n
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Randomized construction:
            # Start with empty graph, add edges that don't create C4.
            # Use a heuristic to bias toward denser graphs.
            
            adj = [set() for _ in range(n)]
            edges = []
            
            # Try to add edges in a random order, skipping those that create C4
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            for (i, j) in possible_edges:
                if time.time() > deadline - 2:
                    break
                # Check if adding edge (i,j) creates a C4
                # A C4 is created if i and j have more than one common neighbor
                common = adj[i] & adj[j]
                if len(common) > 1:
                    continue
                # Also need to check: adding (i,j) means for any k in adj[i], 
                # k and j must not already have a common neighbor other than i
                # And for any k in adj[j], k and i must not already have a common neighbor other than j
                
                # Actually, the C4-free condition is: no two vertices share >1 common neighbor.
                # When we add edge (i,j), we need to verify the condition still holds.
                # The only pairs whose common-neighbor count changes are:
                # - (i,j): their common neighbors are adj[i] ∩ adj[j] (before adding)
                #   After adding, they still have the same common neighbors (the edge doesn't add common neighbors)
                #   Wait, adding edge (i,j) doesn't change adj[i] ∩ adj[j] in terms of common neighbors.
                #   The common neighbors of i and j are vertices adjacent to both. Adding (i,j) doesn't make
                #   any new vertex a common neighbor of i and j.
                # - For any k in adj[i] (k != j): i and k now... wait, i is already adjacent to k.
                #   Adding (i,j) affects the common neighbors of (j,k) if k is in adj[i]:
                #   j and k: common neighbors include i (since j-i and i-k). 
                #   We need to check if j and k have another common neighbor.
                # - Similarly for k in adj[j].
                
                # Let's just check: after hypothetically adding (i,j), is the graph still C4-free?
                # This is expensive. Instead, let's check the necessary conditions:
                
                # 1. i and j must have at most 1 common neighbor (already checked above)
                # 2. For each k in adj[i] (k != j): j and k must have at most 1 common neighbor after adding (i,j)
                #    Before adding, common neighbors of j and k: adj[j] ∩ adj[k]
                #    After adding (i,j): if i is in adj[k] (yes, since k in adj[i] means i in adj[k]), 
                #    then i becomes a common neighbor of j and k.
                #    So we need: |adj[j] ∩ adj[k]| <= 1 (before), and if i in adj[j]∩adj[k] that's fine,
                #    but we need |adj[j] ∩ adj[k] ∪ {i}| <= 1, i.e., |adj[j] ∩ adj[k]| <= 0 if i not already in it,
                #    or |adj[j] ∩ adj[k]| <= 1 if i already in it.
                #    Wait, i is in adj[k] (since k in adj[i]). Is i in adj[j]? Not yet (we're adding it).
                #    So after adding, i is in adj[j] and in adj[k], so i is a common neighbor of j and k.
                #    We need total common neighbors of j and k to be <= 1.
                #    So we need |adj[j] ∩ adj[k]| <= 1, and if i is not already in adj[j]∩adj[k] (it's not, since i not in adj[j] yet),
                #    then after adding, common neighbors of j,k = (adj[j] ∩ adj[k]) ∪ {i}.
                #    So we need |adj[j] ∩ adj[k]| + 1 <= 1, i.e., |adj[j] ∩ adj[k]| == 0.
                #    Wait, that's only if i is not already a common neighbor. Since i is not in adj[j] yet,
                #    i is not in adj[j] ∩ adj[k]. So after adding, common(j,k) = old_common(j,k) ∪ {i}.
                #    We need |old_common(j,k) ∪ {i}| <= 1, which means old_common(j,k) must be empty.
                
                can_add = True
                # Check for all k in adj[i]
                for k in adj[i]:
                    if k == j:
                        continue
                    # j and k: need common neighbors <= 1 after adding (i,j)
                    # i will be a common neighbor of j and k (since i-j and i-k)
                    # So we need no other common neighbors of j and k
                    if adj[j] & adj[k]:
                        can_add = False
                        break
                if not can_add:
                    continue
                # Check for all k in adj[j]
                for k in adj[j]:
                    if k == i:
                        continue
                    # i and k: need common neighbors <= 1 after adding (i,j)
                    # j will be a common neighbor of i and k (since j-i and j-k)
                    # So we need no other common neighbors of i and k
                    if adj[i] & adj[k]:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Also need to check: for pairs (i, k) where k in adj[j], k != i
                # and pairs (j, k) where k in adj[i], k != j
                # These are covered above.
                
                # What about pairs not involving i or j? Their common neighbors don't change.
                
                # Add the edge
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
            
            # Check C4-free property
            if not _is_c4_free(adj, n):
                continue
            
            # Check independence number: need alpha <= 10, i.e., no independent set of size 11
            has_indep_11 = _has_independent_set_k(adj, n, 11)
            if not has_indep_11:
                # Valid graph!
                best_result = {'n': n, 'edges': edges}
                break  # Found a valid graph for this n, try larger n
        
        # If we found a valid graph for this n, break to try larger
        if best_result['n'] == n:
            break
    
    return best_result

# EVOLVE-BLOCK-END
