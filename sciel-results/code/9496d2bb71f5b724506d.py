import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    def is_valid_fast(n, edges):
        """
        Check if graph has no regular induced subgraph on k>=6 vertices.
        Uses vectorized numpy operations for speed.
        """
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        # We only need to check k=6 to n.
        # If a subset of size k is regular, all its vertices have the same degree within the subset.
        # For k=6, degrees are 0..5.
        
        # Optimization: If we find a regular subset of size 6, it's invalid.
        # If we check all sizes, it's expensive. 
        # However, note that if a set of size K is regular, it might contain smaller regular subsets? 
        # Not necessarily. But usually, small k is the bottleneck.
        # Let's check k from 6 to n.
        
        # Precompute all masks? No, too many.
        # Generate combinations of indices.
        
        # Vectorized check for a specific k:
        # 1. Generate all combinations of k vertices.
        # 2. For each combination, extract the k x k submatrix.
        # 3. Compute row sums (degrees).
        # 4. Check if all row sums are equal.
        
        # This is still O(C(n,k) * k^2) per k.
        # For n=21, k=6: C(21,6) = 54264.
        # 54264 * 36 ops ~ 2M ops. Numpy overhead is high for small arrays.
        
        # Alternative: Use bitmasks and precomputed popcounts?
        # Let's stick to a hybrid: 
        # For small k (6,7,8), use itertools + numpy vectorization on batches.
        # For large k, it's less likely to be regular, but we must check.
        
        # Actually, a simpler observation:
        # A graph is "regular-free" on k>=6 if no subset of size k is regular.
        # Most random graphs will have regular subsets of size 6.
        # The constraint is very tight.
        
        # Let's use a smarter check:
        # For each subset of size k, calculate degrees.
        # We can vectorize the degree calculation for all subsets of a fixed k.
        
        def check_k(k):
            if k > n: return True
            
            # Get all combinations of k vertices
            # To avoid memory explosion, process in chunks if k is large, but k<=21 is small.
            # C(21, 10) is ~350k. Storing 350k masks is fine.
            
            # However, generating combinations in Python is slow.
            # Let's try to generate masks using numpy or a fast generator.
            
            # For simplicity and speed in Python, let's use a recursive or iterative mask generator
            # or just itertools for k=6..n but with a very fast inner loop using numpy.
            
            # Actually, for n=21, total subsets is 2^21 ~ 2M.
            # We can iterate through all masks from 0 to 2^n - 1? No, 2M is too many for pure Python loop.
            
            # Let's stick to itertools for k=6..n but optimize the check.
            # The current program fails because is_valid is too slow (2s per check).
            # If we make is_valid 100x faster (0.02s), we can do 5000 checks.
            
            # How to speed up is_valid?
            # 1. Precompute adjacency as integers (bitmasks).
            # 2. For each subset mask, compute degrees of vertices in mask.
            #    deg[v] = popcount(adj[v] & mask)
            #    This requires extracting vertices from mask.
            
            # Better approach:
            # Iterate subsets of size k.
            # For a given subset (list of vertices), compute degrees.
            # Use numpy to compute degrees for all subsets of a fixed k in parallel?
            # Hard to vectorize variable-length subsets easily without padding.
            
            # Let's try a different angle:
            # Random search is unlikely to hit a valid graph directly for n=20/21.
            # We need Local Search.
            # Start with a random graph.
            # Find all violating subsets (regular subsets of size >= 6).
            # Flip an edge to break as many violations as possible.
            # Repeat.
            
            # To find violations fast:
            # We only need to find ONE violating subset to know it's invalid.
            # If we find one, we can use it to guide the mutation.
            
            # Revised is_valid: returns (valid, violating_subset)
            # We check k=6 first (most likely to fail).
            # Then k=7, etc.
            
            # Fast check for k=6:
            # There are C(20,6) = 38760 subsets.
            # We can check them in a loop.
            # For each subset, compute 6 degrees.
            # 38760 * 6 = 232k bit_count ops.
            # In Python, 232k bit_count ops take ~0.05s.
            # So checking k=6 takes ~0.05s.
            # Checking k=6..20 takes ~0.5s.
            # This is 4x faster than the current 2s.
            # Still, 100s / 0.5s = 200 checks.
            # 200 random tries for n=20 might not find a valid one if the probability is low.
            
            # Therefore, LOCAL SEARCH is essential.
            
            pass
            
        # Let's implement the local search strategy directly in the main loop.
        # This function is just for final validation.
        
        # Fast validation using bitmasks
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # Precompute list of vertices for each mask? No.
        # We will iterate k from 6 to n.
        # For efficiency, we break early if a regular subset is found.
        
        for k in range(6, n + 1):
            # Generate combinations
            # Optimization: If k > n//2, checking the complement might be faster? No, regularity isn't preserved under complement easily.
            
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                # Get degrees of vertices in subset
                d0 = (adj_masks[subset[0]] & mask).bit_count()
                regular = True
                for v in subset[1:]:
                    d = (adj_masks[v] & mask).bit_count()
                    if d != d0:
                        regular = False
                        break
                if regular:
                    return False
        return True

    def find_violation(n, edges):
        """
        Finds one regular induced subgraph of size >= 6.
        Returns (subset, k) or None if valid.
        Optimized to find small k first.
        """
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                d0 = (adj_masks[subset[0]] & mask).bit_count()
                regular = True
                for v in subset[1:]:
                    d = (adj_masks[v] & mask).bit_count()
                    if d != d0:
                        regular = False
                        break
                if regular:
                    return subset, k
        return None

    def get_edge_set(edges):
        return set(map(tuple, edges))

    def local_search(n, initial_edges, max_iters=5000):
        """
        Local search to eliminate regular subsets.
        """
        edges_set = get_edge_set(initial_edges)
        
        # Precompute all pairs for potential flips
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        current_edges = set(edges_set)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Find a violation
            violation = find_violation(n, list(current_edges))
            if violation is None:
                return list(current_edges), True
            
            subset, k = violation
            
            # Try to flip an edge to break this violation.
            # An edge flip changes degrees of vertices in the subset.
            # If we add an edge (u,v) where u,v in subset:
            #   deg[u] += 1, deg[v] += 1
            # If we remove an edge (u,v) where u,v in subset:
            #   deg[u] -= 1, deg[v] -= 1
            # Edges connecting subset to outside don't change internal degrees.
            
            # Strategy: Pick a random edge (u,v) with both u,v in subset.
            # Flip it.
            # Check if this reduces the number of violations or breaks this one.
            
            # To be efficient, we just pick a random internal edge and flip it.
            # Then we re-check.
            
            # Better: Pick an edge that is most likely to break regularity.
            # If all degrees are d, adding an edge makes two vertices d+1.
            # This breaks regularity unless d+1 == d (impossible) or other vertices also change (they don't).
            # So ANY internal edge flip breaks the regularity of THIS specific subset.
            # But it might create new violations.
            
            # Let's pick a random internal edge.
            internal_edges = [(u, v) for u in subset for v in subset if u < v and (u, v) in current_edges]
            non_internal_edges = [(u, v) for u in subset for v in subset if u < v and (u, v) not in current_edges]
            
            candidates = []
            if internal_edges:
                candidates.append(('remove', random.choice(internal_edges)))
            if non_internal_edges:
                candidates.append(('add', random.choice(non_internal_edges)))
                
            if not candidates:
                # Should not happen for k>=2
                break
            
            action, (u, v) = random.choice(candidates)
            
            if action == 'remove':
                current_edges.discard((u, v))
            else:
                current_edges.add((u, v))
                
        # Final check
        violation = find_violation(n, list(current_edges))
        if violation is None:
            return list(current_edges), True
        return list(current_edges), False

    # Main Strategy:
    # 1. Try to find a valid n=20 graph via local search starting from random graphs.
    # 2. If found, try to expand to n=21? Or just return 20.
    # 3. The task says "Maximize n". 20 is known. 21 is the goal.
    # 4. Finding a valid 21-vertex graph is much harder.
    # 5. We will focus on reliably finding a valid 20-vertex graph quickly, 
    #    and spend remaining time trying to find a 21-vertex graph.
    
    # Step 1: Find a valid n=20 graph.
    best_result = {'n': 5, 'edges': []}
    
    # Try several random starts for n=20
    found_20 = False
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        random.seed(attempt * 12345 + 67890)
        # Start with p=0.5
        edges = [[i, j] for i in range(20) for j in range(i+1, 20) if random.random() < 0.5]
        
        # Local search
        valid_edges, is_valid = local_search(20, edges, max_iters=1000)
        
        if is_valid:
            best_result = {'n': 20, 'edges': valid_edges}
            found_20 = True
            break
            
    # If we found a 20-vertex graph, try to see if we can find a 21-vertex one.
    # This is very hard. We can try random starts for n=21 with local search.
    if found_20 and time.time() < deadline - 10:
        # Try n=21
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 98765 + 54321)
            edges = [[i, j] for i in range(21) for j in range(i+1, 21) if random.random() < 0.5]
            
            valid_edges, is_valid = local_search(21, edges, max_iters=500)
            
            if is_valid:
                best_result = {'n': 21, 'edges': valid_edges}
                break
                
    # If we didn't find 20, try 19 as fallback
    if not found_20:
        for attempt in range(50):
            if time.time() > deadline:
                break
            random.seed(attempt * 11111 + 22222)
            edges = [[i, j] for i in range(19) for j in range(i+1, 19) if random.random() < 0.5]
            valid_edges, is_valid = local_search(19, edges, max_iters=1000)
            if is_valid:
                best_result = {'n': 19, 'edges': valid_edges}
                break
                
    return best_result
