"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
import time

def construct():
    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def sum_free_spectral(elements):
        """
        Find a large sum-free subset using spectral bisection on the Schur graph.
        Vertices: elements. Edge between i,j if elements[i]+elements[j] is in the set.
        We want an independent set in this graph (no edges).
        Use max-cut heuristic: partition into two sets, take the larger independent set,
        or iteratively refine.
        """
        n = len(elements)
        if n == 0:
            return []
        
        # Build the conflict graph: edge (i,j) if elements[i] + elements[j] is in elements
        elem_set = set(elements)
        elem_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Use sparse matrix for the graph
        # We'll build the adjacency matrix
        row = []
        col = []
        
        for i in range(n):
            for j in range(i+1, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    # Edge between i and k, and j and k
                    # Actually, the constraint is: no a+b=c where a,b,c in subset.
                    # So if a+b=c, then we cannot have all three in the subset.
                    # This is a 3-uniform hypergraph constraint, not a graph!
                    # Let me reconsider.
                    pass
        
        # The sum-free constraint is a 3-uniform hypergraph: for each (a,b,c) with a+b=c,
        # we cannot have all three in the subset. This is not a simple graph independence problem.
        # 
        # However, we can model it differently. Let's think of it as: we want to select a subset S
        # such that for no (a,b,c) in S do we have a+b=c.
        #
        # One approach: use a greedy algorithm with look-ahead, or use a local search.
        # Another approach: since the seed uses "odd numbers are sum-free", let's build on that
        # but improve it.
        #
        # Actually, let's try a different spectral approach. Consider the graph where vertices are
        # elements, and we add an edge between x and y if x+y is also in the set OR if x-y is in the set
        # (to handle the symmetric case). But this still doesn't directly give us an independent set
        # for the 3-uniform constraint.
        #
        # Let me try a hybrid: start with the parity-based solution, then try to add elements that
        # don't create conflicts, using a local search / hill climbing.
        
        # First, get the parity-based solution
        odd_subset = [x for x in elements if x % 2 != 0]
        even_subset = [x for x in elements if x % 2 == 0]
        
        # The odd subset is sum-free. Can we add some even elements?
        # If we add an even number e, we need to ensure that for no a,b in current subset, a+b=e
        # and for no a in current subset, a+e is in current subset (i.e., a+e is odd+even=odd, which could be in odd_subset)
        # and e+a is in current subset.
        
        # Let's try a local search approach:
        # Start with the larger of odd/even parity subsets, then try to improve.
        
        # Actually, let me think about this more carefully. The problem is a maximum independent set
        # in a 3-uniform hypergraph, which is NP-hard. Let's use a practical heuristic.
        
        # Strategy: 
        # 1. Start with all odd numbers (sum-free)
        # 2. Try to add even numbers one by one if they don't create conflicts
        # 3. If adding an even number creates conflicts, try removing some odds to free it up
        # 4. Repeat with even numbers as base
        
        def is_conflict(candidate, current_set, elem_set):
            """Check if adding candidate to current_set creates a sum-free violation."""
            # Check: for any a,b in current_set, a+b != candidate
            # and for any a in current_set, a + candidate != any element in current_set
            # and candidate + a != any element in current_set (same as above)
            # Also 2*candidate != any element in current_set
            for a in current_set:
                if a + candidate in current_set:
                    return True
                if 2 * a == candidate and a in current_set:
                    # a + a = candidate, so if a is in current_set, conflict
                    return True
            for a in current_set:
                for b in current_set:
                    if a + b == candidate:
                        return True
            return False

        # This is O(n^2) per check, which for n=240 is ~57600 ops per check, and we might do many checks.
        # Let's be smarter.
        
        def try_build_sum_free(elements, initial_subset):
            """
            Start with initial_subset (which is sum-free), try to add more elements.
            """
            current = set(initial_subset)
            remaining = [x for x in elements if x not in current]
            
            # Sort remaining by some heuristic - maybe by how many conflicts they have
            # An element x conflicts with pairs (a,b) where a+b=x or x+a=b or x+b=a
            # Let's just try adding elements greedily, checking for conflicts.
            
            # For efficiency, let's precompute which elements can coexist
            # Actually, let's just do a greedy approach with the current set
            
            added = True
            while added:
                added = False
                for x in remaining[:]:
                    # Check if x can be added
                    conflict = False
                    for a in current:
                        if a + x in current:
                            conflict = True
                            break
                        if 2 * a == x:
                            conflict = True
                            break
                    if not conflict:
                        for a in current:
                            for b in current:
                                if a + b == x:
                                    conflict = True
                                    break
                            if conflict:
                                break
                    if not conflict:
                        current.add(x)
                        remaining.remove(x)
                        added = True
            
            return list(current)
        
        # Try starting with odd numbers
        odd_result = try_build_sum_free(elements, odd_subset)
        
        # Try starting with even numbers (evens are NOT sum-free in general, so this might not work)
        # Actually, evens: if a,b even, a+b even, so evens can have conflicts.
        # Let's just try odd-based and also try a different approach.
        
        # Alternative: use a random restart local search
        # Start with a random subset, then add/remove elements to improve
        
        best_size = len(odd_result)
        best_subset = odd_result
        
        # Try a few random restarts
        import random
        rng_local = random.Random(42)
        
        for restart in range(20):
            # Start with a random subset of about half the elements
            current = set(rng_local.sample(elements, len(elements) // 2))
            
            # Remove conflicts
            improved = True
            while improved:
                improved = False
                # Find all conflicts and remove one element from each conflicting triple
                conflicts_found = False
                elems_list = list(current)
                for i in range(len(elems_list)):
                    for j in range(i+1, len(elems_list)):
                        s = elems_list[i] + elems_list[j]
                        if s in current:
                            # Remove the largest of the three (or some heuristic)
                            # Remove s (the sum)
                            current.discard(s)
                            conflicts_found = True
                            improved = True
                            break
                    if improved:
                        break
                if not conflicts_found:
                    break
            
            # Now try to add elements
            remaining = [x for x in elements if x not in current]
            added = True
            while added:
                added = False
                for x in remaining[:]:
                    conflict = False
                    for a in current:
                        if a + x in current:
                            conflict = True
                            break
                        if 2 * a == x:
                            conflict = True
                            break
                    if not conflict:
                        for a in current:
                            for b in current:
                                if a + b == x:
                                    conflict = True
                                    break
                            if conflict:
                                break
                    if not conflict:
                        current.add(x)
                        remaining.remove(x)
                        added = True
            
            if len(current) > best_size:
                best_size = len(current)
                best_subset = list(current)
        
        return best_subset

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        subset = sum_free_spectral(elements)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
