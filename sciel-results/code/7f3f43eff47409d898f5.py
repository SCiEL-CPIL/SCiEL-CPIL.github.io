"""Optimized linear triple system search for 16 points."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # We want to minimize the independence number (largest set with no triple).
    # A linear triple system on 16 points where any two triples share at most one point.
    
    # Strategy: Use simulated annealing or local search to find a packing of triples
    # that minimizes the independence number.
    
    # First, let's think about what a good solution looks like.
    # If we have a Steiner triple system or close to it, the independence number is smaller.
    # For 16 points, a perfect packing would have 16*15/6 = 40 triples if every pair appears exactly once.
    # But we need a linear system (any two triples share at most one point), which is satisfied by STS if it exists.
    # STS(16) doesn't exist since 16 ≡ 1 or 3 mod 6 is required, and 16 ≡ 4 mod 6.
    # So we can't have a complete STS. We can have a maximal packing.
    
    # The independence number α is the largest subset of points with no triple entirely contained in it.
    # To minimize α, we want triples to be spread out as evenly as possible.
    
    # Let's use a greedy approach combined with local search.
    
    def independence_number(triples):
        """Calculate the independence number of the hypergraph."""
        n = 16
        # For each subset size from high to low, check if any subset of that size has no triple.
        # We can use a backtracking or bitset approach.
        # For 16 points, we can enumerate all subsets of size k for k from 16 down to 1.
        # But that's 2^16 = 65536 subsets total, which is feasible.
        
        # For each subset, check if it contains any triple.
        # A subset S contains a triple if there exists (a,b,c) in triples such that a,b,c all in S.
        
        # We can precompute for each triple a bitmask.
        # Then for each subset bitmask, check if any triple bitmask is a subset of it.
        
        triple_masks = [0] * len(triples)
        for i, (a, b, c) in enumerate(triples):
            triple_masks[i] = (1 << a) | (1 << b) | (1 << c)
        
        # Check subsets from size 16 down to 1
        # Actually, we want the LARGEST subset with NO triple.
        # So we check from size 16 down, and return the first size where some subset has no triple.
        
        for size in range(16, 0, -1):
            # Check all subsets of this size
            # We can iterate through all masks and check if popcount == size
            # For efficiency, we can generate combinations, but for 16 points it's fine to check all masks.
            found = False
            for mask in range(1, 1 << 16):
                if bin(mask).count('1') != size:
                    continue
                # Check if this mask contains any triple
                has_triple = False
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        has_triple = True
                        break
                if not has_triple:
                    found = True
                    break
            if found:
                return size
        return 0
    
    # This independence_number function is too slow for repeated calls.
    # Let's optimize it.
    
    def independence_number_fast(triples):
        """Faster independence number calculation using precomputed triple coverage."""
        n = 16
        if not triples:
            return n
        
        triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
        
        # For each subset size from 16 down to 1, check if any subset of that size is triple-free.
        # We can use a different approach: for each point set, track which triples it covers.
        
        # Alternative: use the fact that a set is triple-free iff it doesn't contain any triple.
        # We can iterate over all 2^16 masks and for each, check if it's triple-free.
        # Then find the maximum size among triple-free sets.
        
        max_size = 0
        for mask in range(1, 1 << 16):
            # Check if mask is triple-free
            is_free = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_free = False
                    break
            if is_free:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
        return max_size
    
    # Still might be slow. Let's think differently.
    # For 16 points, 2^16 = 65536 masks. For each mask, we check up to ~40 triples.
    # That's ~2.6 million operations per evaluation. If we do this thousands of times, it's too slow.
    
    # Let's use a smarter approach: for each triple, mark which masks it "hits".
    # A mask is triple-free if it's not hit by any triple.
    # We can precompute for each triple the set of masks it hits, but that's too much memory.
    
    # Alternative: iterate masks, and for each, check triples. With early termination, it might be fast enough
    # if we limit the number of local search iterations.
    
    # Let's try a different approach: use the relationship between independence number and clique cover.
    # Actually, let's just implement a reasonable local search and see how it goes.
    
    # Generate an initial solution using a greedy algorithm.
    def initial_solution():
        """Generate an initial packing of triples."""
        triples = []
        used_pairs = set()
        points = list(range(16))
        
        # Greedy: add triples that don't share more than one point with existing triples.
        random.shuffle(points)
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    a, b, c = points[i], points[j], points[k]
                    # Check if adding (a,b,c) violates linearity
                    valid = True
                    for (x, y, z) in triples:
                        common = len({a,b,c} & {x,y,z})
                        if common > 1:
                            valid = False
                            break
                    if valid:
                        triples.append([a, b, c])
                        used_pairs.add((a,b))
                        used_pairs.add((a,c))
                        used_pairs.add((b,c))
        
        return triples
    
    # Local search: try to improve by swapping points in triples.
    def local_search(triples, max_iters=5000):
        """Improve the solution via local search."""
        current = [list(t) for t in triples]
        current_score = independence_number_fast(current)
        best = [list(t) for t in current]
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random triple and try to modify it.
            if not current:
                break
            idx = random.randint(0, len(current)-1)
            a, b, c = current[idx]
            
            # Try replacing one point with another.
            for pos in range(3):
                old_pt = current[idx][pos]
                for new_pt in range(16):
                    if new_pt == old_pt:
                        continue
                    # Check if new_pt is already in the triple
                    if new_pt in (a, b, c) and new_pt != old_pt:
                        continue
                    
                    # Create new triple
                    new_triple = list(current[idx])
                    new_triple[pos] = new_pt
                    new_triple = sorted(new_triple)
                    
                    # Check linearity
                    valid = True
                    for j, (x, y, z) in enumerate(current):
                        if j == idx:
                            continue
                        common = len(set(new_triple) & {x, y, z})
                        if common > 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Apply change
                    current[idx] = new_triple
                    new_score = independence_number_fast(current)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best = [list(t) for t in current]
                    else:
                        # Revert
                        current[idx] = [a, b, c]
        
        return best, best_score
    
    # Run local search with multiple restarts.
    best_overall = None
    best_score = 16
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        sol = initial_solution()
        improved, score = local_search(sol, max_iters=2000)
        
        if score < best_score:
            best_score = score
            best_overall = improved
        
        if best_score <= 4:  # Can't do much better than this for 16 points
            break
    
    if best_overall is None:
        return initial_solution()
    
    return best_overall
