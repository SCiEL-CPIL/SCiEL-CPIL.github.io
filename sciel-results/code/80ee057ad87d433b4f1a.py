import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Find the largest set of distinct integers in 1..256 such that
    subset sums of different cardinalities are disjoint.
    Uses backtracking with cardinality-indexed subset sum sets,
    plus a constraint-satisfaction removal loop for local improvement.
    """
    deadline = time.time() + 50  # 50s safety margin before 60s limit
    best_set = list(range(226, 257))  # fallback: 31 elements AP
    best_size = len(best_set)
    
    # For a given set, compute all subset sums indexed by cardinality
    def compute_cardinality_sums(s):
        """Return list: k -> set of all k-element subset sums"""
        n = len(s)
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        for x in s:
            for k in range(len(sums_by_k) - 1, -1, -1):
                if sums_by_k[k]:
                    for v in sums_by_k[k]:
                        sums_by_k[k + 1].add(v + x)
        return sums_by_k
    
    def is_valid(s):
        """Check if no two cardinalities share any sum value."""
        sums_by_k = compute_cardinality_sums(s)
        seen = set()
        for k in range(len(sums_by_k)):
            for v in sums_by_k[k]:
                if v in seen:
                    return False
                seen.add(v)
        return True
    
    def count_collisions(s):
        """Return (total_collision_count, element_conflict_dict).
        element_conflict_dict maps each element to the number of cross-cardinality
        collisions it is involved in."""
        n = len(s)
        if n == 0:
            return 0, {}
        sums_by_k = compute_cardinality_sums(s)
        
        # Build a map: sum_value -> list of (cardinality, frozenset_of_indices)
        # For each sum value, track which cardinalities produce it
        # We need to track which elements are in which subsets that produce each sum
        
        # Simpler approach: for each pair of cardinalities (k1, k2) with k1 != k2,
        # find sums that appear in both, and for each such sum, find the subsets
        # that produce it and count element conflicts.
        
        # Even simpler: for each element, count how many times removing it would
        # resolve a collision. But that's expensive.
        
        # Alternative: for each sum value v that appears in multiple cardinality sets,
        # find all (k, subset) pairs that produce v, and count each element's involvement.
        
        # Most efficient: build for each sum value the set of elements that appear
        # in subsets producing that sum at different cardinalities.
        
        # Let's use a different approach: for each element e, compute how many
        # cross-cardinality collisions involve e.
        
        # For a collision between sum v at cardinality k1 and sum v at cardinality k2,
        # we need to find subsets S1 of size k1 with sum v and S2 of size k2 with sum v.
        # Each element in S1 or S2 is "involved" in this collision.
        
        # This is expensive to enumerate. Instead, let's use a heuristic:
        # For each element e, compute the number of sums that would be "freed"
        # by removing e, weighted by how many other cardinalities also have those sums.
        
        # Simpler heuristic: for each element e, count the number of (k, v) pairs
        # such that v is in sums_by_k[k] and v is also in sums_by_k[j] for some j != k,
        # and e is in some k-element subset summing to v.
        
        # This is still expensive. Let's use a proxy: for each element e, compute
        # the number of subset sums that include e and collide with sums from other cardinalities.
        
        # Actually, let's just use a simple heuristic: for each element, count how many
        # sums in each cardinality set would be affected by its removal, and weight by
        # whether those sums collide with other cardinalities.
        
        # For simplicity and speed, let's use: for each element e, for each k, count
        # the number of v in sums_by_k[k] such that v - e is in sums_by_k[k-1] (meaning
        # e is in some k-subset summing to v) AND v is in sums_by_k[j] for some j != k.
        
        # This is O(n * n * |sums|) which is manageable for n <= 60.
        
        element_conflicts = {e: 0 for e in s}
        total_collisions = 0
        
        # First, find all colliding sum values
        sum_to_kardinalities = {}
        for k in range(n + 1):
            for v in sums_by_k[k]:
                if v not in sum_to_kardinalities:
                    sum_to_kardinalities[v] = set()
                sum_to_kardinalities[v].add(k)
        
        colliding_sums = {v: ks for v, ks in sum_to_kardinalities.items() if len(ks) > 1}
        total_collisions = sum(len(ks) - 1 for ks in colliding_sums.values())
        
        # For each element, count how many colliding sums it's involved in
        for e in s:
            for k in range(1, n + 1):
                if k - 1 < 0:
                    continue
                for v in sums_by_k[k]:
                    if v in colliding_sums:
                        # Check if e is in some k-subset summing to v
                        # i.e., v - e is in sums_by_k[k-1]
                        if v - e in sums_by_k[k - 1]:
                            element_conflicts[e] += 1
        
        return total_collisions, element_conflicts
    
    def constraint_satisfaction_improve(s):
        """Iteratively remove elements with highest conflict count until valid."""
        s = list(s)
        while True:
            if time.time() > deadline:
                break
            n = len(s)
            if n == 0:
                break
            total_conflicts, element_conflicts = count_collisions(s)
            if total_conflicts == 0:
                break
            # Find element with max conflicts
            max_elem = max(element_conflicts, key=lambda e: element_conflicts[e])
            if element_conflicts[max_elem] == 0:
                break
            s.remove(max_elem)
        return s
    
    # Backtracking: try adding elements from large to small
    # We build the set incrementally
    def backtrack(available, current, current_sums, depth):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        n = len(current)
        # Update best
        if n > best_size:
            best_size = n
            best_set = list(current)
        
        if not available:
            return
        
        # Try adding each available element (larger first for better separation)
        for i in range(len(available) - 1, -1, -1):
            if time.time() > deadline:
                return
            elem = available[i]
            new_avail = available[:i] + available[i+1:]
            
            # Quick check: can we add this?
            # Compute new cardinality sums
            new_sums = [set() for _ in range(n + 2)]
            for k in range(n + 1):
                new_sums[k] = set(current_sums[k])
            for k in range(n, -1, -1):
                for v in current_sums[k]:
                    new_sums[k + 1].add(v + elem)
            
            # Check validity
            valid = True
            seen = set()
            for k in range(n + 2):
                for v in new_sums[k]:
                    if v in seen:
                        valid = False
                        break
                    seen.add(v)
                if not valid:
                    break
            
            if valid:
                current.append(elem)
                backtrack(new_avail, current, new_sums, depth + 1)
                current.pop()
    
    # Phase 1: Backtracking search
    available = list(range(256, 0, -1))  # 256 down to 1
    current_sums = [set() for _ in range(1)]
    current_sums[0].add(0)
    
    backtrack(available, [], current_sums, 0)
    
    # Phase 2: Try to improve using constraint satisfaction on a random large set
    if time.time() < deadline - 5:
        # Start with a random subset of size best_size + 5 (capped at 256)
        target_size = min(best_size + 10, 256)
        random.seed(42)
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            # Random subset of size target_size from 1..256
            candidate = random.sample(range(1, 257), target_size)
            # Apply constraint satisfaction
            improved = constraint_satisfaction_improve(candidate)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = list(improved)
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
