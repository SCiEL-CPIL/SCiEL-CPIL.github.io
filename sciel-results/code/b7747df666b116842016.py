"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 90s limit
    
    # Build 4-AP hypergraph using bitsets for efficiency
    # Each element maps to a list of 4-APs it participates in
    # We'll use a different approach: for each element, store which other elements 
    # would complete a 4-AP with it
    
    # Precompute: for each pair (a, b), determine if they can be part of a 4-AP
    # and what the other two elements would be
    # 
    # Better: For each element x, precompute the "conflict sets"
    # If we have elements a, b, c in our set, and they form 3 terms of a 4-AP,
    # then the 4th term is forbidden
    #
    # Key insight: A 4-AP is {a, a+d, a+2d, a+3d}
    # For a set S to be 4-AP-free, for any 3 elements in S that are part of a 4-AP,
    # the 4th element must not be in S
    
    # Build adjacency: for each x, which pairs (y,z) would make {x,y,z,w} a 4-AP for some w
    # Actually, let's think about it differently:
    # When we try to add x to current set C, we need to check if adding x creates a 4-AP
    # This happens if there exist y,z,w in C such that {x,y,z,w} is a 4-AP
    
    # For efficiency, precompute for each x the list of "forbidden triples" 
    # i.e., triples (y,z,w) such that {x,y,z,w} is a 4-AP
    
    # Alternative efficient approach:
    # For each x, compute the set of elements that "conflict" with x in the sense that
    # if both are in the set, they constrain other elements
    
    # Let me use a direct approach with bitsets:
    # Represent the current set as a bitset of length n
    # For each x, precompute a list of "patterns" - each pattern is a bitset 
    # representing which elements would complete a 4-AP with x and two existing elements
    
    # Actually, the most efficient approach for this problem size:
    # Precompute all 4-APs, then for each element, store which 4-APs it's in
    # Use branch and bound with good upper bounds
    
    # Step 1: Generate all 4-APs in [1,n]
    all_4aps = []
    ap_index = {}  # element -> list of AP indices
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                idx = len(all_4aps)
                all_4aps.append((a - 1, b - 1, c - 1, e - 1))  # 0-indexed
                for elem in (a - 1, b - 1, c - 1, e - 1):
                    if elem not in ap_index:
                        ap_index[elem] = []
                    ap_index[elem].append(idx)
    
    num_aps = len(all_4aps)
    
    # Step 2: For each element, precompute the 4-APs it belongs to
    # And for each 4-AP, precompute the bitset of its elements
    ap_bitsets = []
    for ap in all_4aps:
        bs = 0
        for elem in ap:
            bs |= (1 << elem)
        ap_bitsets.append(bs)
    
    # Step 3: Greedy initialization
    def greedy_init():
        """Greedy construction using bitset operations."""
        selected = 0  # bitset
        remaining = (1 << n) - 1  # all elements available
        
        for x in range(n):
            # Check if x can be added: no 4-AP containing x should have all other 3 elements in selected
            can_add = True
            if x in ap_index:
                for ap_idx in ap_index[x]:
                    ap = all_4aps[ap_idx]
                    # Check if the other 3 elements are in selected
                    other_count = 0
                    for elem in ap:
                        if elem != x and (selected >> elem) & 1:
                            other_count += 1
                    if other_count == 3:
                        can_add = False
                        break
            if can_add:
                selected |= (1 << x)
        
        return selected
    
    # Step 4: Local search with efficient conflict checking
    def can_add_bitset(x, selected):
        """Check if x can be added to selected (bitset) without forming 4-AP."""
        if x not in ap_index:
            return True
        for ap_idx in ap_index[x]:
            ap = all_4aps[ap_idx]
            other_count = 0
            for elem in ap:
                if elem != x and (selected >> elem) & 1:
                    other_count += 1
            if other_count == 3:
                return False
        return True
    
    def local_search_bitset(selected, max_iters=100):
        """Local search using bitset representation."""
        current = selected
        current_size = bin(current).count('1')
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding any element
            for x in range(n):
                if not (current >> x) & 1:
                    if can_add_bitset(x, current):
                        current |= (1 << x)
                        current_size += 1
                        improved = True
                        break
            
            if not improved:
                # Try removing one element to allow adding more
                for x in range(n):
                    if (current >> x) & 1:
                        trial = current & ~(1 << x)
                        added = 0
                        for y in range(n):
                            if not (trial >> y) & 1:
                                if can_add_bitset(y, trial):
                                    trial |= (1 << y)
                                    added += 1
                        if added > 1:
                            current = trial
                            current_size = bin(current).count('1')
                            improved = True
                            break
            
            if not improved:
                break
        
        return current
    
    # Step 5: Branch and bound for maximum independent set
    # Upper bound: for remaining elements, at most one from each 4-AP
    # We can use a greedy coloring of the conflict hypergraph for better bounds
    
    def upper_bound(remaining, selected):
        """Compute upper bound on additional elements we can add."""
        if remaining == 0:
            return 0
        
        # Simple bound: count remaining elements
        count = bin(remaining).count('1')
        
        # Better bound: for each 4-AP, at most 1 element can be added
        # But this is loose. Use a packing argument.
        
        # Greedy matching bound: greedily match elements to APs they're in
        # Each AP can contribute at most 1 to the independent set among remaining
        matched = 0
        used_aps = set()
        rem = remaining
        while rem:
            # Take lowest set bit
            x = (rem & -rem).bit_length() - 1
            rem ^= (1 << x)
            # Find an AP containing x that hasn't been used
            if x in ap_index:
                for ap_idx in ap_index[x]:
                    if ap_idx not in used_aps:
                        used_aps.add(ap_idx)
                        matched += 1
                        break
        
        return min(count, matched) if matched > 0 else count
    
    def branch_and_bound(remaining, selected, current_size, best_size_ref, max_depth=20):
        """Branch and bound search."""
        if time.time() > deadline:
            return
        
        if current_size >= best_size_ref[0]:
            return
        
        if remaining == 0:
            return
        
        # Upper bound check
        ub = current_size + upper_bound(remaining, selected)
        if ub <= best_size_ref[0]:
            return
        
        if max_depth == 0:
            return
        
        # Choose element with highest degree in remaining subproblem
        # For simplicity, pick first available element
        x = (remaining & -remaining).bit_length() - 1
        
        # Branch 1: Include x
        if can_add_bitset(x, selected):
            new_selected = selected | (1 << x)
            new_remaining = remaining & ~(1 << x)
            # Remove elements that would conflict
            # Actually, we just continue; can_add_bitset will handle conflicts
            branch_and_bound(new_remaining, new_selected, current_size + 1, best_size_ref, max_depth - 1)
        
        # Branch 2: Exclude x
        new_remaining = remaining & ~(1 << x)
        branch_and_bound(new_remaining, selected, current_size, best_size_ref, max_depth - 1)
    
    # Phase 1: Get a good initial solution
    best_set = 0
    best_size = 0
    
    # Multiple greedy starts
    for _ in range(20):
        selected = greedy_init()
        selected = local_search_bitset(selected, max_iters=30)
        size = bin(selected).count('1')
        if size > best_size:
            best_size = size
            best_set = selected
    
    # Phase 2: Branch and bound improvement
    if time.time() < deadline - 5:
        best_size_ref = [best_size]
        # Try to improve from best solution
        # Elements not in best_set
        remaining = ((1 << n) - 1) & ~best_set
        branch_and_bound(remaining, best_set, best_size, best_size_ref, max_depth=15)
        
        if best_size_ref[0] > best_size:
            best_size = best_size_ref[0]
    
    # Phase 3: More local search
    if time.time() < deadline - 2:
        improved = local_search_bitset(best_set, max_iters=100)
        size = bin(improved).count('1')
        if size > best_size:
            best_size = size
            best_set = improved
    
    # Convert bitset to sorted list (1-indexed)
    result = []
    for i in range(n):
        if (best_set >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)

# EVOLVE-BLOCK-END
