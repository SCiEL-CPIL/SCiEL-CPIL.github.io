"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Precompute all conflict hyperedges:
    # A hyperedge is a set {a} ∪ T where T ⊆ [1,24]\{a}, T nonempty,
    # and 1/a = sum_{b in T} 1/b.
    # If a solution contains all elements of a hyperedge, it's invalid.
    # We want the maximum independent set in this hypergraph.
    
    all_elements = list(range(1, 25))
    
    # Precompute all hyperedges
    hyperedges = set()
    
    for a in all_elements:
        target = Fraction(1, a)
        others = [b for b in all_elements if b != a]
        # Find all subsets T of others such that sum(1/b for b in T) = 1/a
        # Use DFS with pruning
        others_sorted = sorted(others, reverse=True)
        
        def find_subsets(idx, current_sum, current_set):
            if time.time() > deadline - 5:
                return
            if current_sum == target:
                if current_set:
                    hyperedges.add(frozenset([a] + list(current_set)))
                return
            if current_sum > target:
                return
            if idx >= len(others_sorted):
                return
            
            # Option 1: skip others_sorted[idx]
            find_subsets(idx + 1, current_sum, current_set)
            
            # Option 2: include others_sorted[idx]
            new_sum = current_sum + Fraction(1, others_sorted[idx])
            if new_sum <= target:
                current_set.append(others_sorted[idx])
                find_subsets(idx + 1, new_sum, current_set)
                current_set.pop()
        
        find_subsets(0, Fraction(0), [])
    
    # Convert to list for easier access
    hyperedge_list = list(hyperedges)
    
    # For each element, precompute which hyperedges contain it
    elem_to_hedges = {e: [] for e in all_elements}
    for i, he in enumerate(hyperedge_list):
        for e in he:
            elem_to_hedges[e].append(i)
    
    # Also precompute: for each pair of elements, which hyperedges contain both
    # This helps with fast conflict checking
    # Actually, let's just check if adding an element creates a conflict by checking
    # if any hyperedge containing that element is fully contained in the current set + {elem}
    
    def is_conflict(elem, current_set):
        """Check if adding elem to current_set creates a conflict."""
        for he_idx in elem_to_hedges[elem]:
            he = hyperedge_list[he_idx]
            # Check if all elements in he except elem are in current_set
            # (elem itself is being added, so we check the rest)
            conflict = True
            for e in he:
                if e != elem and e not in current_set:
                    conflict = False
                    break
            if conflict:
                return True
        return False
    
    # Branch and bound search
    # Order elements by "danger" - elements that appear in more hyperedges are more dangerous
    # Actually, we want to process elements that are LESS dangerous first (safer to include)
    # Or we can use a different strategy
    
    # Let's use a backtracking approach with good pruning
    # Process elements in descending order (larger numbers are generally safer)
    search_order = sorted(all_elements, reverse=True)
    
    best_set = set()
    best_size = 0
    
    def backtrack(idx, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 3:
            return
        
        # Prune: if we can't beat best
        remaining_count = len(search_order) - idx
        if current_size + remaining_count <= best_size:
            return
        
        if idx == len(search_order):
            if current_size > best_size:
                best_set = set(current_set)
                best_size = current_size
            return
        
        elem = search_order[idx]
        
        # Try including elem
        if not is_conflict(elem, current_set):
            current_set.add(elem)
            backtrack(idx + 1, current_set, current_size + 1)
            current_set.remove(elem)
        
        # Try excluding elem
        backtrack(idx + 1, current_set, current_size)
    
    backtrack(0, set(), 0)
    
    # Local search: try to add remaining elements
    best_list = list(best_set)
    remaining = [x for x in all_elements if x not in best_set]
    
    improved = True
    while improved:
        improved = False
        for elem in remaining[:]:
            if time.time() > deadline - 5:
                break
            if not is_conflict(elem, best_set):
                best_list.append(elem)
                best_set.add(elem)
                remaining.remove(elem)
                improved = True
    
    # Try swap-based improvement: remove one, add two
    for a in list(best_set):
        if time.time() > deadline - 5:
            break
        reduced = best_set - {a}
        for b in all_elements:
            if time.time() > deadline - 5:
                break
            if b in reduced:
                continue
            for c in all_elements:
                if time.time() > deadline - 5:
                    break
                if c <= b or c in reduced:
                    continue
                candidate = reduced | {b, c}
                if len(candidate) > len(best_set):
                    # Check if candidate is valid (no hyperedge fully contained)
                    valid = True
                    for he in hyperedge_list:
                        if he.issubset(candidate):
                            valid = False
                            break
                    if valid:
                        best_set = candidate
                        best_list = list(best_set)
                        break
        if time.time() > deadline - 5:
            break
    
    # Ensure nonempty
    if not best_list:
        best_list = [24]
    
    return best_list

# EVOLVE-BLOCK-END
