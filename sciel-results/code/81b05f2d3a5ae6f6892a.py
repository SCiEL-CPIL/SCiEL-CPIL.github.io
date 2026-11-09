import time
import random
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s
    
    # Initial set
    A = set(range(1, 101))
    
    def find_subset_sum_200(elements):
        """
        Returns a list of elements summing to 200, or None if not possible.
        Uses DP to track one path.
        """
        target = 200
        reachable = {0: []}
        
        for x in sorted(elements):
            new_reachable = {}
            for s, subset in reachable.items():
                if s + x <= target:
                    new_reachable[s+x] = subset + [x]
            
            for s, subset in new_reachable.items():
                if s not in reachable:
                    reachable[s] = subset
                elif min(subset) < min(reachable[s]):
                    reachable[s] = subset
                    
        if 200 in reachable:
            return reachable[200]
        return None

    def has_subset_sum_200(elements):
        """
        Returns True if any subset of elements sums to 200.
        """
        target = 200
        reachable = {0}
        for x in sorted(elements):
            new_reachable = set()
            for s in reachable:
                if s + x <= target:
                    new_reachable.add(s + x)
            reachable.update(new_reachable)
        return 200 in reachable

    # Greedy approach: Add numbers in an order that minimizes conflicts
    # We want to maximize cardinality, so we want to keep as many as possible.
    # Strategy: Start with empty set, add elements one by one.
    # If adding an element creates a subset sum of 200, skip it (or remove something else?).
    # Better: Start with all elements, and remove those that are "critical" for forming 200.
    # But the previous approach was removal. Let's try addition to see if we can get a larger set.
    # Actually, the removal approach is usually better for "max independent set" type problems if we remove the "most connected" nodes.
    
    # Let's try a different heuristic: 
    # 1. Find all minimal subsets that sum to 200.
    # 2. This is hard.
    
    # Let's stick to the removal strategy but improve the choice of element to remove.
    # Instead of always removing the min, maybe remove the element that is in the most subsets summing to 200?
    # Or the element that is "hardest" to replace?
    
    # Alternative: 
    # Iterate through elements in descending order.
    # Try to include each element. If including it makes a subset sum of 200 possible, 
    # then we have a conflict. We need to resolve it.
    # This is like a greedy set packing or independent set.
    
    # Let's try: 
    # 1. Start with A = [1..100]
    # 2. While there is a subset summing to 200:
    #    - Find such a subset S.
    #    - Remove the element in S that appears in the most other subsets summing to 200? 
    #      This is expensive to compute.
    #    - Or remove the largest element in S? (Leaves smaller elements which might be more flexible?)
    #    - Or remove the smallest element in S? (Current strategy)
    
    # Let's experiment with removing the LARGEST element in the conflicting subset.
    # Why? Because large elements are less likely to be part of other sums? 
    # Actually, small elements are more likely to be part of many sums. 
    # So removing a small element might break more potential sums. 
    # But we want to KEEP as many elements as possible. 
    # If we remove a small element, we lose it. 
    # If we remove a large element, we lose it. 
    # But large elements have fewer partners to sum to 200. 
    # So large elements are "less central" in the conflict graph? 
    # If we remove the most central node, we break the most conflicts. 
    # Small elements are more central. 
    # So removing small elements might be better to break conflicts quickly? 
    # But we want to maximize the final set size. 
    # Removing a small element (1) is "cheaper" in terms of value, but we care about cardinality. 
    # So removing ANY element reduces cardinality by 1. 
    # We want to remove the element that allows us to keep the most OTHER elements. 
    # This is complex.
    
    # Let's try: Remove the element in the subset that has the highest value.
    # Hypothesis: Large elements are less versatile, so removing them might allow more small elements to stay.
    
    while time.time() < deadline:
        subset = find_subset_sum_200(A)
        if subset is None:
            return list(A)
        
        # Heuristic: Remove the LARGEST element in the subset
        x_to_remove = max(subset)
        A.remove(x_to_remove)
        
    # Fallback: return current A
    return list(A)
