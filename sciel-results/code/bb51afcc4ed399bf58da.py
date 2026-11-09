import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s
    
    # Initial set
    A = set(range(1, 101))
    
    def find_subset_sum_200(elements):
        """
        Returns a list of elements summing to 200, or None if not possible.
        Uses DP to track one path.
        """
        target = 200
        # dp[s] = (parent_sum, element_used) or similar
        # Since we need to reconstruct, we store the element and the previous sum state
        # Better: store the actual subset? No, too big.
        # Store: for each sum s, store one element e and the sum s-e that led to it.
        # But we need to ensure e is in the current set A.
        
        # Initialize
        # reachable[0] = []
        # We want to find a subset.
        # Let's use a dictionary: sum -> (last_element, prev_sum)
        # But we need to know if the prev_sum was formed using distinct elements.
        # Standard subset sum DP assumes distinct elements if we iterate carefully.
        
        # Since N is small, we can do:
        # reachable = {0: []}
        # for x in elements:
        #    for s in list(reachable.keys()):
        #       if s + x <= target:
        #          reachable[s+x] = reachable[s] + [x]
        # If 200 in reachable, return reachable[200]
        
        # Optimization: We only care about sums <= 200.
        reachable = {0: []}
        # Sort elements to make it deterministic and maybe find smaller elements faster?
        # Actually, we want to find *any* subset.
        # To bias towards finding subsets with small elements (so we can remove them),
        # we can iterate elements in ascending order?
        # If we iterate 1, 2, 3..., and find a sum of 200, it might use 1.
        
        for x in sorted(elements):
            new_reachable = {}
            for s, subset in reachable.items():
                if s + x <= target:
                    new_reachable[s+x] = subset + [x]
            # Update reachable with new sums
            # Note: we should not overwrite existing sums with longer subsets?
            # We just need *one* subset.
            # Merging: if s+x is already in reachable, we can keep the old one or new one.
            # Keeping the one with smaller elements might be better for our heuristic?
            # Let's just keep the first one found or the one with smallest min element?
            # For simplicity, just update if not present, or overwrite.
            # Overwriting with later (larger) elements might lead to removing larger elements.
            # We want to find subsets containing small elements.
            # So if we iterate small to large, and we find a sum, it likely contains small elements.
            
            for s, subset in new_reachable.items():
                if s not in reachable:
                    reachable[s] = subset
                # If s is in reachable, we could compare.
                # Let's prefer subsets with smaller minimum element.
                elif min(subset) < min(reachable[s]):
                    reachable[s] = subset
                    
        if 200 in reachable:
            return reachable[200]
        return None

    while time.time() < deadline:
        subset = find_subset_sum_200(A)
        if subset is None:
            return list(A)
        
        # Heuristic: Remove the smallest element in the subset
        x_to_remove = min(subset)
        A.remove(x_to_remove)
        
    # Fallback: return current A
    return list(A)
