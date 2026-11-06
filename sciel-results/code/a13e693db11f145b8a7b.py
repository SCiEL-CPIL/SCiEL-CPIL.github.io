import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin for 90s limit
    
    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # C(16,4) = 1820
    
    best_family = []
    best_size = 0
    
    # Precompute: for each pair of masks, their intersection
    # This is expensive (1820^2 / 2 ~ 1.6M pairs), but we can do it lazily
    # Actually, let's just compute on the fly with caching
    
    intersection_cache = {}
    
    def get_intersection(a, b):
        key = (min(a, b), max(a, b))
        if key not in intersection_cache:
            intersection_cache[key] = a & b
        return intersection_cache[key]
    
    # Better approach: maintain for each mask in the family, 
    # a dictionary mapping intersection value -> count of other masks with that intersection
    # When adding new mask x, we need to check if there exist two masks a, b in family
    # such that a&x == b&x == a&b
    
    # Even better: for each mask x, maintain a dict: for each possible intersection s,
    # how many masks in the family have intersection s with x?
    # But we need to check if two masks a, b both have intersection s with x AND a&b == s
    
    # Let's use a different approach:
    # For a candidate x, group existing family members by (a & x)
    # For each group with intersection value s and at least 2 members,
    # check if any pair in that group has intersection exactly s
    
    def can_add(x, family):
        """Check if x can be added to family without creating a sunflower."""
        # Group family members by their intersection with x
        groups = {}
        for a in family:
            s = a & x
            if s in groups:
                groups[s].append(a)
            else:
                groups[s] = [a]
        
        for s, members in groups.items():
            if len(members) < 2:
                continue
            # Check if any pair in members has intersection exactly s
            m = len(members)
            for i in range(m):
                ai = members[i]
                for j in range(i + 1, m):
                    aj = members[j]
                    if (ai & aj) == s:
                        return False
        return True
    
    # Random greedy with multiple restarts
    while time.time() < deadline:
        shuffled = list(range(n_total))
        random.shuffle(shuffled)
        
        family = []
        for idx in shuffled:
            x = all_masks[idx]
            if can_add(x, family):
                family.append(x)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local improvement: try to add more sets by removing one and adding two
    # This is a 1-move local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_family = best_family[:]
        current_size = len(current_family)
        
        # Try each pair of removals and see if we can add more
        for i in range(len(current_family)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current_family)):
                if time.time() > deadline:
                    break
                # Remove current_family[i] and current_family[j]
                reduced = current_family[:i] + current_family[i+1:j] + current_family[j+1:]
                
                # Try to add back some sets
                added_any = False
                for k in range(n_total):
                    if k == i or k == j:
                        continue
                    x = all_masks[k]
                    if x in reduced:
                        continue
                    if can_add(x, reduced):
                        reduced.append(x)
                        added_any = True
                        if len(reduced) > current_size:
                            improved = True
                            best_family = reduced[:]
                            best_size = len(reduced)
                            current_family = reduced[:]
                            current_size = len(reduced)
                            break
                
                if improved:
                    break
            if improved:
                break
    
    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for m in best_family:
        elem = []
        for i in range(16):
            if m & (1 << i):
                elem.append(i + 1)
        result.append(sorted(elem))
        
    return result

if __name__ == "__main__":
    # Just for testing, not part of the evaluation call
    res = construct()
    print(len(res))
