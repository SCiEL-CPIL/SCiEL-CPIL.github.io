"""B2[2] set construction: greedy + local search to maximize size within 1..1000."""
import time
import random
from collections import defaultdict

def _count_sums(arr):
    """Count all pairwise sums a+b with a<=b from arr. Returns dict sum->count."""
    counts = defaultdict(int)
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i, n):
            b = arr[j]
            counts[a + b] += 1
    return counts

def _max_count_from_counts(counts):
    """Return the maximum count in the counts dict."""
    if not counts:
        return 0
    return max(counts.values())

def _greedy_construct():
    """Greedy: add x if it doesn't cause any sum to exceed count 2."""
    chosen = []
    counts = defaultdict(int)
    for x in range(1, 1001):
        # Compute new sums introduced by x
        new_sums = []
        for y in chosen:
            new_sums.append(x + y)
        new_sums.append(2 * x)
        
        # Check if any new sum would exceed 2
        ok = True
        for s in new_sums:
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            for s in new_sums:
                counts[s] += 1
    return chosen

def _try_remove_and_add(chosen, counts, candidates_to_try):
    """Try to remove one element and add two others to increase size by 1.
    Returns (new_chosen, new_counts) if successful, else None."""
    n = len(chosen)
    chosen_set = set(chosen)
    
    for i in range(n):
        removed = chosen[i]
        # Compute sums that lose one count
        lost_sums = defaultdict(int)
        for j in range(n):
            if j != i:
                s = removed + chosen[j]
                lost_sums[s] += 1
        lost_sums[2 * removed] += 1
        
        # Build temporary counts
        temp_counts = dict(counts)
        for s, c in lost_sums.items():
            temp_counts[s] -= c
            if temp_counts[s] <= 0:
                del temp_counts[s]
        
        # Try to add two new elements
        remaining = [x for x in candidates_to_try if x not in chosen_set and x != removed]
        remaining.sort()
        
        for idx_a in range(len(remaining)):
            a = remaining[idx_a]
            # Check if a can be added
            new_sums_a = [a + y for y in chosen if y != removed] + [2 * a]
            # Also need to check sums with other chosen elements
            ok_a = True
            for s in new_sums_a:
                if temp_counts.get(s, 0) + 1 > 2:
                    ok_a = False
                    break
            if not ok_a:
                continue
            
            # Tentatively add a
            temp2 = dict(temp_counts)
            for s in new_sums_a:
                temp2[s] = temp2.get(s, 0) + 1
            
            # Now try to add b > a
            for idx_b in range(idx_a + 1, len(remaining)):
                b = remaining[idx_b]
                new_sums_b = [b + y for y in chosen if y != removed] + [2 * b]
                # Also b+a
                new_sums_b.append(a + b)
                
                ok_b = True
                for s in new_sums_b:
                    if temp2.get(s, 0) + 1 > 2:
                        ok_b = False
                        break
                if not ok_b:
                    continue
                
                # Success! Build new set
                new_chosen = [x for x in chosen if x != removed]
                new_chosen.append(a)
                new_chosen.append(b)
                new_chosen.sort()
                
                # Build new counts
                new_counts = dict(temp2)
                for s in new_sums_b:
                    new_counts[s] = new_counts.get(s, 0) + 1
                
                return new_chosen, new_counts
    
    return None

def _local_search(chosen, counts, deadline, max_iterations=50000):
    """Local search: repeatedly try to swap one element for two others."""
    chosen_set = set(chosen)
    candidates = [x for x in range(1, 1001) if x not in chosen_set]
    
    iterations = 0
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        result = _try_remove_and_add(chosen, counts, candidates)
        if result is None:
            break
        chosen, counts = result
        chosen_set = set(chosen)
        candidates = [x for x in range(1, 1001) if x not in chosen_set]
    
    return chosen, counts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Phase 1: Greedy construction
    chosen = _greedy_construct()
    counts = _count_sums(chosen)
    
    # Phase 2: Local search to improve
    if time.time() < deadline:
        chosen, counts = _local_search(chosen, counts, deadline)
    
    return chosen

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    # Verify
    c = _count_sums(result)
    max_c = max(c.values()) if c else 0
    print(f"Max representation count: {max_c}")
    assert max_c <= 2, "Invalid B2[2] set!"
