"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 512
    
    # Precompute all 4-term APs and build conflict index
    # For each x, store the list of triples (y,z,w) such that {x,y,z,w} is a 4-AP
    ap_triples = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            ap_triples[a].append((b, c, e))
            ap_triples[b].append((a, c, e))
            ap_triples[c].append((a, b, e))
            ap_triples[e].append((a, b, c))
    
    # Use bitmask representation for fast membership checks
    # Python integers can serve as arbitrary-precision bitmasks
    # bit i represents whether element (i+1) is in the set (0-indexed: bit 0 = element 1)
    
    def can_add_bitmask(x, mask):
        """Check if adding x (1-indexed) to the set represented by mask creates a 4-AP."""
        # x is 1-indexed, so bit position is x-1
        for (y, z, w) in ap_triples[x]:
            if (mask >> (y - 1)) & 1 and (mask >> (z - 1)) & 1 and (mask >> (w - 1)) & 1:
                return False
        return True
    
    def greedy_construct(order, mask=0, count=0):
        """Greedy construction using bitmask. Returns (mask, count)."""
        for x in order:
            if (mask >> (x - 1)) & 1:
                continue
            if can_add_bitmask(x, mask):
                mask |= (1 << (x - 1))
                count += 1
        return mask, count
    
    def local_search(mask, count):
        """Try to improve the set via swaps: remove one element, try to add one or two others."""
        best_mask = mask
        best_count = count
        
        # Get list of elements in the set and not in the set
        in_set = [i for i in range(1, N + 1) if (mask >> (i - 1)) & 1]
        not_in_set = [i for i in range(1, N + 1) if not ((mask >> (i - 1)) & 1)]
        
        # Try single swaps: remove x, add y (net 0 change, but may enable further additions)
        # Better: try remove x, then greedily add as many as possible from not_in_set
        
        # Strategy: for each element x in the set, remove it, then try to add elements
        # from not_in_set in a greedy fashion, keeping the best improvement.
        
        # To be efficient, we'll try a limited number of removals
        random.shuffle(in_set)
        random.shuffle(not_in_set)
        
        # Try removing up to min(len(in_set), 200) elements
        max_removals = min(len(in_set), 200)
        
        for x in in_set[:max_removals]:
            if time.time() > deadline - 5:
                break
            
            # Remove x
            temp_mask = mask & ~(1 << (x - 1))
            temp_count = count - 1
            
            # Try to add elements from not_in_set (excluding x if it was there)
            added = 0
            for y in not_in_set:
                if y == x:
                    continue
                if can_add_bitmask(y, temp_mask):
                    temp_mask |= (1 << (y - 1))
                    temp_count += 1
                    added += 1
            
            if temp_count > best_count:
                best_count = temp_count
                best_mask = temp_mask
        
        return best_mask, best_count
    
    def local_search_double_swap(mask, count):
        """Try removing two elements and adding three (net +1)."""
        best_mask = mask
        best_count = count
        
        in_set = [i for i in range(1, N + 1) if (mask >> (i - 1)) & 1]
        not_in_set = [i for i in range(1, N + 1) if not ((mask >> (i - 1)) & 1)]
        
        random.shuffle(in_set)
        random.shuffle(not_in_set)
        
        # Try pairs of removals - limit to avoid O(n^2)
        max_pairs = min(len(in_set), 100)
        
        for i in range(max_pairs):
            if time.time() > deadline - 5:
                break
            x1 = in_set[i]
            for j in range(i + 1, min(i + 10, max_pairs)):  # limit inner loop
                x2 = in_set[j]
                
                temp_mask = mask & ~(1 << (x1 - 1)) & ~(1 << (x2 - 1))
                temp_count = count - 2
                
                # Try to add elements
                added = 0
                for y in not_in_set:
                    if y == x1 or y == x2:
                        continue
                    if can_add_bitmask(y, temp_mask):
                        temp_mask |= (1 << (y - 1))
                        temp_count += 1
                        added += 1
                
                if temp_count > best_count:
                    best_count = temp_count
                    best_mask = temp_mask
        
        return best_mask, best_count
    
    best_mask = 0
    best_count = 0
    
    # Phase 1: Multiple greedy restarts with different orderings
    random.seed(42)
    
    # Natural order
    order = list(range(1, N + 1))
    m, c = greedy_construct(order)
    if c > best_count:
        best_count = c
        best_mask = m
    
    # Conflict-count based ordering
    conflict_count = [len(ap_triples[i]) for i in range(N + 1)]
    
    # Ascending conflict count (fewest conflicts first)
    order = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
    m, c = greedy_construct(order)
    if c > best_count:
        best_count = c
        best_mask = m
    
    # Descending conflict count
    order = sorted(range(1, N + 1), key=lambda x: -conflict_count[x])
    m, c = greedy_construct(order)
    if c > best_count:
        best_count = c
        best_mask = m
    
    # Random orders
    num_random_trials = 300
    for _ in range(num_random_trials):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        m, c = greedy_construct(order)
        if c > best_count:
            best_count = c
            best_mask = m
    
    # Phase 2: Local search improvements on the best solution
    if time.time() < deadline - 10:
        best_mask, best_count = local_search(best_mask, best_count)
    
    if time.time() < deadline - 10:
        best_mask, best_count = local_search_double_swap(best_mask, best_count)
    
    # Phase 3: More random restarts with local search
    for _ in range(200):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        m, c = greedy_construct(order)
        
        # Apply local search to this candidate
        m2, c2 = local_search(m, c)
        
        if c2 > best_count:
            best_count = c2
            best_mask = m2
    
    # Phase 4: Final polish - try more local search iterations on the best
    if time.time() < deadline - 5:
        best_mask, best_count = local_search(best_mask, best_count)
        best_mask, best_count = local_search_double_swap(best_mask, best_count)
    
    # Convert bitmask to list
    result = [i for i in range(1, N + 1) if (best_mask >> (i - 1)) & 1]
    
    # Verify no 4-AP exists (safety check)
    # This is optional but ensures correctness
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            if ((best_mask >> (a - 1)) & 1 and 
                (best_mask >> (b - 1)) & 1 and 
                (best_mask >> (c - 1)) & 1 and 
                (best_mask >> (e - 1)) & 1):
                # Found a 4-AP, which should not happen
                # Remove one element to fix
                best_mask &= ~(1 << (e - 1))
                break
    
    result = [i for i in range(1, N + 1) if (best_mask >> (i - 1)) & 1]
    
    return result

# EVOLVE-BLOCK-END
