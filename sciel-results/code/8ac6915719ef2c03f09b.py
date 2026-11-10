import time
import sys

def construct():
    n = 256
    start_time = time.time()
    timeout = 55.0 # Leave 5s for overhead
    
    # Precompute 4-APs
    four_aps = []
    for d in range(1, (n-1)//3 + 1):
        for a in range(1, n - 3*d + 1):
            four_aps.append((a, a+d, a+2*d, a+3*d))
    
    # Map vertex to list of conflict masks
    # A conflict mask for vertex v in AP (v, u1, u2, u3) is (1<<u1) | (1<<u2) | (1<<u3)
    # Note: vertices are 1-indexed in problem, 0-indexed in bitmask?
    # Let's use 0-indexed internally for bitmask: v-1.
    conflicts = [[] for _ in range(n)]
    for ap in four_aps:
        v0, v1, v2, v3 = ap
        # For v0, others are v1, v2, v3
        mask0 = (1 << (v1-1)) | (1 << (v2-1)) | (1 << (v3-1))
        conflicts[v0-1].append(mask0)
        
        # For v1, others are v0, v2, v3
        mask1 = (1 << (v0-1)) | (1 << (v2-1)) | (1 << (v3-1))
        conflicts[v1-1].append(mask1)
        
        # For v2, others are v0, v1, v3
        mask2 = (1 << (v0-1)) | (1 << (v1-1)) | (1 << (v3-1))
        conflicts[v2-1].append(mask2)
        
        # For v3, others are v0, v1, v2
        mask3 = (1 << (v0-1)) | (1 << (v1-1)) | (1 << (v2-1))
        conflicts[v3-1].append(mask3)
        
    # Weights: number of 3-APs vertex participates in
    weights = [0] * n
    for x in range(1, n+1):
        idx = x - 1
        w = 0
        w += (n - x) // 2
        w += min(x - 1, n - x)
        w += (x - 1) // 2
        weights[idx] = w
        
    # Sort vertices by weight descending
    sorted_vertices = sorted(range(n), key=lambda i: -weights[i])
    # Map sorted index to original index? 
    # We iterate over sorted_vertices in the B&B.
    # included_mask uses original indices (0 to n-1).
    
    best_true_count = 0
    best_set = []
    best_linear = 0
    
    sys.setrecursionlimit(10000)
    
    # To speed up bound calculation, we can precompute suffix sums of weights
    # but the dynamic blocking makes it hard to use simple suffix sums.
    # We will just loop.
    
    def count_3aps(mask):
        # Efficient count using bit operations?
        # Iterate over set bits
        count = 0
        # For each a in A
        # For each d
        # This is O(|A|^2) roughly.
        # For n=256, |A| is likely < 100. 100^2 = 10000 ops.
        # Number of leaves might be large.
        # This is the bottleneck.
        # Optimization: Precompute 3-AP masks?
        # A 3-AP is (a, a+d, a+2d).
        # Count how many such triples are fully in mask.
        # Precompute list of 3-AP masks.
        pass
        
    # Precompute 3-AP masks
    three_aps_masks = []
    for d in range(1, (n-1)//2 + 1):
        for a in range(1, n - 2*d + 1):
            m = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1))
            three_aps_masks.append(m)
            
    def count_3aps_fast(mask):
        c = 0
        for m in three_aps_masks:
            if (mask & m) == m:
                c += 1
        return c
        
    # There are ~ O(n^2) 3-APs. ~ 256^2 / 2 = 32k.
    # Checking 32k masks per leaf is expensive if there are many leaves.
    # But B&B should prune most.
    
    def branch(idx, included_mask, linear_weight):
        nonlocal best_true_count, best_set, best_linear
        
        if time.time() - start_time > timeout:
            return
            
        # Upper Bound Calculation
        ub = linear_weight
        for i in range(idx, n):
            v = sorted_vertices[i]
            # Check if v is blocked
            blocked = False
            for mask in conflicts[v]:
                if (included_mask & mask) == mask:
                    blocked = True
                    break
            if not blocked:
                ub += weights[v]
            if ub <= best_linear:
                break
                
        if ub <= best_linear:
            return
            
        if idx == n:
            true_count = count_3aps_fast(included_mask)
            if true_count > best_true_count:
                best_true_count = true_count
                best_linear = linear_weight # Update pruning threshold
                # Reconstruct set
                s = []
                for i in range(n):
                    if (included_mask >> i) & 1:
                        s.append(i + 1)
                best_set = s
            return
            
        v = sorted_vertices[idx]
        
        # Branch 1: Exclude v
        branch(idx + 1, included_mask, linear_weight)
        
        # Branch 2: Include v
        # Check validity
        valid = True
        for mask in conflicts[v]:
            if (included_mask & mask) == mask:
                valid = False
                break
                
        if valid:
            branch(idx + 1, included_mask | (1 << v), linear_weight + weights[v])
            
    branch(0, 0, 0)
    
    if not best_set:
        # Fallback if B&B fails or finds nothing
        # Greedy
        selected = set()
        for v in sorted_vertices:
            valid = True
            for mask in conflicts[v]:
                if (selected_mask & mask) == mask: # Need selected_mask
                    valid = False
                    break
            # This fallback is messy. Just return a valid set if empty.
            # But B&B should always find at least one set (empty set is valid, count 0).
            # If best_set is empty, it means we didn't update? 
            # We initialize best_true_count=0. Empty set has count 0.
            # So best_set will be updated at the first leaf (idx=n, mask=0).
            pass
            
    return best_set
