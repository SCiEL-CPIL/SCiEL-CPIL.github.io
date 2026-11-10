import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    S = []
    # K[k] is the set of subset sums of cardinality k
    K = [set() for _ in range(1)]
    K[0].add(0)
    UnionAll = {0}
    
    max_size = 60
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        
        m = len(S)
        
        # Check if x can be added
        valid = True
        new_K = [set() for _ in range(m + 2)]
        
        # Pre-compute new sums for each cardinality
        # We need to check if {s+x for s in K[k]} intersects K[j] for j != k+1
        
        # Optimization: Check conflicts
        # For each k, NewSums_k = K[k] + x
        # We need NewSums_k to be disjoint from UnionAll \ K[k+1]
        
        for k in range(m + 1):
            if not K[k]:
                continue
                
            # Calculate new sums
            # If x is large, sums are large.
            new_sums = {s + x for s in K[k]}
            
            # The target cardinality for these new sums is k+1
            # They must not collide with sums of any other cardinality j != k+1
            # The set of all sums of other cardinalities is UnionAll - K[k+1]
            # Note: K[k+1] might not exist yet if k+1 > m, but in our loop k goes to m, 
            # so k+1 goes to m+1. K has size m+1 (indices 0..m).
            # So if k+1 == m+1, K[k+1] is empty.
            
            other_sums = UnionAll - K[k+1] if k+1 <= m else UnionAll
            
            if new_sums & other_sums:
                valid = False
                break
        
        if valid:
            # Update K and UnionAll
            S.append(x)
            
            new_K = [set() for _ in range(m + 2)]
            new_K[0] = {0}
            
            # Re-construct K to be safe, or update in place
            # Since we need to add x to all subsets, 
            # New K[k] = Old K[k] (subsets not using x)
            # New K[k+1] = Old K[k+1] (subsets of size k+1 not using x) U {s+x | s in Old K[k]}
            
            # Actually, simpler:
            # NewK[k] = K[k]
            # NewK[k+1] = K[k+1] U {s+x for s in K[k]}
            
            # But we already computed new_sums in the check loop? 
            # We broke early if invalid. If valid, we need to recompute or store them.
            # Let's recompute for clarity and correctness in the update step, 
            # or store them in a list during the check phase if valid.
            
            # To avoid double work, let's store new_sums if we didn't break.
            # But the loop breaks on first conflict. 
            # So we must recompute if valid.
            
            for k in range(m + 1):
                new_K[k] = K[k]
                
            for k in range(m + 1):
                if K[k]:
                    new_sums = {s + x for s in K[k]}
                    if k + 1 < len(new_K):
                        new_K[k+1] = new_K[k+1] | new_sums
                        # Also update UnionAll? 
                        # UnionAll_new = UnionAll U new_sums
                        # We can just do UnionAll |= new_sums after the loop
            
            K = new_K
            # Update UnionAll
            # We can just recompute UnionAll = set().union(*K) 
            # Or incrementally.
            # Incremental is faster.
            # But we need to know which new sums were added.
            # Let's just do UnionAll = set().union(*K) for simplicity, 
            # as m is small (<=60) and sets are small.
            # Wait, 2^60 is huge. But the *distinct* sums are limited by the range.
            # Max sum is 60 * 256 = 15360.
            # So UnionAll is at most 15360 elements.
            # Union of 61 sets of size <= 15360 is fast.
            
            UnionAll = set()
            for s in K:
                UnionAll.update(s)
            
            if len(S) >= max_size:
                break
                
    return S
