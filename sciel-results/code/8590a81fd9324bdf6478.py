import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    best_set = []
    best_size = 0
    
    # Pre-compute or generate candidates?
    # We will try to build a set from scratch multiple times.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        current_set = []
        # A_k stores sums of subsets of size k
        A = [set() for _ in range(61)]
        A[0] = {0}
        forbidden = {0}
        
        # Candidates: 1..256
        # We can shuffle candidates or pick randomly.
        # Picking smaller numbers first might be better for packing?
        # Or larger?
        # Let's try random selection from remaining.
        
        remaining = list(range(1, 257))
        random.shuffle(remaining)
        
        valid = True
        while remaining:
            if time.time() > deadline:
                break
                
            # Try to find a valid x from remaining
            # Instead of checking all, we can check a few random ones.
            # Or check them in order.
            # Let's check a random subset of candidates to speed up.
            
            # Optimization: If we just iterate through remaining, it's O(N * 2^n).
            # N=256, 2^n can be large.
            
            # Let's try to pick a random x from remaining.
            # If valid, add it. If not, discard it? 
            # Discarding might be bad.
            # Better: Backtrack?
            # Simple heuristic: If a number fails, remove it from consideration for this restart.
            
            found = False
            # Try a few random candidates
            for _ in range(10):
                if not remaining:
                    break
                idx = random.randint(0, len(remaining) - 1)
                x = remaining[idx]
                
                # Check validity of adding x
                # Generate new sums
                new_sums = set()
                ok = True
                n = len(current_set)
                
                # k goes from 1 to n+1
                # A[k-1] contains sums of size k-1
                for k in range(1, n + 2):
                    prev_sums = A[k-1]
                    if not prev_sums:
                        continue
                    # new sums are s + x for s in prev_sums
                    # Check intersection with forbidden and new_sums
                    # To speed up, we can iterate and check membership
                    for s in prev_sums:
                        ns = s + x
                        if ns in forbidden or ns in new_sums:
                            ok = False
                            break
                        new_sums.add(ns)
                    if not ok:
                        break
                
                if ok:
                    # Update state
                    current_set.append(x)
                    remaining.pop(idx)
                    
                    # Update A and forbidden
                    # We need to update A[k] for k=1..n+1
                    # A[k] = A[k] union {s+x | s in A[k-1]}
                    # Note: We calculated new_sums in the check, but we lost the mapping to k.
                    # We should re-calculate or store it.
                    # For simplicity, re-calculate or store in list.
                    
                    new_A = [set(s) for s in A] # Copy
                    for k in range(1, n + 2):
                        prev_sums = A[k-1]
                        for s in prev_sums:
                            ns = s + x
                            new_A[k].add(ns)
                            forbidden.add(ns)
                    A = new_A
                    found = True
                    break
            if not found:
                break
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set
