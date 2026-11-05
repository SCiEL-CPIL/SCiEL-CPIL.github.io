import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Precompute conflicts
    # conflict[i] is a set of indices j such that candidates[i] and candidates[j] conflict
    # Conflict: lcm(a, b) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            g = gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def is_compatible(val_idx, S_idx_set):
        """Check if val_idx is compatible with all indices in S_idx_set"""
        for j in S_idx_set:
            if j in conflict[val_idx]:
                return False
        return True
        
    def get_sum(idx_set):
        return sum(1.0 / candidates[i] for i in idx_set)

    # Initial Solution: Greedy by weight (smallest x first)
    # Sort indices by weight descending (value ascending)
    sorted_indices = sorted(range(n), key=lambda i: candidates[i])
    
    S = set()
    for i in sorted_indices:
        if is_compatible(i, S):
            S.add(i)
            
    best_S = set(S)
    best_sum = get_sum(S)
    
    # Local Search
    # We maintain S as a set of indices
    improved = True
    while improved:
        improved = False
        
        # 1. Try adding any unselected node
        # To speed up, we can just iterate all unselected
        unselected = [i for i in range(n) if i not in S]
        for i in unselected:
            if is_compatible(i, S):
                S.add(i)
                improved = True
                # Restart loop to check for more additions
                break
        
        if improved:
            continue
            
        # 2. Try 1-swap: remove u, add v
        for u in list(S):
            S.remove(u)
            # Find best v to add? Or just check if any v improves?
            # We want 1/v > 1/u => v < u
            # But v must be compatible with S (which no longer has u)
            # And v must not be in S
            best_v = None
            best_w = 0.0
            for v in unselected:
                if v in S: continue
                if is_compatible(v, S):
                    w = 1.0 / candidates[v]
                    if w > best_w:
                        best_w = w
                        best_v = v
            
            if best_v is not None and best_w > 1.0 / candidates[u]:
                S.add(best_v)
                improved = True
                # Update unselected list? It's expensive. 
                # Since we break and restart, we recompute unselected next time? 
                # No, unselected is computed outside the loop. 
                # We should recompute unselected if we modify S.
                # To keep it simple and safe, we recompute unselected at the start of each outer loop iteration?
                # That's O(N^2). N=255. 255^2 ~ 65k ops. Very fast.
                # But we are inside a while loop. 
                # Let's restructure: recompute unselected at the top of the while loop.
                break
            else:
                # Re-add u if we didn't swap
                S.add(u)
        
        if improved:
            continue
            
        # 3. Try 2-swap: remove u, add v, w
        # This is more expensive.
        # We can limit this to cases where 1-swap didn't work.
        # For each u in S, remove it.
        # Find all v, w compatible with S\{u} and compatible with each other.
        # Maximize 1/v + 1/w subject to 1/v + 1/w > 1/u.
        
        # Optimization: Only try 2-swap if potential gain exists.
        # Max possible gain from 2 nodes is 1/2 + 1/3 = 0.833.
        # Min cost is 1/256 ~ 0.004.
        
        for u in list(S):
            S.remove(u)
            # Get compatible candidates
            comp_v = []
            for v in unselected:
                if v in S: continue
                if is_compatible(v, S):
                    comp_v.append(v)
            
            # Find best pair v, w in comp_v
            # Sort comp_v by weight desc to find best pair quickly?
            # We just need to find if ANY pair improves.
            # To be efficient, sort by weight desc.
            comp_v.sort(key=lambda i: candidates[i])
            
            found = False
            # Check pairs
            # Since we want max sum, check smallest indices first
            for i in range(len(comp_v)):
                v = comp_v[i]
                for j in range(i + 1, len(comp_v)):
                    w = comp_v[j]
                    if w in conflict[v]:
                        continue
                    gain = (1.0 / candidates[v]) + (1.0 / candidates[w])
                    cost = 1.0 / candidates[u]
                    if gain > cost:
                        S.add(v)
                        S.add(w)
                        found = True
                        break
                if found:
                    break
            
            if found:
                improved = True
                break
            else:
                S.add(u)
        
        # Update best
        current_sum = get_sum(S)
        if current_sum > best_sum:
            best_sum = current_sum
            best_S = set(S)
            
        # Time check
        if time.time() > deadline:
            break
            
    # Convert best_S back to values
    result = [candidates[i] for i in best_S]
    if not result:
        # Fallback: just return a single valid element, e.g., 256
        return [256]
        
    return result

# Test helper
if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            g = gcd(a, b)
            l = (a//g)*b
            if l <= 256:
                valid = False
    print("Valid:", valid)
    print("Sum:", sum(1.0/x for x in res))
