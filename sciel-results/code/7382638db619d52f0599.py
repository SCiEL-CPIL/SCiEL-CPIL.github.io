import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin against 90s limit
    
    # Nodes are 2..256
    N = 256
    nodes = list(range(2, N + 1))
    n_vars = len(nodes)
    
    # Precompute conflicts
    # conflict[i] is a set of indices j such that lcm(nodes[i], nodes[j]) <= 256
    # We only need to check i < j for construction, but sets are symmetric.
    conflict = [set() for _ in range(n_vars)]
    
    for i in range(n_vars):
        a = nodes[i]
        for j in range(i + 1, n_vars):
            b = nodes[j]
            # lcm(a, b) <= 256  <=>  a * b <= 256 * gcd(a, b)
            if a * b <= 256 * gcd(a, b):
                conflict[i].add(j)
                conflict[j].add(i)
                
    # Helper function to check if a set of indices is valid (no internal conflicts)
    def is_valid(indices):
        idx_list = list(indices)
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                if idx_list[j] in conflict[idx_list[i]]:
                    return False
        return True
        
    # Helper to calculate score
    def score(indices):
        return sum(1.0 / nodes[i] for i in indices)
        
    # 1. Greedy Initialization
    # Sort by weight descending (1/x descending => x ascending)
    # Smaller numbers have higher weight.
    sorted_indices = sorted(range(n_vars), key=lambda i: nodes[i])
    
    selected = set()
    for i in sorted_indices:
        # Check if i conflicts with any in selected
        compatible = True
        for s in selected:
            if s in conflict[i]:
                compatible = False
                break
        if compatible:
            selected.add(i)
            
    best_subset = set(selected)
    best_value = score(best_subset)
    
    # 2. Local Search
    # We will perform 1-swap and 2-swap moves.
    # 1-swap: Remove s, add n. Valid if n is compatible with selected \ {s}.
    #          Gain if 1/n > 1/s (i.e., n < s).
    # 2-swap: Remove s1, s2; add n1, n2. Valid if {n1, n2} compatible with selected \ {s1, s2}
    #          and n1, n2 compatible with each other.
    #          Gain if (1/n1 + 1/n2) > (1/s1 + 1/s2).
            
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # --- 1-Swap Phase ---
        # Try to replace any selected node with a non-selected node of higher weight
        # To be efficient, iterate non-selected nodes in order of increasing value (highest weight first)
        non_selected = [i for i in range(n_vars) if i not in selected]
        non_selected.sort(key=lambda i: nodes[i]) # Smaller value = higher weight
        
        for n in non_selected:
            if time.time() - start_time > deadline:
                break
            # Try to remove any s that conflicts with n or has lower weight
            # Actually, we just need to find an s such that replacing s with n is valid and improves score.
            # Improvement: 1/n > 1/s => nodes[n] < nodes[s]
            # Validity: n is not in conflict with any s' in selected \ {s}
            
            for s in list(selected):
                if nodes[n] >= nodes[s]:
                    # No improvement possible (or equal), skip
                    # Since we sorted n by value ascending, if nodes[n] >= nodes[s], 
                    # and we are checking s which might be large, this condition might be true.
                    # Wait, if nodes[n] < nodes[s], we have improvement.
                    # So we only care if nodes[n] < nodes[s].
                    continue
                
                # Check if n is compatible with all other selected nodes
                compatible = True
                for s_other in selected:
                    if s_other == s:
                        continue
                    if s_other in conflict[n]:
                        compatible = False
                        break
                
                if compatible:
                    # Perform swap
                    selected.remove(s)
                    selected.add(n)
                    improved = True
                    best_value = score(selected)
                    if best_value > best_value: # Redundant but safe
                        pass
                    # Update best if better
                    if score(selected) > best_value:
                        best_value = score(selected)
                        best_subset = set(selected)
                    
                    # Restart 1-swap loop for this n? Or break to next n?
                    # Breaking to next n is safer as the set changed.
                    # However, we might be able to do another swap with the same n? No, n is now in selected.
                    # So we break out of the 's' loop.
                    break
                    
        if not improved:
            # --- 2-Swap Phase ---
            # This is more expensive. O(N^4) naive.
            # We can optimize by precomputing potential pairs.
            # Let's try a limited 2-swap: 
            # Pick two non-selected nodes n1, n2 with high weight.
            # Pick two selected nodes s1, s2 with low weight.
            # Check if swap is valid and improves.
            
            # To keep it fast, we'll limit the search to a subset of candidates.
            # Top 20 non-selected by weight
            # Bottom 20 selected by weight
            non_selected_top = [i for i in range(n_vars) if i not in selected]
            non_selected_top.sort(key=lambda i: nodes[i])
            non_selected_top = non_selected_top[:20]
            
            selected_sorted = sorted(list(selected), key=lambda i: nodes[i], reverse=True) # Largest values first (lowest weight)
            selected_bottom = selected_sorted[:20]
            
            found_improvement = False
            for n1 in non_selected_top:
                if time.time() - start_time > deadline:
                    break
                for n2 in non_selected_top:
                    if n1 == n2:
                        continue
                    if n1 > n2:
                        continue # Avoid duplicates
                    
                    # Check if n1 and n2 conflict
                    if n2 in conflict[n1]:
                        continue
                    
                    for s1 in selected_bottom:
                        if time.time() - start_time > deadline:
                            break
                        for s2 in selected_bottom:
                            if s1 == s2:
                                continue
                            if s1 > s2:
                                continue
                            
                            # Check weight improvement
                            if (1.0/nodes[n1] + 1.0/nodes[n2]) <= (1.0/nodes[s1] + 1.0/nodes[s2]):
                                continue
                                
                            # Check validity
                            # n1, n2 must be compatible with selected \ {s1, s2}
                            compatible = True
                            for s in selected:
                                if s == s1 or s == s2:
                                    continue
                                if s in conflict[n1] or s in conflict[n2]:
                                    compatible = False
                                    break
                            
                            if compatible:
                                # Perform swap
                                selected.remove(s1)
                                selected.remove(s2)
                                selected.add(n1)
                                selected.add(n2)
                                
                                val = score(selected)
                                if val > best_value:
                                    best_value = val
                                    best_subset = set(selected)
                                
                                improved = True
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break

            if not improved:
                break

    # Final check
    if not best_subset:
        # Fallback: single largest number is always valid
        return [256]
        
    return list(best_subset)

if __name__ == "__main__":
    res = construct()
    print(res)
