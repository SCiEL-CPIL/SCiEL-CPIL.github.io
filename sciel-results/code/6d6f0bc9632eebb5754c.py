import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    N = 256
    # 1. Precompute conflicts
    # conflicts[i] is a set of j such that i, j conflict
    # Conflict condition: let g = gcd(i, j). (i//g + j//g) divides g.
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            # Check if x+y divides g
            if g % (x + y) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    def is_valid_set(S):
        S_list = list(S)
        for k in range(len(S_list)):
            a = S_list[k]
            for l in range(k + 1, len(S_list)):
                b = S_list[l]
                if b in conflicts[a]:
                    return False
        return True

    def greedy_build():
        S = set()
        remaining = set(range(1, N + 1))
        while remaining:
            # Pick node with minimum conflicts with current S
            # Tie-breaker: min degree in remaining graph
            best_node = None
            min_conf_S = float('inf')
            min_deg_rem = float('inf')
            
            # Optimization: If any node has 0 conflicts with S, pick one with min degree
            candidates = []
            for v in remaining:
                conf_s = len(conflicts[v] & S)
                if conf_s == 0:
                    deg_rem = len(conflicts[v] & remaining)
                    candidates.append((deg_rem, v))
            
            if candidates:
                # Pick min degree
                candidates.sort()
                best_node = candidates[0][1]
            else:
                # No node is conflict-free with S? 
                # Wait, greedy MIS usually removes neighbors.
                # If we can't add any, we are stuck? 
                # No, the standard algorithm removes the node and its neighbors.
                # But here we want to build a set.
                # If no node in `remaining` is compatible with `S`, we must have made a bad choice earlier?
                # No, `remaining` contains nodes not yet considered.
                # If `conflicts[v] & S` is not empty for all v in remaining, we can't add any.
                # This implies S is a maximal independent set? Not necessarily.
                # Actually, if we just pick min `len(conflicts[v] & S)`, and it's > 0, we can't add it.
                # So we should only add if 0.
                # If no node has 0 conflicts, we stop.
                break
                
            S.add(best_node)
            # Remove best_node and its neighbors from remaining?
            # No, we just remove best_node. Other nodes might still be valid.
            # But if we don't remove neighbors, we might check them again.
            # It's fine, `conflicts[v] & S` will grow.
            remaining.remove(best_node)
            
        return S

    def local_search(S):
        S = set(S)
        best_S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try add moves first (increase size)
            added = False
            for v in range(1, N + 1):
                if v in S:
                    continue
                if not (conflicts[v] & S):
                    S.add(v)
                    improved = True
                    added = True
                    break # Restart to find more adds
            
            if added:
                continue # Restart loop to check for more adds
            
            # Try swap moves (maintain size, might enable future adds)
            S_list = list(S)
            for u in S_list:
                # Find candidates v not in S
                # To speed up, only consider v that don't conflict with S \ {u}
                # i.e., conflicts[v] & (S - {u}) is empty
                # And v != u
                for v in range(1, N + 1):
                    if v in S or v == u:
                        continue
                    # Check if v conflicts with S without u
                    if not (conflicts[v] & (S - {u})):
                        # Try the swap
                        new_S = S - {u}
                        new_S.add(v)
                        # It is valid by construction (v doesn't conflict with rest of S)
                        # But does it help? We assume yes if it allows future growth.
                        # To be safe, accept swap if it leads to a valid set.
                        # Since we can't guarantee growth, we might cycle.
                        # Use a simple acceptance: accept if valid.
                        # To avoid cycling, we can track visited states or just rely on time limit.
                        # Better: Only accept swap if it reduces "potential conflicts" or similar.
                        # For now, accept any valid swap.
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
            
            if S:
                if len(S) > len(best_S):
                    best_S = set(S)
        
        return best_S

    best_set = set()
    
    # Multi-start
    random.seed(42)
    num_starts = 0
    while time.time() < deadline - 5:
        # Perturb greedy by shuffling tie-breakers or starting with a random node
        # Simple perturbation: Remove a random element from greedy result and repair?
        # Or just run greedy again with randomized tie-breaking.
        
        # Modify greedy to use random tie-breaking
        S_greedy = greedy_build()
        
        # Improve with local search
        S_improved = local_search(S_greedy)
        
        if len(S_improved) > len(best_set):
            best_set = S_improved
        
        num_starts += 1
        if num_starts > 500: # Cap iterations to avoid infinite loop if local search is fast
             break

    # Final check
    if not best_set:
        best_set = greedy_build()
    
    return list(best_set)
