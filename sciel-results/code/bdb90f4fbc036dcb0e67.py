import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    # Use a tight deadline to ensure we return well before the 90s kill.
    # Precomputation is fast (~0.01s). Search can use the rest.
    # Let's aim for 80s of work, but check every second.
    deadline = start_time + 75
    
    n = limit
    # 0-based index i represents number i+1
    # adj[i][j] is True if lcm(i+1, j+1) <= 512
    # Since we only care about i != j for distinct pairs in the set definition "lcm(a,b) for every pair",
    # but usually "pair" in such extremal problems implies distinct. 
    # The problem says "lcm(a,b)<=512 for every pair". If a=b, lcm(a,a)=a<=512 which is always true.
    # So we just need distinct pairs to be compatible.
    
    # Precompute adjacency matrix
    # We can use a list of sets for neighbors for faster "is neighbor" checks and iteration
    neighbors = [set() for _ in range(n)]
    
    for i in range(n):
        num_i = i + 1
        for j in range(i + 1, n):
            num_j = j + 1
            g = gcd(num_i, num_j)
            lcm_val = (num_i * num_j) // g
            if lcm_val <= limit:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    # Helper: check if node v is compatible with all nodes in a set S
    def is_compatible_with_all(v, S):
        # S is a set of indices
        for u in S:
            if u not in neighbors[v]:
                return False
        return True
    
    # Helper: check if a set S is a valid clique (all pairs compatible)
    # Since we build incrementally, we don't strictly need this, but good for verification if needed.
    # For speed, we rely on the incremental check.
    
    best_set = set()
    best_size = 0
    
    def update_best(new_set):
        nonlocal best_set, best_size
        if len(new_set) > best_size:
            best_size = len(new_set)
            best_set = new_set
    
    # --- Phase 1: Greedy Constructions ---
    # Try multiple different orders to seed the local search.
    # Good orders:
    # 1. Ascending number
    # 2. Descending number
    # 3. Sorted by degree (high degree first - "popular" nodes)
    # 4. Sorted by degree (low degree first)
    # 5. Random permutations
    
    degrees = [len(neighbors[i]) for i in range(n)]
    
    orders = []
    orders.append(list(range(n)))  # Ascending
    orders.append(list(reversed(range(n))))  # Descending
    orders.append(sorted(range(n), key=lambda x: -degrees[x]))  # High degree first
    orders.append(sorted(range(n), key=lambda x: degrees[x]))   # Low degree first
    
    random.seed(42)
    for _ in range(50):
        shuffled = list(range(n))
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = set()
        for v in order:
            if is_compatible_with_all(v, current_set):
                current_set.add(v)
                
        update_best(current_set)
        
        # Local Search: Hill Climbing with 1-for-k moves
        # We try to improve the current set by removing some nodes and adding others.
        # Common effective moves for max clique:
        # - 1-for-2: Remove 1, Add 2 (net +1)
        # - 2-for-3: Remove 2, Add 3 (net +1)
        # - 1-for-1: Swap (net 0, but might enable future moves) - often less useful for max size directly unless combined.
        
        local_set = set(current_set)
        improved = True
        local_best_size = len(local_set)
        
        while improved and time.time() < deadline:
            improved = False
            # Try 1-for-2 moves
            # Iterate over all nodes u in local_set
            # For each u, consider local_set - {u}
            # Find all candidates v not in local_set that are compatible with local_set - {u}
            # Among those candidates, can we pick 2 that are mutually compatible?
            
            nodes_in_set = list(local_set)
            
            for i in range(len(nodes_in_set)):
                if time.time() > deadline:
                    break
                u = nodes_in_set[i]
                temp_set = local_set - {u}
                
                # Find candidates compatible with temp_set
                candidates = []
                for v in range(n):
                    if v in local_set:
                        continue
                    if is_compatible_with_all(v, temp_set):
                        candidates.append(v)
                        
                if len(candidates) < 2:
                    continue
                
                # Check if any pair in candidates is an edge
                # If we find such a pair (v1, v2), we can replace u with v1, v2.
                found_pair = False
                # Optimization: Sort candidates by degree in the candidate graph? 
                # Or just check pairs. Number of candidates is usually small.
                for k in range(len(candidates)):
                    v1 = candidates[k]
                    for l in range(k + 1, len(candidates)):
                        v2 = candidates[l]
                        if v2 in neighbors[v1]:
                            # Valid move: remove u, add v1, v2
                            new_set = temp_set | {v1, v2}
                            local_set = new_set
                            local_best_size = len(local_set)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    # Restart the loop over nodes_in_set with the new set
                    # This is O(N^3) worst case per improvement, but sets are small.
                    # We break the inner loop and restart the while loop logic.
                    # To keep it simple, we just update local_set and break out of the for-loop to restart the while.
                    # Note: nodes_in_set is stale, but we re-list it in the next while iteration?
                    # No, the for loop is inside the while. We should break the for loop and let the while loop restart.
                    # The `improved` flag will be True, so the while continues.
                    # We need to break the for loop.
                    pass # The break statements above handle the inner loops. 
                    # We need to break the outer for loop too if we improved, to restart with the new set.
                    # Actually, standard hill climbing: apply best move, restart.
                    # Let's just break the for loop if improved.
                    if improved:
                        break
            
            # If we did an improvement, the while loop will continue.
            # If no 1-for-2 found, try 2-for-3?
            # 2-for-3 is significantly more expensive. 
            # Given the time budget, let's see if 1-for-2 is enough.
            # If stuck, we might need to do a random restart of the local search phase or try 2-for-3.
            
            # If no improvement was found in this pass of 1-for-2 moves,
            # try 2-for-3 moves.
            if not improved:
                # 2-for-3 move: Remove 2 nodes, Add 3 nodes.
                # Iterate over pairs of nodes to remove.
                # This is O(|S|^2 * N^2). Might be slow.
                # Let's limit the number of pairs we try or just do a few.
                
                found_triple = False
                nodes_in_set = list(local_set)
                
                # To save time, only try 2-for-3 if the set is large enough
                # and we have time.
                if len(local_set) > 10 and time.time() < deadline - 5:
                    for i in range(len(nodes_in_set)):
                        if time.time() > deadline:
                            break
                        u1 = nodes_in_set[i]
                        for j in range(i + 1, len(nodes_in_set)):
                            if time.time() > deadline:
                                break
                            u2 = nodes_in_set[j]
                            temp_set = local_set - {u1, u2}
                            
                            # Find candidates compatible with temp_set
                            candidates = []
                            for v in range(n):
                                if v in local_set:
                                    continue
                                if is_compatible_with_all(v, temp_set):
                                    candidates.append(v)
                                
                                if len(candidates) >= 3:
                                    break
                                    
                            if len(candidates) < 3:
                                continue
                            
                            # Check if any triplet in candidates forms a clique
                            # C(3,3) = 1 check if exactly 3, C(k,3) if more.
                            # Just check all triples.
                            for k in range(len(candidates)):
                                v1 = candidates[k]
                                for l in range(k + 1, len(candidates)):
                                    v2 = candidates[l]
                                    if v2 not in neighbors[v1]:
                                        continue
                                    for m in range(l + 1, len(candidates)):
                                        v3 = candidates[m]
                                        if v3 in neighbors[v1] and v3 in neighbors[v2]:
                                            # Valid move
                                            new_set = temp_set | {v1, v2, v3}
                                            local_set = new_set
                                            local_best_size = len(local_set)
                                            improved = True
                                            found_triple = True
                                            break
                                    if found_triple:
                                        break
                                if found_triple:
                                    break
                            if found_triple:
                                break
                        if found_triple:
                            break
                            
                # If we improved with 2-for-3, the while loop continues.
                # If not, we are stuck in a local optimum.
                # Update global best if local is better (it shouldn't be, as we only improved).
                update_best(local_set)
                break # Break out of while if no improvement

    # --- Phase 2: Randomized Local Search from Best Global Set ---
    # Sometimes 1-for-2/2-for-3 gets stuck. 
    # We can perturb the best set and re-local-search.
    
    if best_size > 0 and time.time() < deadline:
        # Copy best set
        current_best = set(best_set)
        
        for restart in range(20): # Limit restarts
            if time.time() > deadline:
                break
                
            # Perturb: Remove 1-2 random elements
            if len(current_best) < 3:
                break
                
            perturb_set = set(current_best)
            num_remove = 1 if random.random() < 0.5 else 2
            to_remove = random.sample(list(perturb_set), num_remove)
            for r in to_remove:
                perturb_set.remove(r)
                
            # Re-add greedily? Or just start local search from here?
            # Starting local search from a non-maximal set might not increase size if we don't fill it first.
            # Let's try to fill it greedily first, then local search.
            # Actually, local search 1-for-2 assumes we are at a local optimum of 1-for-2.
            # If we remove nodes, we are NOT at a local optimum, so we might just re-add the same nodes.
            # Better: Do a "fill" step to get back to a maximal set, then optimize.
            
            # Fill step: add any compatible node
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for v in range(n):
                    if v not in perturb_set and is_compatible_with_all(v, perturb_set):
                        perturb_set.add(v)
                        changed = True
                        break # Restart loop to check others
            
            # Now do local search on perturb_set
            local_set = set(perturb_set)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                nodes_in_set = list(local_set)
                
                # 1-for-2
                for i in range(len(nodes_in_set)):
                    if time.time() > deadline:
                        break
                    u = nodes_in_set[i]
                    temp_set = local_set - {u}
                    candidates = [v for v in range(n) if v not in local_set and is_compatible_with_all(v, temp_set)]
                    
                    if len(candidates) >= 2:
                        for k in range(len(candidates)):
                            v1 = candidates[k]
                            for l in range(k + 1, len(candidates)):
                                v2 = candidates[l]
                                if v2 in neighbors[v1]:
                                    local_set = temp_set | {v1, v2}
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                
                if not improved:
                    # Try 2-for-3 briefly
                    if len(local_set) > 10 and time.time() < deadline - 5:
                        nodes_in_set = list(local_set)
                        found = False
                        for i in range(len(nodes_in_set)):
                            if time.time() > deadline:
                                break
                            u1 = nodes_in_set[i]
                            for j in range(i + 1, len(nodes_in_set)):
                                if time.time() > deadline:
                                    break
                                u2 = nodes_in_set[j]
                                temp_set = local_set - {u1, u2}
                                candidates = [v for v in range(n) if v not in local_set and is_compatible_with_all(v, temp_set)]
                                
                                if len(candidates) >= 3:
                                    for k in range(len(candidates)):
                                        v1 = candidates[k]
                                        for l in range(k + 1, len(candidates)):
                                            v2 = candidates[l]
                                            if v2 not in neighbors[v1]:
                                                continue
                                            for m in range(l + 1, len(candidates)):
                                                v3 = candidates[m]
                                                if v3 in neighbors[v1] and v3 in neighbors[v2]:
                                                    local_set = temp_set | {v1, v2, v3}
                                                    improved = True
                                                    found = True
                                                    break
                                            if found:
                                                break
                                        if found:
                                            break
                                    if found:
                                        break
                            if found:
                                break
                    if not improved:
                        break # Stuck

            if len(local_set) > best_size:
                best_size = len(local_set)
                best_set = local_set
                current_best = set(local_set)

    # Final Verification (Optional but safe)
    # Verify that best_set is a valid clique
    # This can be expensive, but with N=512 and size ~30-40, it's fine.
    is_valid = True
    best_list = list(best_set)
    for i in range(len(best_list)):
        for j in range(i + 1, len(best_list)):
            if best_list[j] not in neighbors[best_list[i]]:
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback to a simple valid set if verification fails (should not happen)
        # Use the largest power of 2 up to 512? No, just [1] is valid.
        # Or re-run a simple greedy.
        # Let's just return [1] to be safe? No, that's bad.
        # Better: return the result of the first greedy (Ascending) which is verified by construction.
        # We didn't store it separately, but we can reconstruct.
        # Actually, the local search only adds/removes based on compatibility checks, so it should be valid.
        # If is_valid is False, there's a bug.
        # For now, return the set.
        pass
        
    return sorted([idx + 1 for idx in best_set])

construct()
