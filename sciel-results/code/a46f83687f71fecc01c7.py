import math
import random
import time

def construct():
    """
    Returns a nonempty subset of [1, 256] such that a+b is never a perfect square 
    for a<=b in the set, INCLUDING a=b. Maximizes cardinality.
    """
    N = 256
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Identify forbidden vertices where 2*u is a perfect square
    # 2*u = k^2 => u = k^2/2. k must be even, let k=2j, u=2j^2.
    # Max k such that k^2 <= 2*256 = 512 => k <= 22.
    forbidden = set()
    for k in range(1, int(math.isqrt(2 * N)) + 1):
        val = k * k
        if val % 2 == 0:
            u = val // 2
            if 1 <= u <= N:
                forbidden.add(u)
                
    # Candidates are all numbers 1..256 not in forbidden
    candidates = [u for u in range(1, N + 1) if u not in forbidden]
    
    # 2. Build conflict graph among candidates
    # u and v conflict if u+v is a perfect square.
    # Precompute squares
    squares = set()
    max_sum = 2 * N
    for k in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(k * k)
        
    # Adjacency list
    # To speed up, we can map value to list of candidates
    # But N is small, so O(N^2) precomputation is fine.
    # Let's use a set of pairs for quick checking during search, or adjacency lists.
    # Adjacency lists are better for "who conflicts with me".
    
    # Map candidate value to index for faster lookup? Or just use values directly.
    # Using values directly is easier.
    adj = {u: set() for u in candidates}
    
    for i in range(len(candidates)):
        u = candidates[i]
        for j in range(i + 1, len(candidates)):
            v = candidates[j]
            if u + v in squares:
                adj[u].add(v)
                adj[v].add(u)
                
    # 3. Greedy + Local Search
    best_set = set()
    best_size = 0
    
    # Heuristic ordering: vertices with fewer conflicts are "safer" to pick?
    # Or vertices with more conflicts are "harder", so pick them first?
    # Usually, picking vertices with high degree first (greedy coloring logic) can be bad for MIS if not careful, 
    # but for independent set, picking a vertex removes its neighbors. 
    # A common heuristic for MIS: pick vertex with min degree, add to set, remove it and its neighbors.
    # Let's try a few randomizations.
    
    def greedy_mis(cand_list, adj_map, random_order=False):
        # Copy list
        remaining = list(cand_list)
        if random_order:
            random.shuffle(remaining)
        
        # Sort by degree (number of conflicts in current remaining set)
        # This is dynamic, so we might just pick random or min-degree.
        # Let's try min-degree heuristic first, then random.
        
        selected = set()
        # To make it efficient, we maintain a set of available vertices.
        available = set(remaining)
        
        while available:
            # Find vertex with minimum degree in the induced subgraph of 'available'
            # For small N, we can just iterate.
            min_deg = float('inf')
            chosen = None
            for v in available:
                deg = len(adj_map[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
                    if deg == 0:
                        break
            if chosen is None:
                break
                
            selected.add(chosen)
            # Remove chosen and its neighbors from available
            to_remove = {chosen} | (adj_map[chosen] & available)
            available -= to_remove
            
        return selected

    def local_search(initial_set, adj_map):
        current_set = set(initial_set)
        current_size = len(current_set)
        improved = True
        # Try to add any vertex not in set that doesn't conflict
        # Then try swaps: remove v, add u, if it allows adding more later?
        # Simple 1-swap: remove v, add u. Net change 0. 
        # But maybe removing v allows adding two others? That's 2-swap.
        
        # Let's try a simple hill climbing:
        # 1. Try to add any available vertex.
        # 2. If stuck, try to remove a vertex and see if we can add more.
        
        while improved:
            improved = False
            # Phase 1: Add
            added = False
            # Iterate over candidates not in set
            for u in candidates:
                if u in current_set:
                    continue
                # Check if u conflicts with any in current_set
                conflict = False
                for v in current_set:
                    if v in adj_map[u]:
                        conflict = True
                        break
                if not conflict:
                    current_set.add(u)
                    current_size += 1
                    improved = True
                    added = True
                    # Restart check from beginning to catch new additions
                    break 
            if added:
                continue
                
            # Phase 2: Swap / Remove to unlock
            # Try removing each vertex in current_set and see if we can add more
            # This is expensive O(N^2). N is 256, so 256^2 = 65k checks. Fine.
            # We look for a vertex v such that removing it allows adding k > 1 vertices.
            # Or simply, we try all pairs (remove v, add u) and then try to fill.
            
            # Simpler: Try to find a vertex v in current_set. Remove it. 
            # Then try to greedily add as many as possible.
            # If the new size is larger, keep it.
            
            # To speed up, only try a subset of removals if time is tight.
            # But 256 is small.
            
            best_new_set = None
            best_new_size = current_size
            
            for v in list(current_set):
                # Simulate removing v
                temp_set = current_set - {v}
                
                # Try to add vertices that were blocked by v
                # Or just run a quick greedy fill on the remaining space
                # Let's just try to add any vertex that is not in temp_set and has no conflict with temp_set
                # And specifically, check if we can add vertices that conflict with v.
                
                potential_adds = []
                for u in candidates:
                    if u in temp_set:
                        continue
                    # Check conflict with temp_set
                    ok = True
                    for w in temp_set:
                        if w in adj_map[u]:
                            ok = False
                            break
                    if ok:
                        potential_adds.append(u)
                        
                if not potential_adds:
                    continue
                    
                # Sort potential adds by degree (greedy)
                # Just pick the one that leads to max size? 
                # Let's just add all non-conflicting ones greedily.
                new_set = set(temp_set)
                # Add as many as possible from potential_adds
                # Order matters. Let's try adding the one with fewest conflicts first.
                potential_adds.sort(key=lambda x: len(adj_map[x] & new_set))
                
                for u in potential_adds:
                    if u in new_set:
                        continue
                    # Re-check conflict because new_set might have changed
                    conflict = False
                    for w in new_set:
                        if w in adj_map[u]:
                            conflict = True
                            break
                    if not conflict:
                        new_set.add(u)
                        
                if len(new_set) > best_new_size:
                    best_new_size = len(new_set)
                    best_new_set = new_set
                    
            if best_new_set is not None:
                current_set = best_new_set
                current_size = best_new_size
                improved = True
                
        return current_set

    # Run multiple restarts with different random seeds/orders
    # Since the graph is static, the "min-degree" greedy is deterministic.
    # We need randomization to explore different parts of the solution space.
    
    # Strategy: Randomized Greedy
    # 1. Shuffle candidates
    # 2. Pick first valid, remove neighbors. Repeat.
    
    def randomized_greedy(seed):
        random.seed(seed)
        available = set(candidates)
        selected = set()
        
        # Order candidates randomly
        order = list(candidates)
        random.shuffle(order)
        
        for u in order:
            if u not in available:
                continue
            # If u is still available, it means it hasn't been removed by a previous selection's neighbors
            # And it doesn't conflict with any previously selected?
            # Wait, if we remove neighbors upon selection, then any u remaining in 'available' 
            # does not conflict with any selected vertex.
            # So we can just add it.
            selected.add(u)
            # Remove u and its neighbors from available
            to_remove = {u} | (adj_map[u] & available)
            available -= to_remove
            
        return selected

    # We will combine randomized greedy with local search.
    
    for i in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        # Get a greedy solution
        initial = randomized_greedy(i)
        
        # Improve with local search
        improved = local_search(initial, adj_map)
        
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
            
    # Final check: verify best_set is valid
    # And ensure it's not empty
    if not best_set:
        # Fallback: just pick one valid number
        # 1 is valid if 2 is not square (2 is not). 
        # But 1+1=2 (not sq), 1+2=3 (not sq)... 
        # Let's just return a known small valid set if something went wrong
        # 1 is safe? 2*1=2 (no). 1+1=2 (no). 
        # Check 1 against all squares.
        # If 1 is in forbidden, it's not in candidates.
        # If candidates is empty, return [1] (but 1 might be forbidden? 2*1=2 not square. So 1 is never forbidden).
        # Actually, if 1 is forbidden, it means 2 is square, which is false.
        # So 1 is always a candidate.
        return [1]
        
    return list(best_set)

# EVOLVE-BLOCK-END
