import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 40  # Safety margin
    
    # 1. Generate 7-smooth numbers up to 20000
    smooth = set()
    for p1 in range(0, 15):
        for p2 in range(0, 15):
            for p3 in range(0, 15):
                for p4 in range(0, 15):
                    val = (2**p1) * (3**p2) * (5**p3) * (7**p4)
                    if val <= 20000:
                        smooth.add(val)
    
    # 2. Search for clique of size 12
    # Try increasing pool sizes
    for pool_size in [50, 100, 200, 500, 1000, 10000]:
        if time.time() > deadline - 5:
            break
        candidates = list(range(1, pool_size + 1))
        # Sort by degree? Maybe not necessary for small pools.
        
        found = find_clique(candidates, smooth, 12, deadline)
        if found:
            return found
            
    # Fallback: Return best effort or a valid set (even if score is worse)
    # A valid set is required. 1..12 is valid.
    return list(range(1, 13))

def find_clique(candidates, smooth, target_size, deadline):
    # Backtracking
    def dfs(clique, avail):
        if time.time() > deadline:
            return None
        if len(clique) == target_size:
            return clique[:]
        
        # Pruning: if not enough candidates left
        if len(avail) < target_size - len(clique):
            return None
            
        for i in range(len(avail)):
            v = avail[i]
            new_clique = clique + [v]
            # Filter avail: keep only those compatible with v
            # Note: we must also ensure they are compatible with all previous clique members?
            # The 'avail' list passed in should already contain only nodes compatible with ALL nodes in 'clique'.
            # So we just need to filter 'avail' (excluding v) for compatibility with v.
            
            new_avail = []
            for u in avail[i+1:]: # Use i+1 to avoid duplicates and order issues
                if (v + u) in smooth:
                    new_avail.append(u)
                    
            res = dfs(new_clique, new_avail)
            if res:
                return res
        return None

    # Initial call: clique is empty, avail is all candidates
    # But we need to pick the first element.
    # To reduce symmetry, we can just iterate.
    # Optimization: Pick the first element from candidates, then recurse.
    # Actually, standard Bron-Kerbosch or simple DFS on ordered list works.
    
    # Let's optimize the first step.
    # We can try each candidate as the root.
    # But to save time, we can just pass the whole list and let the loop handle it.
    # However, passing the whole list and filtering inside is O(N^2) per level.
    
    # Better approach:
    # Start with empty clique.
    # avail = candidates
    # The loop in dfs picks v from avail.
    # This is correct.
    
    return dfs([], candidates)
