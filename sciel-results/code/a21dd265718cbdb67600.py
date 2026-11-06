import time
import random
import itertools

def has_c4(adj, n):
    """
    Check if graph with adjacency lists is C4-free.
    Condition: For any pair u, v, |N(u) intersection N(v)| <= 1.
    """
    # Using sets for fast intersection
    # Precompute sets
    adj_sets = [set(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj_sets[u]:
            if v > u:
                common = adj_sets[u] & adj_sets[v]
                if len(common) > 1:
                    return True
    return False

def exists_independent_set_11(adj, n):
    """
    Check if there is an independent set of size 11.
    Returns True if exists, False otherwise.
    """
    # Backtracking search
    # adj[i] is a set of neighbors. non-neighbors are easier?
    # Let's use complement graph? No, just check if all pairs in set are non-adjacent.
    
    # Simple backtracking with bitmasks or sets.
    # For n=43, sets are fine.
    
    def backtrack(current_set, candidates):
        if len(current_set) >= 11:
            return True
        if len(candidates) + len(current_set) < 11:
            return False
            
        # Pick a vertex from candidates
        # To improve, pick one with fewest non-neighbors in candidates?
        # Or just first one.
        
        # Heuristic: order candidates
        # For simplicity, just iterate
        # We can remove candidate v and all its neighbors from next candidates
        
        # Optimization: If no candidates, fail
        if not candidates:
            return False
            
        # Try each candidate
        # It's often faster to process in some order
        # Let's just take the first element, then recurse on two branches?
        # No, standard independent set branching:
        # Pick v.
        # Branch 1: Include v. Next candidates = candidates \ {v} \ N(v).
        # Branch 2: Exclude v. Next candidates = candidates \ {v}.
        # We want to find ANY set of size 11.
        
        # Let's use the property that we just need size 11.
        v = next(iter(candidates))
        neighbors_in_candidates = candidates & adj[v]
        non_neighbors_in_candidates = candidates - neighbors_in_candidates - {v}
        
        # Branch 1: Include v
        if len(current_set) + len(non_neighbors_in_candidates) + 1 >= 11:
            if backtrack(current_set | {v}, non_neighbors_in_candidates):
                return True
                
        # Branch 2: Exclude v
        if len(current_set) + len(candidates - {v}) >= 11:
            if backtrack(current_set, candidates - {v}):
                return True
                
        return False

    candidates = set(range(n))
    return backtrack(set(), candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 100 # 100s budget, 20s safety
    
    # Try to find a graph with n as large as possible
    # The problem states best known is 39.
    # Let's try from 43 down to 35.
    
    for n in range(43, 34, -1):
        if time.time() > deadline:
            break
            
        # Determine probability p
        # For C4-free, p should be low.
        # p ~ n^-0.5
        base_p = 1.0 / (n ** 0.5)
        
        # Try a few p values
        ps = [base_p * 0.6, base_p * 0.8, base_p * 1.0, base_p * 1.2]
        
        # Shuffle to try different densities
        random.shuffle(ps)
        
        for p in ps:
            for _ in range(50): # Try 50 graphs per (n, p)
                if time.time() > deadline:
                    break
                
                # Generate random graph
                adj = [set() for _ in range(n)]
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < p:
                            adj[u].add(v)
                            adj[v].add(u)
                            edges.append([u, v])
                
                # Check C4
                if has_c4(adj, n):
                    continue
                
                # Check Independent Set
                if exists_independent_set_11(adj, n):
                    continue
                
                # Valid graph found
                return {'n': n, 'edges': edges}

    # Fallback to seed solution (n=30) if no larger found within time
    # But we should try to return at least something valid
    # The seed is 30.
    # We can also generate a valid graph for n=30 quickly if needed, but seed is guaranteed.
    
    # Actually, we should probably try n=35 with higher effort if we run out of time for larger.
    # But let's just return the best we found. If we found nothing > 30, we return seed.
    
    # Wait, if we loop down, and find n=35, we return it.
    # If we don't find anything, we should return the seed.
    
    # Construct seed just in case
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# Let's refine the MIS check to be faster.
# The recursive call `backtrack` creates new sets every time. This might be slow in Python.
# Bitmasks are faster for $n \le 64$.
