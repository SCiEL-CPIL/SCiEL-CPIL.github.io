import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    # Leave margin for evaluator overhead (150s wall clock limit)
    deadline = start_time + 115.0

    def find_K4(n, adj):
        """
        Finds a K4 in the graph. Returns a list of 4 vertices or None.
        adj is a list of bitmasks (integers).
        """
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                
                # Common neighbors of i and j with index > j
                common_mask = adj[i] & adj[j]
                # Mask out bits <= j
                common_mask &= ~((1 << (j + 1)) - 1)
                
                if (common_mask & (common_mask - 1)) == 0:
                    continue
                
                common_verts = []
                m = common_mask
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    common_verts.append(v)
                    m ^= lsb
                
                for a_idx in range(len(common_verts)):
                    u = common_verts[a_idx]
                    for b_idx in range(a_idx + 1, len(common_verts)):
                        v = common_verts[b_idx]
                        if (adj[u] >> v) & 1:
                            return [i, j, u, v]
        return None

    def find_independent_set(n, adj, target_size=6):
        """
        Finds an independent set of size target_size.
        Returns the list of vertices or None.
        Uses backtracking with ordering (vertices must be increasing).
        """
        def backtrack(candidates_mask, current_set):
            if len(current_set) == target_size:
                return current_set[:]
            
            if candidates_mask == 0:
                return None
            
            # Pruning: if not enough candidates left
            if len(current_set) + candidates_mask.bit_count() < target_size:
                return None
            
            verts = []
            m = candidates_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Sort by degree in the induced subgraph (ascending) to find independent sets faster
            # Vertices with fewer neighbors in candidates are better candidates
            verts.sort(key=lambda v: (adj[v] & candidates_mask).bit_count())
            
            for v in verts:
                # To avoid duplicates, we only pick vertices > v for the next step?
                # No, standard backtracking for independent set with ordering:
                # If we pick v, the next vertices must be > v to maintain unique sets?
                # Actually, if we just remove v and its neighbors, we might pick u < v later?
                # No, because we iterate in a specific order. 
                # Let's stick to the "increasing index" constraint for efficiency and uniqueness.
                # If we pick v, subsequent picks must be > v.
                
                # Mask for vertices > v
                mask_gt_v = candidates_mask & ~((1 << (v + 1)) - 1)
                # Remove neighbors of v
                new_candidates = mask_gt_v & ~adj[v]
                
                # Pruning
                if len(current_set) + 1 + new_candidates.bit_count() < target_size:
                    continue
                
                res = backtrack(new_candidates, current_set + [v])
                if res:
                    return res
            return None
        
        return backtrack((1 << n) - 1, [])

    def is_valid(n, adj):
        """
        Checks if graph is K4-free and has no independent set of size 6.
        """
        if find_K4(n, adj):
            return False
        if find_independent_set(n, adj, 6):
            return False
        return True

    def count_violations(n, adj):
        """
        Returns (num_k4, num_indep6) - approximations or exact counts for local search guidance.
        For speed, we just check existence for the main loop, but for scoring swaps,
        we might want a faster metric.
        However, exact counting is expensive.
        Let's use a heuristic score: 
        Score = -10 * (1 if K4 exists else 0) - 10 * (1 if Indep6 exists else 0)
        This is binary. To make local search work, we need a gradient.
        
        Alternative: 
        Count the number of K4s? Too slow for n=35 in Python if done naively.
        
        Let's try a different approach for local search:
        We maintain a "conflict list".
        If K4 is found, remove an edge from it.
        If Indep6 is found, add an edge to it.
        This is the "repair" strategy which worked in the seed.
        
        To improve, we can use a "simulated annealing" style or just random restarts with better initialization.
        
        Let's stick to the repair strategy but make it smarter.
        1. Start with a random graph or a structured graph (e.g., Paley-like or random regular).
        2. Loop:
           a. Find K4. If found, remove a random edge from it.
           b. Else find Indep6. If found, add a random edge to it.
           c. Else return valid.
        
        To avoid getting stuck, we can add random perturbations (flipping random edges) occasionally.
        """
        pass

    def local_search(n, max_time):
        """
        Attempts to construct a valid graph on n vertices.
        """
        # Initialize random graph
        # For n=35, p ~ 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_time = time.time()
        steps = 0
        max_steps = 50000 # Safety cap
        
        # We will alternate between fixing K4s and fixing Independent Sets.
        
        while current_time < max_time and steps < max_steps:
            # Check K4
            k4 = find_K4(n, adj)
            if k4:
                # Remove a random edge from K4
                pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                u, v = pairs[random.randrange(6)]
                u, v = k4[u], k4[v]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                steps += 1
                current_time = time.time()
                continue
            
            # Check Independent Set 6
            indep6 = find_independent_set(n, adj, 6)
            if indep6:
                # Add a random edge within the independent set
                pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
                u_idx, v_idx = pairs[random.randrange(15)]
                u, v = indep6[u_idx], indep6[v_idx]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                steps += 1
                current_time = time.time()
                continue
            
            # If neither K4 nor Indep 6 found, valid
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        return None

    def try_construct(n, max_time):
        """
        Tries multiple restarts of local search to find a valid graph on n vertices.
        """
        best_result = None
        while time.time() < max_time:
            result = local_search(n, max_time)
            if result:
                return result
            # If local_search timed out, we have used up max_time?
            # No, local_search checks time internally.
            # If it returns None, it means it hit the time limit or step limit.
            # If step limit, we can try again?
            # If time limit, we should stop.
            # Let's check if we are close to deadline.
            if time.time() > max_time - 5:
                break
        return best_result

    # Strategy:
    # 1. Try n=36 (new record?) with a short time budget.
    # 2. Try n=35 (known record) with the remaining time.
    # 3. Fallback to n=34, etc.
    
    # Actually, n=36 is very unlikely to be found by simple local search in 115s.
    # The known bound is 35.
    # Let's focus on reliably finding 35, and use any spare time for 36.
    
    # Time allocation:
    # 10s for n=36 (long shot)
    # 90s for n=35
    # 10s for n=34 (fallback)
    
    # But if we find 35 quickly, we should use the rest for 36.
    
    # Let's do this:
    # 1. Try n=35 for 100s.
    # 2. If found, try n=36 for remaining time (if > 5s left).
    # 3. If not found 35, try n=34 for remaining time.
    
    # Actually, let's just iterate downwards, but give priority to higher n.
    
    best_result = None
    
    # Try 36 first with a small budget? No, if it's impossible, we waste time.
    # Let's try 35 first.
    
    # Time for 35: 100s
    t1 = time.time()
    result_35 = try_construct(35, t1 + 100)
    if result_35:
        best_result = result_35
        # If we found 35, do we have time for 36?
        if time.time() < deadline - 5:
            result_36 = try_construct(36, deadline)
            if result_36:
                best_result = result_36
    
    if not best_result:
        # Try 34
        if time.time() < deadline - 5:
            result_34 = try_construct(34, deadline)
            if result_34:
                best_result = result_34
    
    if not best_result:
        # Try 33
        if time.time() < deadline - 5:
            result_33 = try_construct(33, deadline)
            if result_33:
                best_result = result_33

    if not best_result:
        # Fallback: Paley 17
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
