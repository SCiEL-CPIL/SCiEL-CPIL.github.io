import time
import sys

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    
    # 1. All odds
    odds = list(range(1, 257, 2))
    
    # 2. Find safe evens
    safe_evens = []
    for e in range(2, 257, 2):
        # Calculate odd part m and power of 2 part 2^k
        m = e
        k = 0
        while m % 2 == 0:
            m //= 2
            k += 1
        # Safe if m <= 2^k
        if m <= (1 << k):
            safe_evens.append(e)
            
    # 3. Build conflict graph among safe_evens
    # We want to find a large independent set in this small graph.
    # Since N is small (~23), we can use a simple greedy approach or backtracking.
    # Greedy: Sort by degree (number of conflicts) ascending, add if no conflict with already added.
    
    n_se = len(safe_evens)
    # Adjacency list or just check pairs
    # For simplicity and speed, we can just iterate and check against current set.
    # To maximize, we should prioritize nodes with fewer conflicts?
    # Or just take all and remove conflicts?
    # Let's use a simple recursive backtracking for MIS on this small graph, 
    # or a greedy heuristic if time is tight.
    # Given n~23, greedy is likely sufficient and fast.
    
    # Let's try a greedy independent set selection.
    # Sort safe_evens.
    # A better heuristic: Tomita's coloring bound is overkill for 23 nodes.
    # Simple branch and bound or just greedy.
    
    # Let's implement a simple recursive MIS solver for the safe_evens subgraph.
    # It's small enough to be exact and fast.
    
    def get_mis(vertices, adj):
        # vertices: list of indices
        # adj: dict of dict or similar
        # Returns max independent set size and the set itself
        if not vertices:
            return 0, set()
        
        # Choose a vertex with max degree in the induced subgraph?
        # Or just the first one.
        # To keep it simple and fast, we can just use a greedy approach first, 
        # and if we have time, try to improve.
        
        # Actually, for 23 nodes, we can just check all subsets? No, 2^23 is 8 million.
        # 8 million checks is fine in Python if optimized? Maybe.
        # But we can do better.
        
        # Let's use a simple greedy:
        # 1. Calculate degrees in the subgraph.
        # 2. Pick vertex with min degree.
        # 3. Add to independent set.
        # 4. Remove it and its neighbors.
        # 5. Repeat.
        
        # This greedy might not be optimal, but it's fast.
        # Let's try to be slightly smarter:
        # Try all vertices as the first choice? No.
        
        # Given the small size, let's just do a backtracking with pruning.
        # But to ensure speed, let's stick to a good greedy.
        
        pass # Placeholder

    # Let's just use a greedy independent set on the safe_evens.
    # We want to maximize the number of safe evens we keep.
    # Conflicts are rare.
    
    selected_safe = []
    # Sort by value? Or by number of conflicts?
    # Let's count conflicts for each safe even.
    conflict_count = {e: 0 for e in safe_evens}
    for i in range(len(safe_evens)):
        for j in range(i+1, len(safe_evens)):
            a = safe_evens[i]
            b = safe_evens[j]
            if (a * b) % (a + b) == 0:
                conflict_count[a] += 1
                conflict_count[b] += 1
                
    # Sort by conflict count ascending (least connected first)
    sorted_safe = sorted(safe_evens, key=lambda x: conflict_count[x])
    
    for e in sorted_safe:
        # Check if e conflicts with any already selected safe even
        can_add = True
        for s in selected_safe:
            if (e * s) % (e + s) == 0:
                can_add = False
                break
        if can_add:
            selected_safe.append(e)
            
    # Final set
    result = odds + selected_safe
    
    # Verify validity (just in case)
    # For every distinct a, b in result, (a*b) % (a+b) != 0
    # This is O(N^2) which is 256^2 ~ 65k ops. Very fast.
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a = result[i]
            b = result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to just odds if something went wrong
        return odds
        
    return result
