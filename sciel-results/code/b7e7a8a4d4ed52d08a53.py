import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def try_find_graph(n, target_deg, time_limit):
        """
        Attempts to find a C4-free graph on n vertices with min degree >= target_deg.
        Returns list of adj bitsets or None.
        """
        start = time.time()
        max_attempts = 100000 # High number, will be limited by time
        
        for attempt in range(max_attempts):
            if time.time() > start + time_limit:
                break
            
            adj = [0] * n
            # Randomize the order of vertices to break symmetry
            order = list(range(n))
            random.shuffle(order)
            
            # We will iterate through vertices and try to satisfy their degree
            # We might need multiple passes
            success = False
            stuck_count = 0
            
            for _ in range(n * 10): # Max internal iterations
                if time.time() > start + time_limit:
                    return None
                
                # Find vertex with minimum degree
                min_deg = n
                min_v = -1
                for i in range(n):
                    deg = adj[i].bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        min_v = i
                
                if min_deg >= target_deg:
                    # Check if C4-free (should be, but verify just in case)
                    # This check is O(n^2) but we only do it once at the end
                    is_c4_free = True
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj[i] & adj[j]).bit_count() >= 2:
                                is_c4_free = False
                                break
                        if not is_c4_free:
                            break
                    if is_c4_free:
                        return adj
                    else:
                        # Should not happen if we maintain invariant, but if it does, restart
                        break
                
                # Try to add an edge to min_v
                candidates = [j for j in range(n) if j != min_v and not (adj[min_v] >> j) & 1]
                random.shuffle(candidates)
                
                added = False
                for j in candidates:
                    # Check if adding (min_v, j) creates a C4
                    # Condition: |N(min_v) & N(j)| < 2
                    if (adj[min_v] & adj[j]).bit_count() < 2:
                        # Safe to add
                        adj[min_v] |= (1 << j)
                        adj[j] |= (1 << min_v)
                        added = True
                        break
                
                if not added:
                    # Stuck: cannot add any edge to min_v without creating C4
                    # Strategy: Remove a random edge incident to min_v to free up space
                    # Or remove a random edge in the graph
                    if min_deg > 0:
                        # Remove a random neighbor
                        neighbors = [i for i in range(n) if (adj[min_v] >> i) & 1]
                        if neighbors:
                            rem_v = random.choice(neighbors)
                            adj[min_v] &= ~(1 << rem_v)
                            adj[rem_v] &= ~(1 << min_v)
                    else:
                        # min_v has degree 0, but no candidates? 
                        # This means all other vertices have N(v) such that |N(min_v) & N(v)| >= 2.
                        # But N(min_v) is empty, so intersection is 0.
                        # This case should not happen unless n=1.
                        # If it happens, restart.
                        break

            # If loop finishes without success, this attempt failed.
            # The outer loop will try a new random seed/order.
            
        return None

    # Try n=47
    # Time budget: 60s
    adj_47 = try_find_graph(47, 8, 60.0)
    if adj_47:
        edges = []
        for i in range(47):
            for j in range(i+1, 47):
                if (adj_47[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': 47, 'edges': edges}

    # Try n=46
    # Time budget: 40s
    adj_46 = try_find_graph(46, 7, 40.0)
    if adj_46:
        edges = []
        for i in range(46):
            for j in range(i+1, 46):
                if (adj_46[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': 46, 'edges': edges}

    # Try n=45
    # Time budget: 20s
    adj_45 = try_find_graph(45, 6, 20.0)
    if adj_45:
        edges = []
        for i in range(45):
            for j in range(i+1, 45):
                if (adj_45[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': 45, 'edges': edges}

    # Fallback: Return a trivial valid graph (e.g., empty graph or small one)
    # n=39, min degree 0. Empty graph is C4-free.
    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
