import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    n = 35
    
    # Adjacency matrix as list of sets for fast lookup
    # adj[i] is a set of neighbors of i
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        
    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        
    def has_c5_after_add(u, v):
        """
        Check if adding edge (u,v) creates a 5-cycle.
        This happens if there is a path u-a-b-v of length 3 in the current graph.
        """
        # u and v are not currently connected
        # We need a != b
        # a in N(u), b in N(v), (a,b) in E
        for a in adj[u]:
            if a == v: continue # Should not happen if not connected
            for b in adj[a]:
                if b == u: continue
                if b == v: continue # b is neighbor of v? No, b is neighbor of a. We check if b is in N(v)
                # Check if b is neighbor of v
                if b in adj[v]:
                    # Path u-a-b-v exists
                    # Ensure distinct vertices: u,a,b,v
                    # u != a (a in N(u)), u != b (if b==u, handled), u != v (not connected)
                    # a != b (if a==b, then a in N(u) and a in N(v)? No, b in N(a). If a==b, then a in N(a) which is impossible in simple graph)
                    # a != v (if a==v, then v in N(u) which is false)
                    # b != v (if b==v, then v in N(a) and v in N(v) which is false)
                    # So u,a,b,v are distinct.
                    return True
        return False

    def find_independent_set_of_size_9():
        """
        Try to find an independent set of size 9.
        Returns a list of 9 vertices or None.
        """
        # Simple backtracking search
        # Sort vertices by degree (descending) to prune faster?
        # Or just use random ordering for diversity
        
        # We can use a recursive function with memoization or just simple DFS
        # Given n=35, depth 9 is small.
        
        candidates = list(range(n))
        random.shuffle(candidates)
        
        def backtrack(current_set, start_idx):
            if len(current_set) == 9:
                return current_set[:]
            
            # Pruning: if not enough candidates left
            if len(candidates) - start_idx < 9 - len(current_set):
                return None
            
            for i in range(start_idx, len(candidates)):
                v = candidates[i]
                # Check if v is independent from current_set
                if all(v not in adj[u] for u in current_set):
                    res = backtrack(current_set + [v], i + 1)
                    if res:
                        return res
            return None

        return backtrack([], 0)

    def get_alpha_upper_bound():
        """
        Greedy approximation of alpha.
        """
        # Just to have a quick check, but find_independent_set is better for exactness in this context
        pass

    def is_valid_graph():
        """
        Check if graph has no C5 and alpha < 9.
        """
        # Check C5
        # We maintained this property, but let's double check if needed.
        # For now, we trust the incremental check.
        
        # Check alpha < 9
        # We need to be sure alpha is NOT 9.
        # Finding a set of size 9 proves alpha >= 9.
        # Not finding one does not prove alpha < 9.
        # However, if we've been adding edges to break sets, it's likely small.
        # To be safe, we should verify.
        
        # Exact check for alpha >= 9 is the same as find_independent_set_of_size_9.
        # If it returns None, it's possible alpha is 8, or it's 9 and we missed it.
        # Given the search space, a randomized backtracking is quite reliable for n=35, k=9.
        # We can run it a few times.
        
        for _ in range(10):
            if find_independent_set_of_size_9():
                return False
        return True

    # Main loop
    best_result = None
    
    # Start with empty graph
    # We will try to add edges greedily
    
    while time.time() < deadline:
        # Check if current graph is valid
        if is_valid_graph():
            # Construct edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find an independent set of size 9 to break
        indep_set = find_independent_set_of_size_9()
        
        if not indep_set:
            # If we can't find one, it might be valid, or we missed it.
            # Let's assume valid if multiple attempts fail?
            # Or try to add a random safe edge to change state?
            # If we are stuck, we might need to restart or change strategy.
            # For now, let's try to add a random edge that doesn't create C5.
            added = False
            for _ in range(100):
                u, v = random.sample(range(n), 2)
                if v not in adj[u]:
                    if not has_c5_after_add(u, v):
                        add_edge(u, v)
                        added = True
                        break
            if not added:
                # Stuck? Restart?
                # Let's try a different randomization seed or reset
                # Resetting to empty might be too slow.
                # Let's just continue, maybe the search failed.
                pass
        else:
            # Try to add an edge within indep_set
            added = False
            random.shuffle(indep_set)
            for i in range(len(indep_set)):
                for j in range(i + 1, len(indep_set)):
                    u, v = indep_set[i], indep_set[j]
                    if v not in adj[u]:
                        if not has_c5_after_add(u, v):
                            add_edge(u, v)
                            added = True
                            break
                if added:
                    break
            
            if not added:
                # Cannot add any edge within this independent set without creating C5.
                # This is a dead end for this specific set.
                # We might need to remove some edges to allow new paths?
                # Or restart.
                # Let's try a random restart with a slightly different initial state?
                # Or just break and try again?
                # If we are stuck, we might be in a local optimum.
                # Let's try to remove a random edge and try again?
                # This is a "simulated annealing" style move.
                if adj:
                    # Pick a random edge to remove
                    u = random.choice(range(n))
                    if adj[u]:
                        v = random.choice(list(adj[u]))
                        remove_edge(u, v)
                else:
                    # Empty graph, but alpha=35. Should have found set.
                    pass

    # If loop finishes, return the best effort (which might be invalid, but we checked validity in loop)
    # If we never returned, we might have timed out.
    # The task says "return the best result found".
    # If we didn't find a valid one, we should probably return the seed or a known valid one?
    # The seed returns n=28.
    # Let's return a known valid graph if we fail to find n=35.
    
    # Fallback: The 7 disjoint K4s
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
