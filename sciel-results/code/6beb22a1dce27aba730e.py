import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 20
    
    def is_k33_free(adj):
        """Full check for verification."""
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = adj[a] & adj[b]
                if len(common_ab) < 3:
                    continue
                for c in range(b + 1, n):
                    if len(common_ab & adj[c]) >= 3:
                        return False
        return True

    def try_add_edge(adj, u, v):
        """
        Attempt to add edge (u,v).
        Returns True if successful, False otherwise.
        Optimized check: only checks triples involving u or v.
        """
        if u == v:
            return False
        if v in adj[u]:
            return True # Already exists, technically "successful" but no change
        
        adj[u].add(v)
        adj[v].add(u)
        
        # Check triples involving u: {u, x, y}
        # We need |N(u) ∩ N(x) ∩ N(y)| <= 2
        for x in range(n):
            if x == u:
                continue
            for y in range(x + 1, n):
                if y == u:
                    continue
                # Intersection of neighbors of u, x, y
                # N(u) includes v now.
                # We can just compute the set intersection
                if len(adj[u] & adj[x] & adj[y]) > 2:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    return False
        
        # Check triples involving v: {v, x, y}
        for x in range(n):
            if x == v:
                continue
            for y in range(x + 1, n):
                if y == v:
                    continue
                if len(adj[v] & adj[x] & adj[y]) > 2:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    return False
        
        return True

    def remove_edge(adj, u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def get_edge_count(adj):
        return sum(len(s) for s in adj) // 2

    def greedy_build(rng):
        adj = [set() for _ in range(n)]
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            try_add_edge(adj, u, v)
        return adj

    def local_search_swap(adj, rng, time_limit):
        """
        Local search using 1-opt (add/remove) and 2-opt (swap).
        """
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            
            # 1-opt: Try to add missing edges
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
            rng.shuffle(missing)
            for u, v in missing:
                if try_add_edge(adj, u, v):
                    improved = True
            
            # 1-opt: Try to remove edges that allow more additions? 
            # Usually, just adding is enough for 1-opt if we want max edges.
            # But removing might open up space.
            # Let's rely on swaps for that.

            # 2-opt: Swap an existing edge for a missing one
            # To be efficient, we don't try ALL swaps every time, but a sample or all if fast.
            # n=20, m~80. Swaps ~ 80*110 = 8800.
            # Check cost: ~342 intersections.
            # 8800 * 342 ~ 3M ops. Might be slow in Python if done repeatedly.
            # Let's sample a subset of swaps.
            
            if time.time() - start > time_limit * 0.8:
                break
                
            existing = [(u, v) for u in range(n) for v in range(u + 1, n) if v in adj[u]]
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
            
            rng.shuffle(existing)
            rng.shuffle(missing)
            
            # Try a limited number of random swaps
            swaps_to_try = 100
            for _ in range(swaps_to_try):
                if not existing or not missing:
                    break
                a, b = rng.choice(existing)
                c, d = rng.choice(missing)
                
                # Remove (a,b)
                adj[a].discard(b)
                adj[b].discard(a)
                
                # Try add (c,d)
                if try_add_edge(adj, c, d):
                    improved = True
                    # Update lists? 
                    # It's easier to just recompute lists in next iteration or keep them somewhat fresh.
                    # For simplicity, we just mark improved and let the next loop recompute.
                    # But we modified adj.
                    # To avoid invalidating lists too much, we can just break and restart inner loop?
                    # Or just continue. If we add (c,d), the graph changed.
                    # The 'existing' and 'missing' lists are now stale.
                    # Given the small size, recomputing is cheap.
                    # Let's just break out of the swap loop and recompute lists in the next while iteration.
                    break
                else:
                    # Restore (a,b)
                    adj[a].add(b)
                    adj[b].add(a)

        return adj

    def simulated_annealing(adj, rng, time_limit):
        start = time.time()
        current = [set(s) for s in adj]
        best = [set(s) for s in current]
        best_count = get_edge_count(current)
        
        T = 10.0
        cooling = 0.9999
        
        while time.time() - start < time_limit:
            # Generate a neighbor
            r = rng.random()
            if r < 0.6:
                # Add a random missing edge
                missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in current[u]]
                if not missing:
                    continue
                u, v = rng.choice(missing)
                # Tentatively add
                current[u].add(v)
                current[v].add(u)
                
                # Check validity
                if not is_k33_free(current):
                    # Revert
                    current[u].discard(v)
                    current[v].discard(u)
                    # Reject
                else:
                    new_count = get_edge_count(current)
                    if new_count > best_count:
                        best = [set(s) for s in current]
                        best_count = new_count
            elif r < 0.8:
                # Remove a random existing edge
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if v in current[u]]
                if not existing:
                    continue
                u, v = rng.choice(existing)
                current[u].discard(v)
                current[v].discard(u)
            else:
                # Swap: Remove existing, Add missing
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if v in current[u]]
                missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in current[u]]
                if not existing or not missing:
                    continue
                u1, v1 = rng.choice(existing)
                u2, v2 = rng.choice(missing)
                
                # Remove (u1, v1)
                current[u1].discard(v1)
                current[v1].discard(u1)
                
                # Try add (u2, v2)
                current[u2].add(v2)
                current[v2].add(u2)
                
                if not is_k33_free(current):
                    # Revert both
                    current[u2].discard(v2)
                    current[v2].discard(u2)
                    current[u1].add(v1)
                    current[v1].add(u1)
                else:
                    new_count = get_edge_count(current)
                    if new_count > best_count:
                        best = [set(s) for s in current]
                        best_count = new_count
            
            T *= cooling
            
        return best, best_count

    # Main Loop
    rng = random.Random(42)
    
    # 1. Greedy Init
    best_adj = None
    best_count = 0
    
    # Try a few greedy inits
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        adj = greedy_build(rng)
        count = get_edge_count(adj)
        if count > best_count:
            best_adj = adj
            best_count = count

    # 2. Local Search
    if best_adj is not None:
        time_left = deadline - time.time() - 10
        if time_left > 1:
            best_adj = local_search_swap(best_adj, rng, min(5.0, time_left))
            best_count = get_edge_count(best_adj)

    # 3. Simulated Annealing
    if time.time() < deadline - 15:
        time_left = deadline - time.time() - 5
        if time_left > 1:
            sa_adj, sa_count = simulated_annealing(best_adj, rng, min(60.0, time_left))
            if sa_count > best_count:
                best_adj = sa_adj
                best_count = sa_count

    # 4. Final Local Search
    if time.time() < deadline - 2:
        time_left = deadline - time.time() - 1
        if time_left > 0.5:
            best_adj = local_search_swap(best_adj, rng, min(2.0, time_left))
            best_count = get_edge_count(best_adj)

    # Convert to edge list
    edges = []
    for u in range(n):
        for v in sorted(best_adj[u]):
            if u < v:
                edges.append([u, v])
                
    # Final Verification
    if not is_k33_free(best_adj):
        # Fallback to a known valid graph if something went wrong
        # K_{2,18} is valid
        edges = [[a, b] for a in range(2) for b in range(2, 20)]
        
    return edges
