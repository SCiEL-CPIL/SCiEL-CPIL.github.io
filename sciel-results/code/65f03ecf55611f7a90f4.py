import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead
    
    def get_paley_83():
        p = 83
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                # 0 means i -> j, 1 means j -> i
                # i -> j iff (j - i) % p is a square
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    def has_dominator(a, b, c, d, adj):
        # Check a
        if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)):
            return True
        # Check b
        if (adj[b] & (1 << a)) and (adj[b] & (1 << c)) and (adj[b] & (1 << d)):
            return True
        # Check c
        if (adj[c] & (1 << a)) and (adj[c] & (1 << b)) and (adj[c] & (1 << d)):
            return True
        # Check d
        if (adj[d] & (1 << a)) and (adj[d] & (1 << b)) and (adj[d] & (1 << c)):
            return True
        return False

    def count_bad_sets(adj, n):
        count = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if not has_dominator(a, b, c, d, adj):
                            count += 1
        return count

    def find_bad_set(adj, n):
        # Returns a tuple (a,b,c,d) of a bad set, or None if none exist
        # To speed this up, we can randomize the order or just scan
        # Scanning is deterministic but might be slow if bad sets are rare.
        # Randomizing helps escape local minima in the search for a bad set? 
        # Actually, we just need ANY bad set to fix.
        
        # Optimization: If we know the count is 0, we don't call this.
        # We'll just scan.
        
        # To make it faster, we can shuffle the vertices? No, that's expensive.
        # Just scan.
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if not has_dominator(a, b, c, d, adj):
                            return (a, b, c, d)
        return None

    def local_search(n, time_budget):
        # Initialize random tournament
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                else:
                    adj[j] |= (1 << i)
        
        current_cost = count_bad_sets(adj, n)
        if current_cost == 0:
            return adj
        
        best_adj = adj[:]
        best_cost = current_cost
        
        end_time = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 5000 # Restart if stuck
        
        while time.time() < end_time:
            if current_cost == 0:
                return adj
                
            # Find a bad set to target
            bad_set = find_bad_set(adj, n)
            if bad_set is None:
                return adj
            
            a, b, c, d = bad_set
            
            # Try all 6 possible edge flips in this set
            # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            random.shuffle(edges) # Randomize order to avoid bias
            
            best_delta = 0
            best_edge = None
            
            for u, v in edges:
                # Calculate delta for flipping (u,v)
                # Only 4-sets containing u and v are affected.
                # But we are specifically looking at the set {a,b,c,d} and others.
                # To be safe, we calculate the global delta as before, but restricted to sets containing u,v.
                
                u_to_v = (adj[u] & (1 << v)) != 0
                
                delta = 0
                others = [k for k in range(n) if k != u and k != v]
                
                for i in range(len(others)):
                    w = others[i]
                    for j in range(i + 1, len(others)):
                        x = others[j]
                        
                        old_good = has_dominator(u, v, w, x, adj)
                        
                        # Flip edge in adj
                        if u_to_v:
                            adj[u] &= ~(1 << v)
                            adj[v] |= (1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] &= ~(1 << u)
                        
                        new_good = has_dominator(u, v, w, x, adj)
                        
                        # Undo flip
                        if u_to_v:
                            adj[u] |= (1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] |= (1 << u)
                        
                        if old_good and not new_good:
                            delta += 1
                        elif not old_good and new_good:
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_edge = (u, v, u_to_v)
            
            if best_edge is not None and best_delta < 0:
                u, v, u_to_v = best_edge
                # Apply flip
                if u_to_v:
                    adj[u] &= ~(1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] &= ~(1 << u)
                
                current_cost += best_delta
                no_improve_count = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
            else:
                # No improving move found in this bad set.
                # Try a random move to escape?
                # Or just increment no_improve
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Restart
                    adj = [0] * n
                    for i in range(n):
                        for j in range(i + 1, n):
                            if random.random() < 0.5:
                                adj[i] |= (1 << j)
                            else:
                                adj[j] |= (1 << i)
                    current_cost = count_bad_sets(adj, n)
                    if current_cost == 0:
                        return adj
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = adj[:]
                    no_improve_count = 0

        return best_adj if best_cost == 0 else None

    # Try n from 31 to 82
    for n in range(31, 83):
        if time.time() > start_time + time_limit - 10:
            break
        
        time_slice = 2.0 if n < 40 else 1.0
        
        if time.time() + time_slice > start_time + time_limit - 5:
            break
            
        res = local_search(n, time_slice)
        if res is not None:
            orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (res[i] & (1 << j)):
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': n, 'orientation': orientation}

    return get_paley_83()
