import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115
    
    def get_paley_83():
        p = 83
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    def has_dominator(a, b, c, d, adj):
        # adj[i] is bitmask of outgoing edges from i
        # Check if a dominates b,c,d
        if (adj[a] >> b) & 1 and (adj[a] >> c) & 1 and (adj[a] >> d) & 1:
            return True
        if (adj[b] >> a) & 1 and (adj[b] >> c) & 1 and (adj[b] >> d) & 1:
            return True
        if (adj[c] >> a) & 1 and (adj[c] >> b) & 1 and (adj[c] >> d) & 1:
            return True
        if (adj[d] >> a) & 1 and (adj[d] >> b) & 1 and (adj[d] >> c) & 1:
            return True
        return False

    def count_bad_sets_involving_v(v, adj, n):
        # Count 4-sets {v, a, b, c} that are bad
        count = 0
        # Iterate pairs a, b from 0..v-1
        # This is O(v^3) which is too slow for large v if done naively for every check.
        # We need a faster way or accept slower for small n.
        # For n=31..60, v is up to 60. v^3 is ~216,000. 
        # If we do this for each edge flip (v edges), it's v^4 ~ 12 million ops.
        # In Python, 12 million ops might take ~1-2 seconds.
        # We have ~110 seconds total.
        # If we try n=31, v goes from 5 to 30.
        # Sum of v^3 for v=5..30 is roughly integral of x^3 dx = (30^4 - 5^4)/4 ~ 200,000.
        # Wait, for each v, we do local search.
        # Local search: check all v edges. For each edge, calc delta.
        # Delta calc requires checking all 4-sets involving that edge.
        # An edge (v, u) is in (v-1 choose 2) 4-sets? No.
        # The set is {v, u, w, x}. w, x are from 0..v-1, w != u.
        # So (v-1) * (v-2) / 2 sets.
        # For v=30, ~400 sets.
        # Checking each set is O(1) bit ops.
        # So delta calc is ~400 ops.
        # We have v edges. So one full pass is v * v^2 = v^3.
        # For v=30, 27,000 ops.
        # We might need many passes. Say 100 passes. 2.7 million ops.
        # This is fast in Python if optimized (bit ops).
        
        # Let's implement a faster check.
        # Precompute for each pair (u, w) whether u->w? No, adj is bitmasks.
        
        # Actually, let's just write the check.
        for a in range(v):
            for b in range(a + 1, v):
                for c in range(b + 1, v):
                    if not has_dominator(v, a, b, c, adj):
                        count += 1
        return count

    def greedy_add_vertex(adj, v, n, time_budget):
        # adj is list of bitmasks for 0..v-1
        # We add vertex v.
        # Initialize adj[v] = 0
        adj.append(0)
        
        # Randomly initialize edges from v to 0..v-1
        for u in range(v):
            if random.random() < 0.5:
                adj[v] |= (1 << u)
            else:
                adj[u] |= (1 << v)
                
        current_cost = count_bad_sets_involving_v(v, adj, n)
        
        end_time = time.time() + time_budget
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < end_time and current_cost > 0:
            # Find best edge to flip
            best_delta = 0
            best_u = -1
            best_flip = False # True if we flip v->u to u->v
            
            for u in range(v):
                # Check current state of edge (v, u)
                # v -> u iff (adj[v] >> u) & 1
                v_to_u = (adj[v] >> u) & 1
                
                # Calculate delta
                # We need to check all 4-sets {v, u, w, x}
                delta = 0
                # Optimize: 
                # For each w, x in 0..v-1, w != u, x != u, w < x
                # We can iterate w, x
                
                # To speed up, we can precompute something?
                # Let's just iterate.
                for w in range(v):
                    if w == u: continue
                    for x in range(w + 1, v):
                        if x == u: continue
                        
                        # Check old state
                        old_good = has_dominator(v, u, w, x, adj)
                        
                        # Flip
                        if v_to_u:
                            adj[v] &= ~(1 << u)
                            adj[u] |= (1 << v)
                        else:
                            adj[v] |= (1 << u)
                            adj[u] &= ~(1 << v)
                            
                        # Check new state
                        new_good = has_dominator(v, u, w, x, adj)
                        
                        # Undo
                        if v_to_u:
                            adj[v] |= (1 << u)
                            adj[u] &= ~(1 << v)
                        else:
                            adj[v] &= ~(1 << u)
                            adj[u] |= (1 << v)
                            
                        if old_good and not new_good:
                            delta += 1
                        elif not old_good and new_good:
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_u = u
                    best_flip = v_to_u # If v_to_u was True, we flip to False
            
            if best_u != -1 and best_delta < 0:
                # Apply flip
                if best_flip:
                    adj[v] &= ~(1 << best_u)
                    adj[best_u] |= (1 << v)
                else:
                    adj[v] |= (1 << best_u)
                    adj[best_u] &= ~(1 << v)
                
                current_cost += best_delta
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        return current_cost == 0

    def try_construct(n, time_budget):
        # Start with n=5
        # Find a valid tournament for n=5
        # There are 1024 tournaments.
        # We can just randomize and check.
        base_adj = None
        for _ in range(100):
            adj = [0] * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                    else:
                        adj[j] |= (1 << i)
            # Check if valid for n=5
            # 4-sets: {0,1,2,3}, {0,1,2,4}, {0,1,3,4}, {0,2,3,4}, {1,2,3,4}
            valid = True
            for a in range(5):
                for b in range(a+1, 5):
                    for c in range(b+1, 5):
                        for d in range(c+1, 5):
                            if not has_dominator(a, b, c, d, adj):
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if valid:
                base_adj = adj[:]
                break
        
        if base_adj is None:
            return None
            
        adj = base_adj[:]
        current_n = 5
        
        while current_n < n:
            v = current_n
            # Allocate time slice for this vertex
            # Total time budget for this n is time_budget
            # Remaining vertices: n - current_n
            # Time per vertex: time_budget / (n - current_n)
            slice_time = time_budget / (n - current_n)
            # Cap slice time to avoid long waits if stuck
            slice_time = min(slice_time, 2.0)
            
            if not greedy_add_vertex(adj, v, n, slice_time):
                # Failed to add vertex v
                return None
                
            current_n += 1
            
        return adj

    # Try n from 31 to 82
    for n in range(31, 83):
        if time.time() > start_time + time_limit - 5:
            break
            
        # Allocate time for this n
        # Give more time to smaller n? Or equal?
        # Let's give 1.0s to each n initially.
        # If we find a solution, return.
        
        # We can try multiple restarts for each n
        # Let's say 5 restarts of 0.2s each? Or 1 restart of 1.0s?
        # Greedy is deterministic given random seed.
        # Let's do 10 restarts of 0.1s each.
        
        found = False
        for restart in range(10):
            if time.time() > start_time + time_limit - 2:
                break
            res = try_construct(n, 0.1)
            if res is not None:
                orientation = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (res[i] >> j) & 1:
                            orientation.append(0)
                        else:
                            orientation.append(1)
                return {'n': n, 'orientation': orientation}
        
    return get_paley_83()
