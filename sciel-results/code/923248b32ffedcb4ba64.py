import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    def time_left():
        return deadline - (time.time() - start_time)

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try smaller n with Greedy Repair
    # n must be at least 31 (Erdos bound). 
    # Random tournaments have ~ n^4/16 violations. 
    # n=50: 50^4/16 = 1.95M. 
    # n=40: 40^4/16 = 400k.
    # n=35: 35^4/16 = 150k.
    
    # We will try to find a solution for n < 83.
    # If we find one, we return it.
    # If not, we return Paley(83).
    
    # Greedy Repair Algorithm:
    # 1. Initialize random tournament.
    # 2. Compute out_masks (bitset of out-neighbors).
    # 3. Find all violating 4-sets.
    # 4. If none, done.
    # 5. For each violating 4-set, find the edge (u,v) whose flip maximizes the reduction in violations.
    #    Actually, flipping an edge (u,v) changes the out-neighborhood of u and v.
    #    This affects the status of any 4-set containing u or v.
    #    To be efficient, we can't just check all 4-sets.
    #    
    #    Alternative: Local Search with "steepest descent" on a subset of violations.
    #    Pick a random violating 4-set Q.
    #    For each vertex w in V, check if flipping an edge incident to w helps.
    #    Actually, simpler:
    #    Pick a random violating 4-set Q = {a,b,c,d}.
    #    Try flipping each edge (u,v) where u,v are in Q? No, the dominator might be outside Q.
    #    Wait, if Q is not dominated, no vertex dominates all of Q.
    #    To fix Q, we need some vertex x to dominate all of Q.
    #    This requires changing edges from x to Q.
    #    If x is in Q, say x=a, then a already dominates a. We need a->b, a->c, a->d.
    #    If some are missing, we can flip them.
    #    If x is not in Q, we need x->a, x->b, x->c, x->d.
    #    
    #    Strategy:
    #    1. Find a violating 4-set Q.
    #    2. For each vertex v in V:
    #       Let missing = Q \ out_neighbors(v).
    #       If |missing| is small (e.g., 1 or 2), we can fix it by flipping edges from v to missing vertices.
    #       But flipping an edge might break other 4-sets.
    #       
    #    Let's use a cost function: Cost = number of violating 4-sets.
    #    We will perform local search:
    #    - Pick a random violating 4-set Q.
    #    - For each vertex v in V:
    #      - Calculate the number of edges from v to Q that are "wrong" (i.e., v does not dominate Q).
    #      - If we flip all wrong edges, v would dominate Q.
    #      - But we can only flip one edge at a time.
    #      
    #    Let's try a simpler heuristic:
    #    For a violating 4-set Q, pick a random vertex v in V.
    #    If v is not in Q, and v dominates 3 of the 4 vertices in Q, flip the edge to the 4th.
    #    This might fix Q but break others.
    #    
    #    Let's implement a "Greedy Edge Flip" based on immediate benefit.
    #    Benefit of flipping (u,v):
    #    - u gains v as out-neighbor, loses u's previous in-neighbor status for v? No, u->v becomes v->u.
    #    - So u's out_mask loses v, v's out_mask loses u?
    #    - Wait, if we flip u->v to v->u:
    #      - u no longer dominates v.
    #      - v now dominates u.
    #    - This affects all 4-sets containing u or v.
    
    def count_violations_fast(n, out_mask):
        # out_mask[v] is bitset of out-neighbors of v.
        # A 4-set {a,b,c,d} is violated if no v dominates all of them.
        # i.e., for all v, (out_mask[v] & (1<<a | 1<<b | 1<<c | 1<<d)) != (1<<a | 1<<b | 1<<c | 1<<d)
        
        # We can precompute for each 4-set? Too many.
        # We will just count.
        count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            dominated = False
            for v in range(n):
                if (out_mask[v] & mask) == mask:
                    dominated = True
                    break
            if not dominated:
                count += 1
        return count

    def greedy_repair(n, adj, max_iters=2000):
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        # Precompute list of 4-sets for faster iteration?
        # For n=50, 230k sets. 
        # Checking each set takes n operations. 230k * 50 = 11.5M ops.
        # In Python, this is slow. ~1s per check.
        # 2000 iters * 1s = 2000s. Too slow.
        
        # We need a faster violation check or repair.
        
        # Optimization:
        # Instead of checking all 4-sets, we can maintain a count of "coverage" for each 4-set?
        # No, memory is 230k ints, fine. But updating is hard.
        
        # Let's try a different approach:
        # Random Restart Local Search with a good move selection.
        # Move: Flip an edge (u,v).
        # Delta Cost:
        #   Let Q be a 4-set.
        #   Q is covered if exists v such that v->Q.
        #   Flipping (u,v) changes out_mask of u and v.
        #   Only 4-sets containing u or v are affected.
        #   Number of such 4-sets:
        #   - Containing u: C(n-1, 3)
        #   - Containing v: C(n-1, 3)
        #   - Containing both: C(n-2, 2)
        #   Total ~ 2 * n^3 / 6 = n^3 / 3.
        #   For n=50, 50^3/3 = 41k sets.
        #   Checking each set is O(n). 41k * 50 = 2M ops.
        #   Still slow in Python.
        
        # We must use NumPy or bit operations efficiently.
        
        # Let's try n=40.
        # C(40,4) = 91k.
        # n^3/3 = 21k.
        
        # Let's implement a C-like loop in Python but with bit masks.
        
        # For a given flip (u,v):
        # u loses v, v loses u? No.
        # If u->v becomes v->u:
        # out_mask[u] &= ~(1<<v)
        # out_mask[v] |= (1<<u)
        # 
        # A 4-set Q is covered if (OR over all x of (out_mask[x] & Q_mask) == Q_mask) is True.
        # Let S_Q = {x | out_mask[x] covers Q}.
        # Q is covered if S_Q is non-empty.
        # Flipping (u,v) changes S_Q only if u or v is in Q? No, u or v is in S_Q.
        # If u in S_Q, u might leave S_Q.
        # If v in S_Q, v might enter S_Q.
        # Also, if u or v is in Q, the definition of "covering Q" for other vertices doesn't change,
        # but u and v's ability to cover Q changes?
        # If u in Q, u covers u. If u->v changes, u's coverage of v changes.
        
        # This is getting complex. Let's stick to the simple "Try to fix a random violation" approach
        # but with a very fast check.
        
        # Fast Check for a specific 4-set Q:
        # Precompute for each vertex x, the set of 4-sets it covers? No.
        
        # Let's try:
        # 1. Find one violating 4-set Q.
        # 2. For each vertex x, check if x is "almost" dominating Q (misses 1 edge).
        # 3. If found, flip that edge.
        # 4. Repeat.
        
        # Finding one violating 4-set:
        # Randomly sample 4-sets until one is found.
        # If density of violations is high, this is fast.
        
        def find_one_violation(n, out_mask, samples=1000):
            for _ in range(samples):
                quad = random.sample(range(n), 4)
                mask = (1 << quad[0]) | (1 << quad[1]) | (1 << quad[2]) | (1 << quad[3])
                dominated = False
                for v in range(n):
                    if (out_mask[v] & mask) == mask:
                        dominated = True
                        break
                if not dominated:
                    return quad
            return None

        def try_fix(n, adj, out_mask, quad):
            mask = (1 << quad[0]) | (1 << quad[1]) | (1 << quad[2]) | (1 << quad[3])
            # Find vertices that miss exactly 1 edge to quad
            candidates = []
            for v in range(n):
                # Count edges from v to quad
                # out_mask[v] & mask
                # If v is in quad, v dominates itself.
                # We need v to dominate all 4.
                # Let missing = mask & ~out_mask[v]
                # But if v is in quad, v is in out_mask[v]? No, no self-loops.
                # So if v in quad, v dominates 3 others at best?
                # Wait, a vertex v dominates a set Q if for all q in Q, v->q.
                # If v in Q, v->v is false. So v can never dominate a set containing itself.
                # Therefore, the dominator MUST be outside the 4-set.
                
                # Correction:
                # "Every set of 4 vertices must have a vertex with arcs to all four."
                # This vertex is NOT in the set of 4.
                
                # So for a 4-set Q, we look for v NOT in Q such that v->Q.
                
                if v in quad:
                    continue
                    
                # Check if v dominates Q
                # v must have edges to all 4 in Q.
                # out_mask[v] & mask == mask
                
                # We want to find v that is close.
                # Let missing = mask & ~out_mask[v]
                # If popcount(missing) == 1, we can fix it by flipping the edge.
                
                missing = mask & ~out_mask[v]
                # Count bits in missing
                # Since mask has 4 bits, missing has 0,1,2,3,4 bits.
                # We want 1.
                
                # Fast bit count for small numbers:
                # missing is a bitset.
                # We can just check if missing has exactly 1 bit set.
                # missing & (missing - 1) == 0 and missing != 0
                
                if missing != 0 and (missing & (missing - 1)) == 0:
                    # Find the bit
                    u = missing.bit_length() - 1
                    # Flip edge v->u to u->v?
                    # Currently v does NOT point to u (since u is in missing).
                    # So u->v is currently true.
                    # We want v->u to be true.
                    # So we flip u->v to v->u.
                    candidates.append((v, u))
            
            if not candidates:
                return False
            
            # Pick a random candidate
            v, u = random.choice(candidates)
            
            # Flip edge u->v to v->u
            # adj[u][v] should be 1 (u->v). We set it to 0.
            # adj[v][u] should be 0. We set it to 1.
            
            # Update out_masks
            out_mask[v] |= (1 << u)
            out_mask[u] &= ~(1 << v)
            
            # Update adj
            adj[u][v] = 0
            adj[v][u] = 1
            
            return True

        # Main Loop
        for i in range(max_iters):
            if time_left() < 5:
                break
                
            quad = find_one_violation(n, out_mask)
            if quad is None:
                return adj, 0
                
            if not try_fix(n, adj, out_mask, quad):
                # Try a random flip to escape local minima?
                # Or break and restart.
                break
                
        # Count violations to return cost
        # This is expensive, but we only do it at the end or periodically.
        # For now, we assume if we can't find a violation in 1000 samples, it's likely 0 or low.
        # But we need to be sure.
        # Let's do a full check if time permits.
        
        # Full check is O(n^5). Too slow.
        # We will return the adj and let the caller verify.
        # But we need to know if it's valid to stop.
        
        # Let's do a faster full check using bitsets.
        # For each 4-set, check if dominated.
        # We can vectorize this?
        
        # Let's just run the full check if n < 50.
        if n < 50:
            cost = count_violations_fast(n, out_mask)
            return adj, cost
        else:
            # Estimate cost
            # Sample 1000 quads
            violations = 0
            for _ in range(1000):
                quad = random.sample(range(n), 4)
                mask = (1 << quad[0]) | (1 << quad[1]) | (1 << quad[2]) | (1 << quad[3])
                dominated = False
                for v in range(n):
                    if (out_mask[v] & mask) == mask:
                        dominated = True
                        break
                if not dominated:
                    violations += 1
            # Extrapolate
            total_quads = len(list(combinations(range(n), 4)))
            estimated_cost = int(violations / 1000 * total_quads)
            return adj, estimated_cost

    # Try n from 40 to 60
    # If we find a valid tournament, we return it.
    # We prioritize smaller n.
    
    for n in range(40, 61):
        if time_left() < 20:
            break
            
        # Try a few random starts
        for seed in range(5):
            if time_left() < 10:
                break
                
            random.seed(seed)
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                    else:
                        adj[j][i] = 1
                        
            best_adj, cost = greedy_repair(n, adj, max_iters=1000)
            
            if cost == 0:
                # Verify
                # Rebuild out_mask for verification
                out_mask = [0] * n
                for v in range(n):
                    mask = 0
                    for w in range(n):
                        if best_adj[v][w]:
                            mask |= (1 << w)
                    out_mask[v] = mask
                if count_violations_fast(n, out_mask) == 0:
                    best_n = n
                    best_orient = adj_to_orient(n, best_adj)
                    break
                    
        if best_n < 83:
            break

    return {'n': best_n, 'orientation': best_orient}
