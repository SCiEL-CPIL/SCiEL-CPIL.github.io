import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0

    def time_left():
        return deadline - (time.time() - start_time)

    def orient_to_adj(n, orient):
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        return adj

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    def get_out_masks(n, adj):
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
        return out_mask

    def verify_tournament(n, adj, out_mask):
        """
        Fast verification using bitmasks.
        """
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            dominated = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    dominated = True
                    break
            if not dominated:
                return False
        return True

    def count_violations_and_get_list(n, adj, out_mask):
        """
        Returns the number of violations and a list of violating 4-sets.
        """
        violations = 0
        violating_quads = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            dominated = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    dominated = True
                    break
            if not dominated:
                violations += 1
                violating_quads.append(quad)
        return violations, violating_quads

    def paley_tournament(p):
        """
        Generate a Paley tournament on p vertices.
        p must be a prime congruent to 3 mod 4.
        """
        squares = set(x * x % p for x in range(1, p))
        adj = [[0]*p for _ in range(p)]
        for i in range(p):
            for j in range(i+1, p):
                if (j - i) % p in squares:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
        return adj

    def targeted_local_search(n, adj, out_mask, max_iters=5000):
        """
        Structural Change: Instead of random flips, we pick a violating 4-set
        and try to fix it by flipping an edge incident to one of its vertices.
        Specifically, we look for a vertex v that dominates 3 of the 4 vertices
        in the set, and flip the edge to the 4th.
        """
        current_violations, violating_quads = count_violations_and_get_list(n, adj, out_mask)
        if current_violations == 0:
            return adj, out_mask, True

        # If violations are too high, a random restart or different init might be needed,
        # but for now, we try to fix them.
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            if not violating_quads:
                return adj, out_mask, True

            # Pick a random violating quad
            quad = random.choice(violating_quads)
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            # Find candidates for repair
            # A vertex v can dominate the quad if it already dominates 3 of them
            # and we flip the edge to the 4th.
            candidates = []
            for v in range(n):
                if v in quad:
                    # If v is in the quad, it can't dominate itself.
                    # But it can dominate the other 3? No, the dominator must be in V.
                    # The definition says "a vertex with arcs to all four".
                    # So the dominator v must be distinct from a,b,c,d?
                    # "Every set of 4 vertices must have a vertex with arcs to all four."
                    # Usually, this implies the dominator is in the tournament, but does it have to be IN the set of 4?
                    # "Common dominator" in Schütte's problem usually means a vertex v (which can be in the set or not? 
                    # Actually, if v is in the set, it cannot have an arc to itself. So v must be OUTSIDE the set of 4?
                    # Let's check the problem statement: "Every set of 4 vertices must have a vertex with arcs to all four."
                    # If the set is {a,b,c,d}, and v=a, a cannot have an arc to a. So v must be distinct from a,b,c,d.
                    # Wait, is it possible that the "vertex" is one of the 4?
                    # If v=a, it needs arcs to a, b, c, d. Impossible for a->a.
                    # So the dominator must be a vertex OUTSIDE the set of 4.
                    # This implies n must be at least 5. The problem says 5 <= n.
                    # So yes, v not in {a,b,c,d}.
                    continue
                
                # Check how many of a,b,c,d are already dominated by v
                # We want v to dominate all 4.
                # Currently, v might dominate k of them.
                # If k == 3, we can fix it by flipping the edge to the missing one.
                # If k == 2, we need 2 flips (not a single move).
                # So we only consider k == 3 for single-edge repair.
                
                # Efficient check:
                # out_mask[v] & quad_mask
                # Let m = out_mask[v] & quad_mask
                # popcount(m) should be 3.
                # And the missing bit is the one to flip.
                
                m = out_mask[v] & quad_mask
                # Check if popcount is 3
                # Python 3.10+ has int.bit_count()
                if m.bit_count() == 3:
                    # Find the missing vertex
                    # quad_mask has 4 bits. m has 3.
                    # missing = quad_mask ^ m
                    # The vertex index is the bit position.
                    missing_mask = quad_mask ^ m
                    # Extract index
                    # Since it's a power of 2, we can use log2 or bit tricks
                    # Or just check which of a,b,c,d is not in m
                    if not (m & (1 << a)):
                        target = a
                    elif not (m & (1 << b)):
                        target = b
                    elif not (m & (1 << c)):
                        target = c
                    else:
                        target = d
                    
                    # We need to flip edge (v, target) such that v -> target
                    # Currently, if adj[v][target] is 0, we flip it to 1.
                    # If adj[v][target] is 1, it's already dominated, so this case shouldn't happen if count is 3.
                    # Wait, if adj[v][target] is 1, then target is in out_mask[v], so m would have that bit.
                    # So if m has 3 bits, the missing bit corresponds to an edge where v does NOT point to target.
                    # So we just need to set adj[v][target] = 1.
                    
                    if adj[v][target] == 0:
                        candidates.append((v, target))
            
            if not candidates:
                # No single flip can fix this quad.
                # This is a "dead end" for this specific quad with 1 flip.
                # We might need to accept a worse state or restart.
                # For now, pick a random edge flip to escape.
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                
                # Flip
                adj[i][j] = 1 - adj[i][j]
                adj[j][i] = 1 - adj[j][i]
                if adj[i][j]:
                    out_mask[i] |= (1 << j)
                    out_mask[j] &= ~(1 << i)
                else:
                    out_mask[i] &= ~(1 << j)
                    out_mask[j] |= (1 << i)
                
                new_violations, violating_quads = count_violations_and_get_list(n, adj, out_mask)
                # We don't check if it's better, we just move.
                current_violations = new_violations
                continue

            # Pick a random candidate
            v, target = random.choice(candidates)
            
            # Perform the flip
            # We know adj[v][target] is 0. We set it to 1.
            adj[v][target] = 1
            adj[target][v] = 0
            out_mask[v] |= (1 << target)
            out_mask[target] &= ~(1 << v)
            
            # Update violations
            new_violations, violating_quads = count_violations_and_get_list(n, adj, out_mask)
            
            # Accept if it doesn't increase violations too much?
            # Actually, fixing a quad should reduce violations by at least 1 (the quad itself).
            # It might create new violations.
            # If new_violations < current_violations:
            #     current_violations = new_violations
            # else:
            #     # Revert? Or keep?
            #     # Let's keep it if it's not much worse, to explore.
            #     if new_violations > current_violations + 2:
            #         # Revert
            #         adj[v][target] = 0
            #         adj[target][v] = 1
            #         out_mask[v] &= ~(1 << target)
            #         out_mask[target] |= (1 << v)
            #         # Recount
            #         current_violations, violating_quads = count_violations_and_get_list(n, adj, out_mask)
            #     else:
            #         current_violations = new_violations
            #         violating_quads = violating_quads
            #         if current_violations == 0:
            #             return adj, out_mask, True
            
            # Simpler: Just accept the move if it reduces violations.
            # If it increases, revert.
            if new_violations < current_violations:
                current_violations = new_violations
                if current_violations == 0:
                    return adj, out_mask, True
            else:
                # Revert
                adj[v][target] = 0
                adj[target][v] = 1
                out_mask[v] &= ~(1 << target)
                out_mask[target] |= (1 << v)
                # Note: violating_quads is stale here, but we'll recount next iter if needed.
                # To save time, we don't recount immediately if we revert.
                # But we need to ensure we don't loop.
                # Let's just break and restart with a different seed if stuck.
                # For now, we'll just continue the loop, but the state is unchanged.
                # To avoid infinite loop on same quad, we can shuffle or just accept the risk.
                # Actually, if we always revert when not improving, we might get stuck.
                # Let's add a "random kick" every 100 iterations if no progress.
                pass

        return adj, out_mask, current_violations == 0

    # Fallback: Paley(83)
    p = 83
    adj_paley = paley_tournament(p)
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try n from 60 to 75
    # We know f(4) >= 31, smallest known 67.
    # We will try to find a valid tournament for n < 67 if possible, 
    # but likely we will just find 67 or 68.
    
    # To speed up, we only do targeted search if initial violations are low.
    for n in range(60, 75):
        if time_left() < 10:
            break
        
        found = False
        for seed in range(100):
            if time_left() < 5:
                break
            
            # Use Paley-like structure or random?
            # Paley is not defined for all n.
            # Let's use a "Random with Bias" or just Random.
            # Actually, let's try a specific structured init:
            # Split into two sets A, B.
            # A dominates B with high prob.
            # This helps with 4-sets in B.
            
            random.seed(seed)
            adj = [[0]*n for _ in range(n)]
            
            # Random orientation
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                    else:
                        adj[j][i] = 1
            
            out_mask = get_out_masks(n, adj)
            
            # Check if valid
            if verify_tournament(n, adj, out_mask):
                best_n = n
                best_orient = adj_to_orient(n, adj)
                found = True
                break
            
            # Try targeted search
            # Limit iterations to keep time under control
            max_iters = 2000
            adj, out_mask, success = targeted_local_search(n, adj, out_mask, max_iters)
            
            if success:
                # Final verify
                if verify_tournament(n, adj, out_mask):
                    best_n = n
                    best_orient = adj_to_orient(n, adj)
                    found = True
                    break
        
        if found:
            break
    
    return {'n': best_n, 'orientation': best_orient}
