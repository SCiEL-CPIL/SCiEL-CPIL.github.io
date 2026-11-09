"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 255
    # Numbers are 2 to 256
    # Index i corresponds to number i + 2
    
    # 1. Precompute Compatibility
    # We want a set S such that for all a, b in S, a != b, lcm(a,b) > 256.
    # This is equivalent to finding a clique in the graph G where edge (u,v) exists if lcm(u,v) > 256.
    # Or an independent set in the conflict graph H where edge (u,v) exists if lcm(u,v) <= 256.
    
    # Let's work with the Conflict Graph for easier "can add?" checks.
    # conflict_mask[i] has bit j set if i and j conflict (lcm <= 256).
    # If we have a set S (bitmask), a new vertex v can be added if (conflict_mask[v] & S) == 0.
    
    conflict_mask = [0] * N
    weights = [0.0] * N
    
    for i in range(N):
        a = i + 2
        weights[i] = 1.0 / a
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 2
            g = gcd(a, b)
            if a * b // g <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Helper to calculate weight of a bitmask
    def calc_weight(mask):
        w = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            w += weights[idx]
            m ^= lsb
        return w

    # Helper to check if a vertex can be added to a set
    def can_add(v, S_mask):
        return (conflict_mask[v] & S_mask) == 0

    # Greedy Initialization
    # Strategy: Sort vertices by weight descending.
    # Try to add each vertex if it doesn't conflict with the current set.
    # This tends to pick small numbers (high weight) first.
    # However, small numbers conflict with many other small numbers.
    # A better greedy might be to look ahead or just standard greedy.
    
    # Let's try a few different greedy strategies and keep the best.
    best_set_mask = 0
    best_weight = 0.0
    
    # Strategy 1: Weight Descending Greedy
    order = sorted(range(N), key=lambda x: -weights[x])
    current_mask = 0
    for v in order:
        if can_add(v, current_mask):
            current_mask |= (1 << v)
    w = calc_weight(current_mask)
    if w > best_weight:
        best_weight = w
        best_set_mask = current_mask

    # Strategy 2: Randomized Greedy (Multiple runs)
    # Use random order to escape local optima of the simple greedy.
    for _ in range(50):
        random.shuffle(order)
        current_mask = 0
        for v in order:
            if can_add(v, current_mask):
                current_mask |= (1 << v)
        w = calc_weight(current_mask)
        if w > best_weight:
            best_weight = w
            best_set_mask = current_mask

    # Strategy 3: "Large First" Greedy (Inspiration 2)
    # Sort by number ascending (weight descending) but maybe prioritize numbers that have fewer conflicts?
    # Actually, let's just do a weighted greedy with a tie-breaker on degree (fewer conflicts is better for expansion?).
    # Or just stick to the randomized greedy which is robust.
    
    # Let's try a "Conflict-Aware" Greedy:
    # At each step, pick the vertex with highest weight that can be added, 
    # but break ties by preferring vertices with fewer conflicts with remaining candidates?
    # This is complex to implement dynamically. 
    # Instead, let's use the Randomized Greedy result as the seed for Local Search.

    # Local Search: Variable Neighborhood Search (VNS) / Simulated Annealing style
    # Move: Swap one vertex in S with one vertex not in S.
    # Or: Add one vertex, Remove one vertex (if needed to maintain validity, but we only add valid ones).
    # Actually, since we start with a valid set, we can only do:
    # 1. Add a vertex v not in S if valid. (Always improves weight if weight > 0, which it is).
    #    Wait, if we can add a vertex, we MUST add it to maximize weight? 
    #    No, adding a vertex might prevent adding other higher weight vertices later?
    #    But we are in a local search on a *complete* valid set? 
    #    No, the greedy set is maximal w.r.t the order, but not necessarily maximal in the graph sense?
    #    Actually, if a vertex can be added, adding it increases weight. 
    #    So any optimal solution is a *maximal* independent set in the conflict graph (maximal clique in compatibility).
    #    Therefore, we should first "saturate" the set by adding any possible vertices.
    
    # Saturate the current best_set_mask
    def saturate(mask):
        changed = True
        while changed:
            changed = False
            for v in range(N):
                if not (mask & (1 << v)):
                    if can_add(v, mask):
                        mask |= (1 << v)
                        changed = True
        return mask

    best_set_mask = saturate(best_set_mask)
    best_weight = calc_weight(best_set_mask)

    # Now perform Local Search on the space of Maximal Independent Sets.
    # Moves:
    # 1. Swap: Remove u in S, Add v not in S. 
    #    Condition: After removing u, v must be compatible with S \ {u}.
    #    Gain: weight(v) - weight(u). We want positive gain.
    # 2. Triple Swap: Remove u1, u2, Add v1, v2, v3? (Too complex for now).
    
    # We will implement a simple "Best Improvement" local search with swap moves.
    # To speed it up, we can limit the number of iterations or use a timeout.
    
    current_mask = best_set_mask
    current_weight = best_weight
    
    # Precompute list of vertices for faster iteration?
    # N is 255. 
    # For each u in S, for each v not in S:
    #   Check if (conflict_mask[v] & (current_mask & ~(1<<u))) == 0
    #   If yes, and weights[v] > weights[u], it's a valid improving move.
    
    # This is O(|S| * (N-|S|)) per iteration.
    # |S| is likely around 10-20. N=255. 20 * 235 = 4700 checks.
    # Each check is a bit operation. Very fast.
    # We can loop until no improvement is found or timeout.
    
    max_iterations = 1000
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() > deadline:
            break
            
        improved = False
        best_gain = 0.0
        best_u = -1
        best_v = -1
        
        # Iterate over u in current_mask
        m_u = current_mask
        while m_u:
            lsb_u = m_u & -m_u
            u = lsb_u.bit_length() - 1
            m_u ^= lsb_u
            
            # Potential removal of u
            # The remaining set is current_mask without u
            rest_mask = current_mask & ~(1 << u)
            
            # Iterate over v NOT in current_mask
            # v must be compatible with rest_mask
            # i.e., (conflict_mask[v] & rest_mask) == 0
            # And we want weights[v] > weights[u] to ensure improvement (strictly)
            # Actually, we just want max weights[v] - weights[u].
            
            # To optimize, we can iterate v in order of descending weight and stop if weight[v] <= weight[u]
            # But we need to check compatibility.
            
            # Let's just iterate all v not in current_mask.
            m_v = ((1 << N) - 1) ^ current_mask
            while m_v:
                lsb_v = m_v & -m_v
                v = lsb_v.bit_length() - 1
                m_v ^= lsb_v
                
                # Check compatibility
                if (conflict_mask[v] & rest_mask) == 0:
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                        best_v = v
            
            # Optimization: If we found a gain, we might break early? 
            # No, we want the BEST swap in this iteration (Best Improvement LS).
            # But checking all pairs is expensive if done naively in Python loops.
            # 255*255 = 65k iterations per LS step. 1000 steps = 65M iterations.
            # Python might be too slow for 65M complex ops in 50s?
            # Let's check speed.
            # Bit operations are fast, but Python loop overhead is high.
            # We should limit the search or use a smarter heuristic.
            
            # Alternative: Only consider v with weight > weight[u].
            # And maybe only check a subset of v?
            
            # Let's break the inner loop if we are just looking for ANY improvement (First Improvement LS)
            # which is faster but less effective.
            # Or, we can pre-sort v by weight and break early if weights[v] <= weights[u].
            
        if best_gain > 1e-9:
            # Apply the best swap
            current_mask &= ~(1 << best_u)
            current_mask |= (1 << best_v)
            current_weight += best_gain
            improved = True
            iter_count += 1
        else:
            # No single swap improvement found.
            # Try a random perturbation (VNS) to escape local optimum?
            # Or try "Double Swap"?
            # Let's try a random restart or perturbation.
            # Perturbation: Remove k random vertices from S, then re-saturate.
            # k=1 is just removal, which is bad for weight.
            # k=2? Remove 2, add 3?
            
            # Let's try a simple random perturbation:
            # Pick a random vertex in S, remove it.
            # Then re-saturate.
            # This might find a different local optimum.
            
            if time.time() > deadline:
                break
                
            # Random Perturbation
            # Get list of vertices in current_mask
            verts_in_set = []
            m = current_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                verts_in_set.append(idx)
                m ^= lsb
            
            if verts_in_set:
                # Remove 1 or 2 random vertices
                k = random.choice([1, 2])
                removed = random.sample(verts_in_set, min(k, len(verts_in_set)))
                perturbed_mask = current_mask
                for r in removed:
                    perturbed_mask &= ~(1 << r)
                
                # Saturate
                perturbed_mask = saturate(perturbed_mask)
                perturbed_weight = calc_weight(perturbed_mask)
                
                if perturbed_weight > current_weight:
                    current_mask = perturbed_mask
                    current_weight = perturbed_weight
                    # Continue search from here
                else:
                    # If perturbation didn't help immediately, maybe try another?
                    # Or accept if it's close? No, we want max weight.
                    # Let's just try a few perturbations.
                    found = False
                    for _ in range(5):
                        removed = random.sample(verts_in_set, min(k, len(verts_in_set)))
                        perturbed_mask = current_mask
                        for r in removed:
                            perturbed_mask &= ~(1 << r)
                        perturbed_mask = saturate(perturbed_mask)
                        perturbed_weight = calc_weight(perturbed_mask)
                        if perturbed_weight > current_weight:
                            current_mask = perturbed_mask
                            current_weight = perturbed_weight
                            found = True
                            break
                    if not found:
                        # Stuck. Break out of local search.
                        break
            else:
                break

    # Final Result
    result_numbers = []
    m = current_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result_numbers.append(idx + 2)
        m ^= lsb

    # Verification
    for i in range(len(result_numbers)):
        for j in range(i+1, len(result_numbers)):
            a = result_numbers[i]
            b = result_numbers[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                # Fallback to valid single node
                return [2]

    if not result_numbers:
        return [2]

    return result_numbers

# EVOLVE-BLOCK-END
