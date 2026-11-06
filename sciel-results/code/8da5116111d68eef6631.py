"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # First, filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build conflict graph using bitsets (Python ints as bitsets)
    # conflict_mask[i] is a bitmask of all valid numbers that conflict with valid_numbers[i]
    conflict_mask = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                conflict_mask[i] |= (1 << j)

    # Also set self-conflict? No, we handle a<=b including a=b already filtered.
    # The conflict is only for a != b pairs where a+b is square.
    # But we need to ensure that when we pick a node, we remove all its neighbors.

    # Branch and bound for Maximum Independent Set
    # We use a recursive approach with bitsets for speed.
    # State: candidate set as a bitmask, current selected count, current best.
    
    best_size = [0]
    best_set = [set()]
    
    # Upper bound: for a set of candidates, the max independent set size <= 1 + max_degree_in_candidates
    # Actually a tighter bound: greedy coloring bound or simply: size <= number of candidates
    # But we can use: in the subgraph induced by candidates, max IS <= 1 + max degree
    # Even better: use the fact that if we pick a vertex v, we get 1 + MIS(candidates \ N[v])
    # And MIS(candidates) <= max over v in candidates of (1 + MIS(candidates \ N[v]))
    # This is exact but exponential. We need a good pruning bound.
    
    # A simple but effective bound: MIS(candidates) <= size(candidates)
    # Better: MIS(candidates) <= 1 + max_{v in candidates} |N(v) ∩ candidates|  ... no that's wrong
    # The correct bound from the literature: MIS <= 1 + max degree in the subgraph
    # Actually: for any vertex v, MIS(G) <= 1 + MIS(G - N[v])
    # So MIS(G) <= max_v (1 + MIS(G - N[v]))
    # For pruning: if 1 + popcount(candidates) <= best_size, prune.
    # Better bound: use greedy coloring on the subgraph to get an upper bound.
    
    # Let's use a simpler approach: 
    # Upper bound = 1 + max degree in the candidate subgraph
    # This is valid because: pick a vertex v with max degree d. 
    # Any independent set either contains v (then at most 1 + MIS(cand \ N[v])) 
    # or doesn't contain v (then at most MIS(cand \ {v}))
    # But a simpler valid bound: MIS <= 1 + max_degree (for connected components)
    # Actually the standard bound is: MIS(G) <= 1 + max_{v} deg(v) is NOT generally true.
    # The correct simple bound: MIS(G) <= |V| and for a vertex v: MIS(G) <= 1 + MIS(G - N[v])
    
    # Let me use a different, well-known approach:
    # Tomita's algorithm for maximum clique (which is MIS on complement)
    # But our graph is sparse, so let's work directly on MIS.
    
    # Key insight: the conflict graph is sparse. Each number conflicts with at most
    # sqrt(512) ~ 22 others. So max degree is small (~22).
    
    # Use recursive branch and bound:
    # - Order candidates by degree (ascending) for better pruning
    # - Bound: if current_size + popcount(candidates) <= best_size, prune
    # - Pick a vertex, branch on include/exclude
    
    # Actually, for sparse graphs with max degree ~22 and 256 nodes, a well-implemented
    # branch and bound should be fast. Let's use the standard approach:
    
    # Precompute degree in the full graph for ordering
    degrees = [bin(conflict_mask[i]).count('1') for i in range(num_valid)]
    
    # Sort indices by degree (ascending) - process low-degree vertices first
    order = sorted(range(num_valid), key=lambda i: degrees[i])
    
    # We'll work with the original indices but use the ordering for branching
    # Actually, let's just use a bitmask and branch on the lowest-index vertex in the candidate set
    
    def popcount(x):
        return bin(x).count('1')
    
    # Map from sorted position to original index for faster iteration
    # Actually, let's just iterate through candidates in the bitmask
    
    def solve(candidates, current_size, current_set):
        """Branch and bound for MIS. candidates is a bitmask."""
        if time.time() > deadline:
            return
        
        if candidates == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = set(current_set)
            return
        
        # Upper bound: current_size + popcount(candidates)
        if current_size + popcount(candidates) <= best_size[0]:
            return
        
        # Pick a vertex to branch on. Use the one with highest degree in the subgraph
        # for better pruning, or lowest for simpler code.
        # Let's pick the lowest set bit for simplicity, but we can improve.
        
        # Find a vertex in candidates
        # Use a heuristic: pick vertex with max degree in the candidate subgraph
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            # Degree of v in the candidate subgraph
            deg = popcount(conflict_mask[v] & candidates)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        # New candidates = candidates minus best_v and all its neighbors
        new_candidates = candidates & ~(conflict_mask[best_v] | (1 << best_v))
        new_set = list(current_set) + [best_v]
        solve(new_candidates, current_size + 1, new_set)
        
        # Branch 2: Exclude best_v
        new_candidates2 = candidates & ~(1 << best_v)
        solve(new_candidates2, current_size, current_set)
    
    # Initial candidates: all valid numbers
    all_candidates = (1 << num_valid) - 1
    solve(all_candidates, 0, [])
    
    if time.time() > deadline:
        # Fallback if we timed out mid-search
        if best_size[0] == 0:
            # Return a simple valid set
            return [valid_numbers[0]]
        return sorted([valid_numbers[i] for i in best_set[0]])
    
    result = sorted([valid_numbers[i] for i in best_set[0]])
    return result
# EVOLVE-BLOCK-END
