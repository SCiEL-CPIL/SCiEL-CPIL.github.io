import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s wall-clock

    n = 256

    # Precompute conflict neighbors: for each x, list of (y, z) such that {x,y,z} is a 3-AP
    # and x is the "middle" or one endpoint. We need: adding x to S is valid iff
    # for every (y,z) in conflict_pairs[x], not both y and z are in S.
    conflict_pairs = {i: [] for i in range(1, n + 1)}
    for b in range(2, n):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))

    # Precompute Fraction weights for exact comparison
    w = {i: Fraction(1, i) for i in range(1, n + 1)}

    # Bitmask helpers
    def bit(x):
        return 1 << (x - 1)

    def can_add(x, M):
        for y, z in conflict_pairs[x]:
            if (M & bit(y)) and (M & bit(z)):
                return False
        return True

    def get_elements(M):
        return [i for i in range(1, n + 1) if M & bit(i)]

    def sum_recip_frac(M):
        s = Fraction(0, 1)
        for i in range(1, n + 1):
            if M & bit(i):
                s += w[i]
        return s

    # ---------- Branch and Bound with Fraction Bounds ----------
    # State: (M, candidates_mask, current_sum_frac)
    # We process candidates by branching on the vertex with the highest degree in the remaining subgraph.
    # This is a structural change from the sequential 1..n order.

    # To make "highest degree" efficient, we precompute the list of vertices.
    # In the subgraph defined by candidates_mask, the "degree" or "constraint strength" of a vertex x
    # is roughly the number of conflict pairs it has within candidates_mask.
    # However, computing this dynamically is expensive.
    # Alternative structural change: Use the "branch on max weight" or "branch on most constrained" heuristic.
    # Let's stick to a robust B&B but use Fraction for the bound.

    # Upper Bound Strategy:
    # For a node with current set S (mask M) and remaining candidates C (mask C_mask),
    # the upper bound is current_sum + sum_{x in C, x compatible with S} w[x].
    # This is the "greedy upper bound" on the remaining weight.
    # To make it tighter, we can ignore conflicts among the remaining candidates themselves,
    # which is valid because the actual set must be an independent set, so it's a subset of the compatible ones.

    # We need to find a good initial lower bound (best solution found so far).
    # Greedy approach: Add vertices in order of decreasing weight (increasing index) if compatible.
    M_greedy = 0
    for x in range(1, n + 1):
        if can_add(x, M_greedy):
            M_greedy |= bit(x)
    
    best_mask = M_greedy
    best_sum = sum_recip_frac(best_mask)
    best_elements = get_elements(best_mask)

    # DFS Stack: (M, C_mask, cur_sum)
    # C_mask contains vertices that are still "available" for consideration.
    # Initially, C_mask is all vertices.
    # When we branch on x:
    #   Branch 1 (Include x): New M = M | bit(x). New C_mask = C_mask & ~bit(x).
    #                         Also, any vertex y that forms a 3-AP with x and another vertex in M
    #                         must be removed from C_mask? No, that's handled by can_add.
    #                         But we can prune C_mask to only contain vertices compatible with new M.
    #   Branch 2 (Exclude x): New M = M. New C_mask = C_mask & ~bit(x).
    
    # To speed up, we can precompute for each x, the set of vertices that conflict with x.
    # If we include x, we don't immediately remove conflicting vertices from C_mask,
    # because they might still be added if the *other* member of the AP is not in M.
    # Wait, the condition for adding y is: no (u,v) in conflict_pairs[y] such that u,v in M.
    # If we include x, M changes. So the compatibility of other vertices changes.
    
    # Let's use a recursive function with memoization or just a stack.
    # Given n=256, the state space is huge. We need aggressive pruning.
    
    # Heuristic for branching order:
    # Pick the vertex x in C_mask with the highest "impact".
    # Impact can be defined as w[x] + sum of w[y] for y in C_mask that become incompatible if x is excluded?
    # Simpler: Pick the vertex with the highest weight in C_mask.
    # Or: Pick the vertex with the most conflicts in C_mask.
    
    # Let's try picking the vertex with the highest weight in C_mask.
    # This tends to find high-weight solutions quickly.

    def find_max_weight_vertex(C_mask):
        max_w = -1
        max_v = 0
        # Iterate through set bits in C_mask
        m = C_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length()
            if w[v] > max_w:
                max_w = w[v]
                max_v = v
            m ^= lsb
        return max_v

    def compute_ub(M, C_mask, cur_sum):
        # UB = cur_sum + sum of w[x] for x in C_mask if can_add(x, M)
        ub = cur_sum
        m = C_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length()
            if can_add(v, M):
                ub += w[v]
            m ^= lsb
        return ub

    # Initial C_mask is all vertices
    full_mask = (1 << n) - 1
    # Note: bit(i) is 1 << (i-1). So full_mask for 1..n is (1<<n)-1.
    # Let's verify: bit(1) = 1. bit(256) = 1<<255.
    # (1<<256)-1 has bits 0..255 set. Correct.

    stack = [(0, full_mask, Fraction(0, 1))]
    
    nodes_visited = 0

    while stack:
        if time.time() > deadline:
            break

        M, C_mask, cur_sum = stack.pop()
        
        if C_mask == 0:
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_mask = M
                best_elements = get_elements(M)
            continue

        # Prune if current sum + max possible remaining <= best
        # Quick check: if cur_sum + sum(all weights in C_mask) <= best_sum, prune.
        # We can precompute suffix sums or just use the UB function.
        
        # Compute tight UB
        ub = compute_ub(M, C_mask, cur_sum)
        if ub <= best_sum:
            continue

        nodes_visited += 1

        # Select branching variable
        # Strategy: Pick the vertex in C_mask with the highest weight.
        x = find_max_weight_vertex(C_mask)
        
        if x == 0:
            continue

        # Branch 1: Include x
        if can_add(x, M):
            new_M = M | bit(x)
            new_C_mask = C_mask & ~bit(x)
            new_sum = cur_sum + w[x]
            # Push Exclude first, then Include, so Include is popped first (LIFO)
            # This helps find good solutions early.
            stack.append((M, C_mask & ~bit(x), cur_sum))
            stack.append((new_M, new_C_mask, new_sum))
        else:
            # Cannot include, so must exclude
            stack.append((M, C_mask & ~bit(x), cur_sum))

    # Final verification
    final_elements = get_elements(best_mask)
    elem_set = set(final_elements)
    valid = True
    for a in final_elements:
        for b in final_elements:
            if a == b:
                continue
            c = 2 * b - a
            if c in elem_set and c != a and c != b:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy if something went wrong (shouldn't happen)
        M_greedy = 0
        for x in range(1, n + 1):
            if can_add(x, M_greedy):
                M_greedy |= bit(x)
        final_elements = get_elements(M_greedy)

    if not final_elements:
        final_elements = [1]

    return final_elements
