"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Find all bad triples
    # A triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Build "bad pair" relation: pair (a,b) is "bad" if there exists c such that
    # (a,b,c) is a bad triple. In the independent set formulation, two elements
    # CANNOT both be in the set if they form a bad pair (because the third element
    # might also be in the set).
    #
    # Wait - that's not quite right. Two elements CAN be in the set together as long
    # as no THIRD element in the set completes a bad triple. The constraint is
    # on TRIPLES, not pairs.
    #
    # So this is NOT a simple graph independent set problem. It's a 3-uniform
    # hypergraph independent set problem.
    #
    # However, we can still use branch and bound effectively. Let me think of
    # a different structural approach.
    #
    # ALTERNATIVE: Use the fact that bad triples have a specific structure.
    # If lcm(a,b) = lcm(a,c) = lcm(b,c) = L, then:
    #   - L is a common multiple of a, b, c
    #   - For each pair, their LCM is exactly L
    #
    # Key insight: If a, b, c all divide L and lcm(a,b)=lcm(a,c)=lcm(b,c)=L,
    # then each pair's LCM equals L. This means:
    #   - a | L, b | L, c | L
    #   - lcm(a,b) = L means a*b/gcd(a,b) = L
    #
    # For a given L, the set of numbers in [1,128] that divide L is limited.
    # Let's group by LCM value.
    #
    # Actually, let me use a different structural approach:
    # The key observation is that for a bad triple (a,b,c), all three must divide
    # the same L = lcm(a,b) = lcm(a,c) = lcm(b,c).
    #
    # For each possible L, let D(L) = {x in [1,128] : x | L and x > 1} (excluding 1
    # since lcm(1,x)=x which is rarely equal to lcm of other pairs).
    # Actually 1 can be in bad triples: lcm(1,a)=a, lcm(1,b)=b, lcm(a,b)=L.
    # For this to equal: a = b = L, but a != b. So 1 cannot be in a bad triple
    # unless... wait: lcm(1,a)=a, lcm(1,b)=b, lcm(a,b)=L.
    # We need a = b = L, impossible for distinct a,b.
    # So 1 is NEVER in a bad triple! Great, 1 is always safe to include.
    #
    # For L, the numbers dividing L in [1,128]: let's call this divisors_L.
    # A triple from divisors_L is bad iff lcm of each pair equals L.
    #
    # Strategy: For each L, find all "bad triples" within divisors_L.
    # Then the problem decomposes somewhat by L.
    #
    # But elements can participate in bad triples for multiple L values.
    #
    # Let me just do an efficient branch-and-bound directly.
    #
    # APPROACH: Branch and bound with the following:
    # 1. Order elements by "danger" (number of bad triples they participate in)
    # 2. At each step, branch on include/exclude
    # 3. Upper bound: current_size + remaining_elements (optimistic)
    # 4. Better upper bound: current_size + maximum independent set of remaining
    #    (approximated by greedy coloring bound)
    #
    # For the 3-uniform hypergraph, a valid set S means:
    #   For every bad triple (a,b,c), NOT all three are in S.
    #
    # This is equivalent to: the set S is an independent set in the 3-uniform
    # hypergraph H where edges are bad triples.
    #
    # Branch and bound:
    # - Maintain candidate set C (elements not yet decided)
    # - Maintain current set S (elements included)
    # - Upper bound: |S| + |C| (trivial)
    # - Better: |S| + alpha(H_C) where H_C is the hypergraph induced by C
    #   (upper bounded by |C|, or by a greedy bound)
    #
    # For efficiency with n=128, I'll use:
    # - Bitmask representation for C
    # - Precomputed: for each element x, the set of "forbidden pairs"
    # - When we include x, we don't immediately remove anything from C,
    #   but we note that for any pair (u,v) in forbidden_pairs[x],
    #   we cannot include BOTH u and v.
    #
    # This is getting complex. Let me use a simpler but effective approach:
    #
    # APPROACH 2: Iterative deepening with smart pruning
    # - Try to find a set of size k for k = 128, 127, ...
    # - Use the constraint that no bad triple is fully contained
    #
    # APPROACH 3 (most practical): Use the structure of bad triples.
    #
    # Let me first count how many bad triples there are and understand the structure.

    # Count bad triples and build structure
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            if lab > 128:
                # c must divide lab, but c <= 128
                # Still possible if lab has small divisors
                pass
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # Build for each element, the list of bad triples it's in
    triples_by_element = {i: [] for i in range(1, n + 1)}
    for (a, b, c) in bad_triples:
        triples_by_element[a].append((b, c))
        triples_by_element[b].append((a, c))
        triples_by_element[c].append((a, b))

    # Number of bad triples
    num_bad = len(bad_triples)

    # For each element, count how many bad triples it's in
    danger = {i: len(triples_by_element[i]) for i in range(1, n + 1)}

    # 1 is always safe (never in a bad triple)
    # Elements with danger 0 are always safe

    always_safe = [i for i in range(1, n + 1) if danger[i] == 0]
    dangerous = [i for i in range(1, n + 1) if danger[i] > 0]

    # Sort dangerous elements by danger (descending) for branching
    dangerous_sorted = sorted(dangerous, key=lambda x: -danger[x])

    # Branch and bound
    # State:
    #   S_mask: bitmask of included elements
    #   C_mask: bitmask of candidate elements (not yet decided)
    #   For each x in S, we track which pairs are "blocked" (cannot both be added)
    #
    # Actually, let me use a cleaner formulation:
    # We process elements in a fixed order (dangerous_sorted + always_safe).
    # At position i, we decide to include or exclude element dangerous_sorted[i].
    #
    # Constraint check: If we include x, then for every (u,v) in triples_by_element[x],
    # we cannot include both u and v (if both are still candidates or already included).
    #
    # For pruning: if we've already included both u and v for some forbidden pair,
    # then x CANNOT be included.
    #
    # Upper bound: current_size + number of remaining candidates that are "compatible"
    # (i.e., can potentially be added without violating constraints with S)
    #
    # Let me implement this with bitmasks for speed.

    # Map element to bit position
    elem_to_bit = {i: i - 1 for i in range(1, n + 1)}

    # Precompute for each element x: bitmask of "partner pairs"
    # partner_mask[x] is a dict: for each pair (u,v) in triples_by_element[x],
    # we need to track if both u and v are in S.
    #
    # Alternative: For each x, precompute a list of (u_bit, v_bit) pairs.
    # When checking if x can be added to S:
    #   For each (u_bit, v_bit): if (S_mask >> u_bit) & 1 and (S_mask >> v_bit) & 1:
    #       x cannot be added.

    # This is O(danger[x]) per check. Total work in B&B is manageable.

    # Process order: always_safe first (they're free), then dangerous in order of danger
    process_order = always_safe + dangerous_sorted
    # Actually, let's interleave: process all in a fixed order
    # For B&B, we process in order and branch include/exclude.

    # Let me use a different, cleaner B&B:
    # We maintain:
    #   S: set of included elements (as bitmask)
    #   blocked_pairs: set of (u,v) pairs that are both in S (these block certain elements)
    #   Actually, we just check on the fly.
    #
    # For the upper bound, I'll use:
    #   |S| + (number of remaining elements that are not immediately blocked by S)
    # An element x is "immediately blocked" if there exists (u,v) in triples_by_element[x]
    # such that both u and v are in S.

    best_size = len(always_safe)  # At minimum, all always_safe elements are in
    best_mask = 0
    for i in always_safe:
        best_mask |= (1 << elem_to_bit[i])

    # Try to add dangerous elements
    # Greedy initial solution
    greedy_mask = best_mask
    for x in dangerous_sorted:
        can_add = True
        for u, v in triples_by_element[x]:
            if ((greedy_mask >> elem_to_bit[u]) & 1) and ((greedy_mask >> elem_to_bit[v]) & 1):
                can_add = False
                break
        if can_add:
            greedy_mask |= (1 << elem_to_bit[x])

    best_size = bin(greedy_mask).count('1')
    best_mask = greedy_mask

    # Now B&B to try to improve
    # We'll process dangerous elements in order.
    # At each step, branch: include or exclude.
    #
    # To make this efficient, I'll use a recursive approach with:
    # - Pruning by upper bound
    # - Forward checking: when we include x, mark which future elements are now blocked
    #
    # Let me use an iterative deepening approach instead:
    # Try to find a solution of size k for k from best_size+1 upward.
    #
    # Actually, for n=128 with ~hundreds of bad triples, B&B should work well
    # if we have good pruning. Let me implement it carefully.

    # For the B&B, I'll use the following structure:
    # - Process elements in a fixed order (dangerous_sorted)
    # - Maintain S (bitmask of included dangerous elements; always_safe are always in)
    # - Maintain a "blocked" set: elements that cannot be added because S already
    #   contains both partners of some forbidden pair for that element.
    #
    # When we include x:
    #   For each other dangerous element y not yet processed:
    #     Check if y is now blocked (i.e., there's a pair (u,v) in triples_by_element[y]
    #     where u=x and v is in S, or v=x and u is in S).
    #   If y is blocked, remove it from candidates.
    #
    # Upper bound: |S| + |candidates| + |always_safe|
    #   (always_safe are always included)
    #   Wait, always_safe are already counted. Let me restructure.
    #
    # Total size = |always_safe| + |included_dangerous|
    # Upper bound = |always_safe| + |included_dangerous| + |remaining_candidates|

    # Precompute: for each pair of dangerous elements (x, y), do they form a "conflict"?
    # x and y conflict if there exists z such that (x, y, z) is a bad triple.
    # In that case, if z is in S, then x and y cannot both be in S.
    # But this is a 3-way constraint, not a pairwise one.
    #
    # The constraint is: for each bad triple (a,b,c), at most 2 of {a,b,c} can be in S.
    #
    # This is a 3-SAT-like constraint. B&B should still work.

    # Let me implement a practical B&B:

    # Order: process dangerous elements from most dangerous to least
    # For each, try include first (since we want max size), then exclude.

    # Precompute for fast checking:
    # For each element x, store triples_by_element[x] as list of (u_bit, v_bit)
    triples_bits = {}
    for x in range(1, n + 1):
        triples_bits[x] = [(elem_to_bit[u], elem_to_bit[v]) for u, v in triples_by_element[x]]

    def can_include(x, S_mask):
        """Check if x can be added to S (given always_safe are in S)."""
        for ub, vb in triples_bits[x]:
            if (S_mask >> ub) & 1 and (S_mask >> vb) & 1:
                return False
        return True

    # B&B with time limit
    best_size_bb = best_size
    best_mask_bb = best_mask
    nodes_visited = 0

    def upper_bound(S_mask, candidates_mask):
        """Upper bound on final size given current S and remaining candidates."""
        # Count how many candidates are not immediately blocked
        # A candidate y is blocked if there's (u,v) in triples_by_element[y]
        # with both u,v in S_mask.
        # For speed, just use |S| + |candidates| as a loose bound.
        # Better: count unblocked candidates.
        s_count = bin(S_mask).count('1')
        # For a tighter bound, count candidates that aren't blocked
        # But this is expensive. Use loose bound for now.
        c_count = bin(candidates_mask).count('1')
        return s_count + c_count

    def bb(S_mask, idx, candidates_mask):
        """Branch and bound. idx is index into dangerous_sorted."""
        nonlocal best_size_bb, best_mask_bb, nodes_visited

        if time.time() > deadline - 3:
            return

        nodes_visited += 1

        current_size = bin(S_mask).count('1')
        remaining = bin(candidates_mask).count('1')

        # Prune: even adding all remaining can't beat best
        if current_size + remaining <= best_size_bb:
            return

        if idx >= len(dangerous_sorted):
            if current_size > best_size_bb:
                best_size_bb = current_size
                best_mask_bb = S_mask
            return

        x = dangerous_sorted[idx]
        x_bit = elem_to_bit[x]
        new_candidates = candidates_mask & ~(1 << x_bit)

        # Option 1: Include x (try first for maximization)
        if can_include(x, S_mask):
            new_S = S_mask | (1 << x_bit)
            # Forward checking: remove from candidates any element that is now blocked
            # An element y is blocked if there's (u,v) in triples_by_element[y]
            # where {u,v} contains x and the other is in new_S.
            # i.e., for each (u,v) in triples_by_element[y]:
            #   if (u==x and v in new_S) or (v==x and u in new_S): y is blocked
            #
            # For efficiency, precompute: for each x, which elements y have x in one
            # of their forbidden pairs?
            # Actually, let's just check on the fly for candidates.
            #
            # Simpler: just recurse and let can_include filter.
            # The pruning by upper bound will handle most cases.
            bb(new_S, idx + 1, new_candidates)

        # Option 2: Exclude x
        bb(S_mask, idx + 1, new_candidates)

    # Start B&B from the greedy solution's decisions
    # Actually, let's start fresh: S starts with always_safe, candidates = all dangerous
    initial_S = best_mask  # always_safe bits set
    initial_candidates = 0
    for x in dangerous_sorted:
        initial_candidates |= (1 << elem_to_bit[x])

    # To improve pruning, let's start with a good lower bound from greedy
    # and use it for pruning.

    # Run B&B
    bb(initial_S, 0, initial_candidates)

    # If B&B didn't finish or didn't improve, try local search from best found
    if best_size_bb > best_size:
        best_size = best_size_bb
        best_mask = best_mask_bb

    # Local search: try to improve by swaps
    current_mask = best_mask
    current_set = set(i + 1 for i in range(n) if (current_mask >> i) & 1)

    # Try 1-swap improvements
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in list(current_set):
            if time.time() > deadline - 5:
                break
            trial_mask = current_mask & ~(1 << elem_to_bit[x])
            trial_set = set(current_set)
            trial_set.discard(x)

            # Try to add elements back (greedy)
            for y in dangerous_sorted:
                if y in trial_set:
                    continue
                if y in always_safe:
                    continue  # always_safe should always be in
                if can_include(y, trial_mask):
                    trial_mask |= (1 << elem_to_bit[y])
                    trial_set.add(y)

            if len(trial_set) > len(current_set):
                current_mask = trial_mask
                current_set = trial_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_mask = current_mask
                improved = True

    # Try adding elements that were excluded
    for y in range(1, n + 1):
        if y in current_set:
            continue
        if can_include(y, current_mask):
            current_mask |= (1 << elem_to_bit[y])
            current_set.add(y)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_mask = current_mask

    # Try random restarts with perturbation
    import random
    for _ in range(200):
        if time.time() > deadline - 5:
            break

        # Start from best, remove 3-5 random elements, greedy refill
        pert_mask = best_mask
        pert_set = set(i + 1 for i in range(n) if (pert_mask >> i) & 1)

        # Remove some random dangerous elements
        removable = [x for x in pert_set if x not in always_safe]
        k = min(len(removable), random.randint(2, 5))
        for _ in range(k):
            if not removable:
                break
            x = random.choice(removable)
            removable.remove(x)
            pert_set.remove(x)
            pert_mask &= ~(1 << elem_to_bit[x])

        # Greedy refill
        for y in dangerous_sorted:
            if y in pert_set:
                continue
            if can_include(y, pert_mask):
                pert_mask |= (1 << elem_to_bit[y])
                pert_set.add(y)

        # Also try adding always_safe (shouldn't be needed, they're always safe)
        for y in always_safe:
            if y not in pert_set:
                pert_set.add(y)
                pert_mask |= (1 << elem_to_bit[y])

        if len(pert_set) > best_size:
            best_size = len(pert_set)
            best_mask = pert_mask

    # Final: ensure all always_safe are included
    for y in always_safe:
        best_mask |= (1 << elem_to_bit[y])

    result = sorted(i + 1 for i in range(n) if (best_mask >> i) & 1)

    # Validate: check no bad triple is fully contained
    result_set = set(result)
    for (a, b, c) in bad_triples:
        if a in result_set and b in result_set and c in result_set:
            # This shouldn't happen, but if it does, remove one element
            result_set.remove(c)
            result = sorted(result_set)
            break

    if not result:
        result = [1]

    return result

# EVOLVE-BLOCK-END
