# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse matrix to masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    next_wire = n + 1
    gates = []
    # memo: mask -> wire number
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n

    target_masks = set(row_masks)
    target_masks.discard(0)

    if not target_masks:
        return {'gates': [], 'outputs': [n] * m}

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # Key insight: we want to minimize total gates.
    # Each target mask needs to be expressible as XOR of available wires.
    # Strategy: greedily add intermediate terms that help most targets.

    # Represent each target as needing its mask.
    # A term t (mask) helps a target mask T if T XOR t is "simpler" than T.
    # "Simpler" = fewer bits, or already in memo.

    # We'll maintain for each target a current best expression (list of masks that XOR to target)
    # and iteratively improve by adding shared terms.

    # Initial: each target is just itself (cost = popcount - 1 gates if we synthesize directly)
    # But we can share sub-terms.

    # Better formulation:
    # We have a set of "available" masks (initially just unit vectors and 0).
    # For each target T, its cost is the minimum number of available masks whose XOR = T, minus 1 (gates).
    # Actually, if T is in available, cost is 0 extra gates (it's already a wire).
    # If T = A XOR B where A,B in available, cost is 1 gate.
    # etc.

    # Greedy approach:
    # 1. Start with available = {unit masks, 0}
    # 2. For each candidate term C (a mask not yet in available), compute how much it reduces total cost
    # 3. Add the term with maximum benefit
    # 4. Repeat until no improvement or deadline

    # To make this tractable, we only consider candidate terms that are "useful":
    # - Subsets of target masks (XORs of subsets of bits in a target)
    # - Specifically, all 2-bit and 3-bit combinations from targets

    # Let's compute candidate terms from the targets
    candidates = set()
    for T in target_masks:
        bits = [j for j in range(n) if (T >> j) & 1]
        if len(bits) >= 2:
            # Add all pairs
            for a, b in combinations(bits, 2):
                mask = (1 << a) | (1 << b)
                candidates.add(mask)
            # Add all triples if not too many
            if len(bits) <= 12:
                for a, b, c in combinations(bits, 3):
                    mask = (1 << a) | (1 << b) | (1 << c)
                    candidates.add(mask)
        # Also add the target itself if it's a single bit (already in memo)
    
    # Also add XOR of pairs of targets for potential sharing
    tlist = list(target_masks)
    if len(tlist) <= 8:
        for i in range(len(tlist)):
            for j in range(i+1, len(tlist)):
                candidates.add(tlist[i] ^ tlist[j])

    # Filter candidates: remove those already in memo and 0
    candidates.discard(0)
    candidates = candidates - set(memo.keys())

    # Sort candidates by some heuristic: prefer those with moderate popcount
    # and that appear in multiple targets
    def candidate_score(c):
        pc = c.bit_count()
        if pc < 2 or pc > 8:
            return 0
        # Count how many targets contain this as a subset
        count = 0
        for T in target_masks:
            if (c & T) == c:  # c is subset of T
                count += 1
        return count * (10 - pc)

    candidates = sorted(candidates, key=candidate_score, reverse=True)

    # Now iteratively add terms
    # For each target, track the current best decomposition
    # Decomp: a frozenset of masks that XOR to the target
    # Cost of a decomp with k masks: if all in memo, 0 extra gates; otherwise (k-1) gates

    def decomp_cost(decomp):
        """Cost in terms of additional gates needed"""
        if not decomp:
            return 0
        if len(decomp) == 1:
            mask = next(iter(decomp))
            if mask in memo:
                return 0
            # Need to synthesize this mask
            # Approximate cost: popcount - 1
            return mask.bit_count() - 1
        # Multiple masks: need len(decomp)-1 gates, plus cost of any non-memo masks
        cost = len(decomp) - 1
        for mask in decomp:
            if mask not in memo and mask != 0:
                cost += max(0, mask.bit_count() - 1)
        return cost

    # Initial decomposition: each target is just itself
    decomps = {T: frozenset([T]) for T in target_masks}

    # Iteratively try to improve
    max_iterations = 200
    for iteration in range(max_iterations):
        if elapsed() > DEADLINE - 5.0:
            break

        best_gain = 0
        best_candidate = None

        for c in candidates:
            if c in memo:
                continue
            if elapsed() > DEADLINE - 5.0:
                break

            # If we add c, how does it help each target?
            total_cost_before = 0
            total_cost_after = 0

            for T in target_masks:
                # Cost before
                total_cost_before += decomp_cost(decomps[T])

                # Cost after: try to use c in decomposition of T
                # T = c XOR (T XOR c)
                rem = T ^ c
                if rem == 0:
                    # T == c, so decomp is just {c}
                    new_cost = 0 if c in memo else (c.bit_count() - 1)
                    # But c is not in memo yet, so we'd need to synthesize it
                    # Actually if we're adding c, after adding it will be in memo
                    # So cost is 0 for this part, but we pay 1 gate to create c
                    new_decomp = frozenset([c])
                    # The cost is 0 for the decomp itself (c will be a wire)
                    # But we need to account for the gate to create c
                    total_cost_after += 0
                else:
                    # T = c XOR rem
                    # If rem is in memo or can be cheaply expressed
                    rem_cost = 0 if rem in memo else (rem.bit_count() - 1)
                    # Total: 1 gate for (c XOR rem) + rem_cost
                    # But c itself costs c.bit_count()-1 to create
                    # Net benefit compared to before
                    new_decomp = frozenset([c, rem])
                    total_cost_after += rem_cost + 1  # 1 gate for XOR, rem_cost for rem

                    # Also try: maybe rem can be further decomposed using existing memo
                    # For now, keep it simple

                # Compare with current best decomp for T
                cur_cost = decomp_cost(decomps[T])
                # The new approach: use c and rem
                new_cost = 0
                if rem == 0:
                    new_cost = 0
                else:
                    new_cost = (0 if rem in memo else rem.bit_count() - 1) + 1

                if new_cost < cur_cost:
                    total_cost_after += 0  # will be replaced below
                else:
                    total_cost_after += cur_cost

            # This calculation is getting complex. Let me simplify.

        # Simplified approach: just add candidates in order of score, 
        # and after adding each, update decompositions

        # Let me restart with a cleaner approach
        break

    # Cleaner approach: Greedy set cover-like
    # 1. For each target, the direct cost is popcount(T) - 1
    # 2. Each intermediate term c reduces cost of target T if T contains c as subset
    #    and (T XOR c) is simpler
    # 3. Greedily add terms that give maximum total cost reduction

    # Reset
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n
    next_wire = n + 1
    gates = []

    # Recompute candidates more carefully
    all_candidates = set()
    for T in target_masks:
        bits = [j for j in range(n) if (T >> j) & 1]
        if len(bits) >= 2:
            for a, b in combinations(bits, 2):
                all_candidates.add((1 << a) | (1 << b))
            if len(bits) <= 10:
                for a, b, c in combinations(bits, 3):
                    all_candidates.add((1 << a) | (1 << b) | (1 << c))

    all_candidates.discard(0)
    all_candidates -= set(memo.keys())

    # Score each candidate by how many targets it's a subset of
    cand_scores = {}
    for c in all_candidates:
        score = 0
        for T in target_masks:
            if (c & T) == c:
                score += 1
        cand_scores[c] = score

    # Sort by score descending, then by popcount ascending
    sorted_cands = sorted(all_candidates, key=lambda c: (-cand_scores[c], c.bit_count(), c))

    # Add candidates one by one if beneficial
    added_count = 0
    for c in sorted_cands:
        if elapsed() > DEADLINE - 3.0:
            break
        if c in memo:
            continue
        if cand_scores[c] < 2:
            break  # Only worth adding if shared by 2+ targets

        # Determine how to synthesize c: split into two submasks
        bits = [j for j in range(n) if (c >> j) & 1]
        if len(bits) <= 1:
            if len(bits) == 1:
                memo[c] = bits[0]
            else:
                memo[c] = n
            continue

        # Try to find a split where both parts are in memo or cheap
        best_cost = len(bits) - 1
        best_split = None

        # Balanced split
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for i, b in enumerate(bits):
            if i < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)

        cost1 = 0 if sub1 in memo else max(0, sub1.bit_count() - 1)
        cost2 = 0 if sub2 in memo else max(0, sub2.bit_count() - 1)
        total_cost = cost1 + cost2 + 1

        if total_cost < best_cost:
            best_cost = total_cost
            best_split = (sub1, sub2)

        # Try splits using existing memo terms
        for existing in list(memo.keys()):
            if existing == 0:
                continue
            if (c & existing) == existing:
                rem = c ^ existing
                if rem in memo:
                    tc = 1  # just 1 gate
                    if tc < best_cost:
                        best_cost = tc
                        best_split = (existing, rem)
                elif rem == 0:
                    memo[c] = memo[existing]
                    best_cost = -1
                    break

        if best_cost == -1:
            continue  # already handled

        if best_split is not None:
            s1, s2 = best_split
            if s1 == 0:
                memo[c] = memo[s2]
            elif s2 == 0:
                memo[c] = memo[s1]
            else:
                w = add_gate(memo[s1], memo[s2])
                memo[c] = w
        else:
            # Direct synthesis
            w = bits[0]
            for b in bits[1:]:
                w = add_gate(w, b)
            memo[c] = w

        added_count += 1

    # Now synthesize each target using available terms
    for T in target_masks:
        if elapsed() > DEADLINE - 1.0:
            break
        if T in memo:
            continue

        bits = [j for j in range(n) if (T >> j) & 1]
        if not bits:
            memo[T] = n
            continue
        if len(bits) == 1:
            memo[T] = bits[0]
            continue

        # Try to express T as XOR of 2 existing terms
        found = False
        for a in list(memo.keys()):
            if a == 0 or a == T:
                continue
            b = T ^ a
            if b == 0:
                memo[T] = memo[a]
                found = True
                break
            if b in memo:
                if a in memo and b in memo:
                    memo[T] = add_gate(memo[a], memo[b])
                    found = True
                    break

        if found:
            continue

        # Try 3-term
        if T.bit_count() <= 8:
            found = False
            existing = [k for k in memo.keys() if k != 0]
            for a in existing:
                if a == T:
                    continue
                rem1 = T ^ a
                if rem1 == 0:
                    memo[T] = memo[a]
                    found = True
                    break
                for b in existing:
                    if b == a or b == T:
                        continue
                    rem2 = rem1 ^ b
                    if rem2 == 0:
                        memo[T] = add_gate(memo[a], memo[b])
                        found = True
                        break
                    if rem2 in memo:
                        w1 = add_gate(memo[a], memo[b])
                        memo[T] = add_gate(w1, memo[rem2])
                        found = True
                        break
                if found:
                    break
            if found:
                continue

        # Fallback: direct
        w = bits[0]
        for b in bits[1:]:
            w = add_gate(w, b)
        memo[T] = w

    outputs = [memo.get(mask, n) for mask in row_masks]
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
