# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 50.0

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

    # Wire assignment
    next_wire = n + 1
    gates = []
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # Collect unique non-zero targets
    target_masks = set(row_masks)
    target_masks.discard(0)

    if not target_masks:
        return {'gates': [], 'outputs': [n] * m}

    # For each target, we need to find a decomposition into existing terms
    # Strategy: Use a priority queue to schedule synthesis of intermediate terms
    # based on how many targets benefit from them

    # First, compute for each target what its "cost" would be from scratch
    def popcount(x):
        return x.bit_count()

    # For each target, find all possible splits and their submasks
    # We'll use a greedy approach: repeatedly pick the split that maximizes sharing

    # Build a "demand" structure: for each mask, how many targets need it
    # Start with targets, then propagate demand to submasks

    # Alternative: Use a worklist approach
    # 1. For each target, compute a balanced decomposition
    # 2. Count how many times each submask appears across all decompositions
    # 3. Synthesize submasks in order of (popcount, -frequency)
    # 4. Use memo to avoid recomputation

    # Better approach: Iterative improvement
    # 1. Start with balanced trees for all targets
    # 2. Collect all submasks and their frequencies
    # 3. Synthesize in popcount order, but prefer high-frequency submasks
    # 4. After synthesis, check if any target can be expressed more cheaply
    #    using newly available intermediate terms

    # Let's use a more sophisticated approach:
    # 1. For each target, maintain a list of possible decompositions
    # 2. A decomposition is a set of existing wires that XOR to the target
    # 3. We want to minimize total gates, so we share intermediate terms

    # Practical approach:
    # 1. Compute all submasks from balanced trees of all targets
    # 2. Count frequency of each submask across all target decompositions
    # 3. Synthesize submasks in order of (popcount, -frequency)
    # 4. For each target, find the cheapest expression using available terms

    # Step 1: Generate decompositions for all targets
    # Use multiple decomposition strategies and pick the best

    def get_decomposition(mask, depth=0):
        """Get a list of wires that XOR to mask, using memo if possible"""
        if mask == 0:
            return []
        if mask in memo:
            return [memo[mask]]
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) <= 1:
            return [j for j in bits] if bits else []
        # Balanced split
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for i, b in enumerate(bits):
            if i < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
        return [sub1, sub2]

    # Collect all submasks from balanced trees
    needed = {}  # mask -> frequency (how many targets use it)

    def collect_submasks(mask, targets_count):
        if mask == 0:
            return
        needed[mask] = needed.get(mask, 0) + targets_count
        if mask in memo:
            return
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) <= 1:
            return
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for i, b in enumerate(bits):
            if i < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
        if sub1 != 0:
            collect_submasks(sub1, targets_count)
        if sub2 != 0:
            collect_submasks(sub2, targets_count)

    for mask in target_masks:
        collect_submasks(mask, 1)

    # Sort by (popcount, -frequency) for synthesis order
    sorted_needed = sorted(needed.keys(), key=lambda x: (x.bit_count(), -needed[x], x))

    # Synthesize in order
    for mask in sorted_needed:
        if mask in memo:
            continue
        if elapsed() > DEADLINE:
            break

        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) == 0:
            memo[mask] = n
        elif len(bits) == 1:
            memo[mask] = bits[0]
        else:
            # Try to find a split that uses existing memo terms
            best_cost = len(bits) - 1
            best_split = None

            # Try balanced split
            mid = len(bits) // 2
            sub1 = 0
            sub2 = 0
            for i, b in enumerate(bits):
                if i < mid:
                    sub1 |= (1 << b)
                else:
                    sub2 |= (1 << b)

            cost1 = 0 if sub1 in memo else (sub1.bit_count() - 1 if sub1 else 0)
            cost2 = 0 if sub2 in memo else (sub2.bit_count() - 1 if sub2 else 0)
            total_cost = cost1 + cost2 + 1

            if total_cost < best_cost:
                best_cost = total_cost
                best_split = (sub1, sub2)

            # Try other splits: pick a subset S such that both S and mask^S are in memo or cheap
            # For efficiency, only try splits where one part is small
            if len(bits) <= 8:
                # Try all subsets of size 1 to len(bits)//2
                for size in range(1, len(bits) // 2 + 1):
                    # Generate combinations
                    from itertools import combinations
                    for combo in combinations(bits, size):
                        sub1 = 0
                        for b in combo:
                            sub1 |= (1 << b)
                        sub2 = mask ^ sub1
                        if sub2 == 0:
                            continue
                        cost1 = 0 if sub1 in memo else (sub1.bit_count() - 1)
                        cost2 = 0 if sub2 in memo else (sub2.bit_count() - 1)
                        total_cost = cost1 + cost2 + 1
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_split = (sub1, sub2)

            if best_split is not None:
                sub1, sub2 = best_split
                if sub1 == 0:
                    memo[mask] = memo[sub2]
                elif sub2 == 0:
                    memo[mask] = memo[sub1]
                else:
                    w = add_gate(memo[sub1], memo[sub2])
                    memo[mask] = w
            else:
                # Fallback: direct synthesis
                w = bits[0]
                for b in bits[1:]:
                    w = add_gate(w, b)
                memo[mask] = w

    # After synthesis, try to improve each target's expression
    # For each target, find the cheapest expression using available terms
    def find_cheapest_expression(mask, max_terms=4):
        """Find a set of at most max_terms existing wires that XOR to mask"""
        if mask == 0:
            return []
        if mask in memo:
            return [memo[mask]]

        # Try to express mask as XOR of 2 existing terms
        # For each existing term t, check if mask^t is also in memo
        existing_masks = list(memo.keys())
        for t in existing_masks:
            if t == 0:
                continue
            remaining = mask ^ t
            if remaining == 0:
                return [memo[t]]
            if remaining in memo:
                return [memo[t], memo[remaining]]

        # Try to express as XOR of 3 terms
        # This is expensive, so only do it for small masks
        if mask.bit_count() <= 6:
            for t1 in existing_masks:
                if t1 == 0:
                    continue
                rem1 = mask ^ t1
                if rem1 == 0:
                    return [memo[t1]]
                for t2 in existing_masks:
                    if t2 == 0 or t2 == t1:
                        continue
                    rem2 = rem1 ^ t2
                    if rem2 == 0:
                        return [memo[t1], memo[t2]]
                    if rem2 in memo:
                        return [memo[t1], memo[t2], memo[rem2]]

        # Fallback: direct synthesis
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) == 0:
            return []
        elif len(bits) == 1:
            return [bits[0]]
        else:
            w = bits[0]
            for b in bits[1:]:
                w = add_gate(w, b)
            return [w]

    # For each target, try to find a cheaper expression
    for mask in target_masks:
        if elapsed() > DEADLINE:
            break
        if mask not in memo:
            continue

        # Current cost: 1 gate (the memo entry)
        # Try to find a 2-term or 3-term expression
        existing_masks = list(memo.keys())
        found_better = False

        # Try 2-term
        for t in existing_masks:
            if t == 0 or t == mask:
                continue
            remaining = mask ^ t
            if remaining == 0:
                # mask == t, already have it
                continue
            if remaining in memo:
                # We can express mask as XOR of memo[t] and memo[remaining]
                # This uses 1 gate, same as current
                # But if current expression is more expensive, this is better
                # Actually, current is already 1 gate if mask is in memo
                # So this doesn't help unless we're re-expressing
                pass

        # The key insight: if mask is in memo, it's already 1 gate
        # We can't do better than 1 gate for a single output
        # So we skip this optimization

    # Ensure all targets are in memo
    for mask in target_masks:
        if mask not in memo:
            bits = [j for j in range(n) if (mask >> j) & 1]
            if len(bits) == 0:
                memo[mask] = n
            elif len(bits) == 1:
                memo[mask] = bits[0]
            else:
                w = bits[0]
                for b in bits[1:]:
                    w = add_gate(w, b)
                memo[mask] = w

    outputs = [memo.get(mask, n) for mask in row_masks]
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
