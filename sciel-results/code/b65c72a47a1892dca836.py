# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire assignment:
    # Inputs: 0 .. n-1
    # Constant zero: n
    # Gates produce wires: n+1, n+2, ...

    gates = []
    next_wire = n + 1

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # Build a circuit that computes all row_masks with minimal gates
    # Strategy: Find a minimal set of "basis" vectors that can generate all rows,
    # then use CSE to share sub-expressions.

    # Step 1: Gaussian elimination to find row space basis
    # We want to identify which rows are linearly independent
    basis = []  # list of (mask, row_index)
    basis_masks = set()
    for i in range(m):
        mask = row_masks[i]
        if mask == 0:
            continue
        # Reduce by existing basis
        reduced = mask
        for b_mask in basis_masks:
            if reduced & b_mask:
                # Find highest set bit of b_mask
                hb = b_mask.bit_length() - 1
                if (reduced >> hb) & 1:
                    reduced ^= b_mask
        if reduced != 0:
            # Add to basis (reduce to canonical form)
            # Find pivot
            hb = reduced.bit_length() - 1
            # Reduce existing basis by this new vector
            new_basis = []
            for b in basis:
                if (b[0] >> hb) & 1:
                    new_basis.append((b[0] ^ reduced, b[1]))
                else:
                    new_basis.append(b)
            new_basis.append((reduced, i))
            basis = new_basis
            basis_masks = {b[0] for b in basis}

    # Step 2: For each row, express it as XOR of basis vectors
    # Build a mapping from row index to XOR of basis vectors
    # First, build basis as a list of masks in a fixed order
    basis_list = [b[0] for b in basis]
    k = len(basis_list)

    # For each row, find which basis vectors XOR to it
    row_to_basis = []
    for i in range(m):
        mask = row_masks[i]
        if mask == 0:
            row_to_basis.append(0)  # zero vector
            continue
        # Reduce mask by basis
        basis_indices = []
        temp = mask
        for bi, bmask in enumerate(basis_list):
            # Check if we need this basis vector
            hb = bmask.bit_length() - 1
            if (temp >> hb) & 1:
                basis_indices.append(bi)
                temp ^= bmask
        # temp should be 0
        # Encode as bitmask over basis indices
        bm = 0
        for bi in basis_indices:
            bm |= (1 << bi)
        row_to_basis.append(bm)

    # Step 3: Now we need to compute k basis vectors and then combine them for each row.
    # But we can do better: use CSE over all target vectors (the basis vectors themselves
    # and the combinations).

    # Actually, let's think differently. The optimal approach for XOR circuits:
    # Each gate computes XOR of two existing wires. We want to compute all row_masks
    # with minimum gates.

    # A known result: the minimum number of XOR gates to compute a set of linear
    # functions is at least the rank of the matrix (of coefficients) minus 1... 
    # but actually it's related to the minimum number of additions in a straight-line program.

    # For our purposes, let's use a practical approach:
    # 1. Compute a basis for the row space (k vectors)
    # 2. For each basis vector, build an XOR tree from its input bits
    # 3. For each row, if it's not a basis vector, XOR the appropriate basis wires

    # But we can share sub-expressions between basis vectors.

    # Let's use a CSE approach:
    # - Start with all input wires available
    # - Repeatedly find pairs of wires that appear in multiple targets and merge them

    # Simpler approach: Build a circuit by processing rows in order,
    # but use memoization to share any sub-expression.

    # Let's try a different structural approach: 
    # Use the fact that we can compute intermediate XORs and reuse them.

    # Build circuit for all rows using a worklist algorithm with CSE:
    # 1. Identify all "needed" vectors (the row_masks)
    # 2. For each needed vector, if it's an input, done.
    #    Otherwise, find two sub-vectors that XOR to it, and are "easier" to compute.
    # 3. Use a priority queue to process vectors by "cost" (popcount).

    # Memo: mask -> wire
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def compute_wire(mask, deadline):
        """Compute wire for mask using CSE-like greedy with memo."""
        if mask in memo:
            return memo[mask]
        if mask == 0:
            return n

        if elapsed() > deadline:
            # Fallback: balanced tree
            bits = [j for j in range(n) if (mask >> j) & 1]
            return build_tree(bits)

        # Find best split: mask = a ^ b where a, b are "simpler"
        # Try all possible a that are subsets of mask
        # To be efficient, try a with popcount roughly half of mask
        pc = mask.bit_count()
        if pc <= 1:
            # Single bit or zero
            if pc == 1:
                j = mask.bit_length() - 1
                memo[mask] = j
                return j
            else:
                memo[mask] = n
                return n

        # Try to find a in memo such that b = mask ^ a is also "easy"
        # First, check if any single-bit XOR works
        for j in range(n):
            bit = 1 << j
            if mask & bit:
                other = mask ^ bit
                if other in memo:
                    w = add_gate(memo[other], j)
                    memo[mask] = w
                    return w

        # Try to find two memoized vectors that XOR to mask
        # Limit search for efficiency
        memo_keys = list(memo.keys())
        best_a = None
        best_b = None
        best_cost = float('inf')

        # Sample if memo is large
        if len(memo_keys) > 200:
            import random
            random.seed(42)
            sample = random.sample(memo_keys, 200)
        else:
            sample = memo_keys

        for a in sample:
            if a == 0 or a == mask:
                continue
            b = mask ^ a
            if b in memo:
                cost_a = a.bit_count()
                cost_b = b.bit_count()
                cost = cost_a + cost_b
                if cost < best_cost:
                    best_cost = cost
                    best_a = a
                    best_b = b
                    if cost <= 2:
                        break

        if best_a is not None:
            w = add_gate(memo[best_a], memo[best_b])
            memo[mask] = w
            return w

        # Fallback: split into two halves
        bits = [j for j in range(n) if (mask >> j) & 1]
        mid = len(bits) // 2
        a_bits = bits[:mid]
        b_bits = bits[mid:]
        a_mask = 0
        for j in a_bits:
            a_mask |= (1 << j)
        b_mask = 0
        for j in b_bits:
            b_mask |= (1 << j)

        w_a = compute_wire(a_mask, deadline)
        w_b = compute_wire(b_mask, deadline)
        w = add_gate(w_a, w_b)
        memo[mask] = w
        return w

    def build_tree(bits):
        """Build balanced XOR tree from input indices."""
        if not bits:
            return n
        if len(bits) == 1:
            return bits[0]
        if len(bits) == 2:
            return add_gate(bits[0], bits[1])
        mid = len(bits) // 2
        w1 = build_tree(bits[:mid])
        w2 = build_tree(bits[mid:])
        return add_gate(w1, w2)

    # Process all rows
    outputs = [0] * m
    for i in range(m):
        mask = row_masks[i]
        if elapsed() > DEADLINE - 3.0:
            # Fast fallback
            if mask in memo:
                outputs[i] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                outputs[i] = build_tree(bits)
                memo[mask] = outputs[i]
        else:
            outputs[i] = compute_wire(mask, DEADLINE - 5.0)

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
