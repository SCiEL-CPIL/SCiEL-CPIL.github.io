"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 80  # safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n  # constant-zero wire

    # For each row, get the set of input indices where value is 1
    row_sets = []
    for i in range(n_rows):
        cols = np.nonzero(mat[i])[0]
        row_sets.append(frozenset(cols.tolist()))

    # Identify independent rows (basis) using GF(2) row reduction
    # We'll track which rows are independent by checking linear independence
    def is_dependent(set_s, basis_sets):
        """Check if set_s is in the span of basis_sets over GF(2)."""
        # Try to express set_s as XOR of some subset of basis_sets
        # Use greedy: try to cancel elements
        remaining = set(set_s)
        for bs in basis_sets:
            bset = set(bs)
            # Check if bset is subset of remaining (XOR would cancel)
            # Actually, in GF(2), we can XOR bset into remaining if it helps
            # Simple approach: if bset ⊆ remaining, XOR it out
            if bset <= remaining:
                remaining ^= bset
        return len(remaining) == 0

    # Build basis greedily
    basis_indices = []
    basis_sets = []
    for i in range(n_rows):
        s = row_sets[i]
        if len(s) == 0:
            continue  # zero row, skip
        # Check if s is dependent on current basis
        dependent = False
        # Try all subsets of basis (exponential but basis is small)
        # Better: use the is_dependent check
        if is_dependent(s, basis_sets):
            dependent = True
        if not dependent:
            basis_indices.append(i)
            basis_sets.append(s)

    # Separate rows into basis and dependent
    dependent_indices = [i for i in range(n_rows) if i not in set(basis_indices)]

    # Gate management
    gates = []
    next_wire = n + 1
    wire_cache = {}  # frozenset -> wire index

    def get_wire(subset):
        """Get wire for XOR of all elements in subset, building tree with cache reuse."""
        nonlocal next_wire
        if len(subset) == 0:
            return zero_wire
        if len(subset) == 1:
            return next(iter(subset))
        if subset in wire_cache:
            return wire_cache[subset]

        elems = sorted(subset)
        k = len(elems)

        # Try to find a split that maximizes cache reuse
        # Try all possible split points
        best_w1 = None
        best_w2 = None
        best_cost = float('inf')

        for mid in range(1, k):
            s1 = frozenset(elems[:mid])
            s2 = frozenset(elems[mid:])

            # Check if either is cached
            c1 = s1 in wire_cache
            c2 = s2 in wire_cache

            # Cost: number of new gates needed
            # If cached, cost is 0 for that part
            # Otherwise, estimate by popcount - 1 (chain) or better
            cost1 = 0 if c1 else max(0, len(s1) - 1)
            cost2 = 0 if c2 else max(0, len(s2) - 1)
            total_cost = cost1 + cost2

            if total_cost < best_cost:
                best_cost = total_cost
                best_w1 = s1
                best_w2 = s2

            # Early exit if we found a perfect split (both cached)
            if c1 and c2:
                break

        s1 = best_w1
        s2 = best_w2

        w1 = get_wire(s1)
        w2 = get_wire(s2)

        if w1 == w2:
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1

        wire_cache[subset] = result
        return result

    # Phase 1: Synthesize basis rows (independent rows) first
    # Sort by popcount ascending to build small sets first
    basis_sorted = sorted(basis_indices, key=lambda i: len(row_sets[i]))
    for i in basis_sorted:
        get_wire(row_sets[i])

    # Phase 2: For dependent rows, try to match as XOR of two cached wires
    # Build a lookup: wire -> frozenset (inverse of wire_cache for non-input wires)
    # Actually, we need: for a target set S, find A, B in wire_cache such that A ^ B = S
    # i.e., B = A ^ S, and both A and B are in wire_cache

    # Create set of all cached sets
    cached_sets = set(wire_cache.keys())
    # Also include singleton sets (input wires)
    for i in range(n):
        cached_sets.add(frozenset([i]))

    def find_two_sum(target):
        """Find two cached sets A, B such that A ^ B = target."""
        if target in cached_sets:
            # Could be A ^ empty = target, but empty is zero_wire
            # Check if target itself is a single cached wire
            if target in wire_cache:
                return (wire_cache[target], zero_wire)  # but this is 0 gates... 
            # Actually if target is in wire_cache, we just return that wire directly
            # This case is handled before calling find_two_sum
            pass
        
        # Try all cached sets A, check if A ^ target is also cached
        # For efficiency, iterate over smaller set
        for a in cached_sets:
            b = frozenset(a ^ target)
            if b in cached_sets:
                # Get wires
                wa = wire_cache.get(a, next(iter(a)) if len(a) == 1 else None)
                wb = wire_cache.get(b, next(iter(b)) if len(b) == 1 else None)
                if wa is None or wb is None:
                    # Need to handle input wires
                    if len(a) == 1:
                        wa = next(iter(a))
                    if len(b) == 1:
                        wb = next(iter(b))
                if wa is not None and wb is not None:
                    if wa == wb:
                        return (zero_wire, zero_wire)  # shouldn't happen for non-empty target
                    return (wa, wb)
        return None

    # Process dependent rows
    for i in dependent_indices:
        target = row_sets[i]
        if len(target) == 0:
            continue  # zero row, output zero_wire
        
        # First check if target is already cached
        if target in wire_cache:
            continue  # already have a wire for it
        
        # Try two-sum match
        result = find_two_sum(target)
        if result is not None:
            wa, wb = result
            if wa == wb:
                # XOR with self = 0
                wire_cache[target] = zero_wire
            else:
                gates.append([wa, wb])
                wire_cache[target] = next_wire
                next_wire += 1
        else:
            # Fallback: build tree with cache reuse
            get_wire(target)

    # Build outputs
    outputs = []
    for i in range(n_rows):
        s = row_sets[i]
        if len(s) == 0:
            outputs.append(zero_wire)
        elif s in wire_cache:
            outputs.append(wire_cache[s])
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            # Should not happen, but guard
            outputs.append(get_wire(s))

    # Check deadline
    if time.time() > DEADLINE:
        pass

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
