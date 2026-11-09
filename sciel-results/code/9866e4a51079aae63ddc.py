"""XOR-circuit synthesis: minimum-cost via iterative ordering + gate-drop local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 82  # safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n  # constant-zero wire

    # For each row, get the set of input indices where value is 1
    row_sets = []
    for i in range(n_rows):
        cols = np.nonzero(mat[i])[0]
        row_sets.append(frozenset(cols.tolist()))

    def build_circuit(order):
        """Build circuit processing rows in given order. Returns (gates, wire_cache, next_wire)."""
        gates = []
        next_wire = n + 1
        wire_cache = {}
        for i in range(n):
            wire_cache[frozenset([i])] = i
        wire_cache[frozenset()] = zero_wire

        def get_wire(subset):
            nonlocal next_wire
            if len(subset) == 0:
                return zero_wire
            if len(subset) == 1:
                return next(iter(subset))
            if subset in wire_cache:
                return wire_cache[subset]

            elems = sorted(subset)
            k = len(elems)

            # Try to compose from two already-cached subsets
            best_cached_split = None
            min_split_cost = float('inf')
            for cached_sub in list(wire_cache.keys()):
                if cached_sub == subset or len(cached_sub) == 0:
                    continue
                if cached_sub.issubset(subset):
                    remainder = subset - cached_sub
                    if remainder in wire_cache:
                        cost_metric = max(len(cached_sub), len(remainder))
                        if cost_metric < min_split_cost:
                            min_split_cost = cost_metric
                            best_cached_split = (cached_sub, remainder)

            if best_cached_split:
                s1, s2 = best_cached_split
                w1 = wire_cache[s1]
                w2 = wire_cache[s2]
                if w1 == w2:
                    result = zero_wire
                else:
                    gates.append([w1, w2])
                    result = next_wire
                    next_wire += 1
                wire_cache[subset] = result
                return result

            # Find best split
            if k <= 12:
                best_w1 = None
                best_w2 = None
                best_cost = float('inf')
                best_depth = float('inf')
                for mask in range(1, (1 << k) - 1):
                    if mask == 0 or mask == (1 << k) - 1:
                        continue
                    s1 = frozenset(elems[i] for i in range(k) if (mask >> i) & 1)
                    s2 = frozenset(elems[i] for i in range(k) if not ((mask >> i) & 1))
                    if len(s1) == 0 or len(s2) == 0:
                        continue
                    c1 = s1 in wire_cache
                    c2 = s2 in wire_cache
                    cost1 = 0 if c1 else max(0, len(s1) - 1)
                    cost2 = 0 if c2 else max(0, len(s2) - 1)
                    total_cost = cost1 + cost2
                    d1 = 0 if c1 else (0 if len(s1) == 1 else int(np.ceil(np.log2(len(s1)))))
                    d2 = 0 if c2 else (0 if len(s2) == 1 else int(np.ceil(np.log2(len(s2)))))
                    total_depth = 1 + max(d1, d2)
                    if total_cost < best_cost or (total_cost == best_cost and total_depth < best_depth):
                        best_cost = total_cost
                        best_depth = total_depth
                        best_w1 = s1
                        best_w2 = s2
                    if c1 and c2:
                        break
                if best_w1 is None:
                    mid = k // 2
                    best_w1 = frozenset(elems[:mid])
                    best_w2 = frozenset(elems[mid:])
            else:
                best_w1 = None
                best_w2 = None
                best_cost = float('inf')
                best_depth = float('inf')
                for mid in range(1, k):
                    s1 = frozenset(elems[:mid])
                    s2 = frozenset(elems[mid:])
                    c1 = s1 in wire_cache
                    c2 = s2 in wire_cache
                    cost1 = 0 if c1 else max(0, len(s1) - 1)
                    cost2 = 0 if c2 else max(0, len(s2) - 1)
                    total_cost = cost1 + cost2
                    d1 = 0 if c1 else (0 if len(s1) == 1 else int(np.ceil(np.log2(len(s1)))))
                    d2 = 0 if c2 else (0 if len(s2) == 1 else int(np.ceil(np.log2(len(s2)))))
                    total_depth = 1 + max(d1, d2)
                    if total_cost < best_cost or (total_cost == best_cost and total_depth < best_depth):
                        best_cost = total_cost
                        best_depth = total_depth
                        best_w1 = s1
                        best_w2 = s2
                    if c1 and c2:
                        break
                if best_w1 is None:
                    mid = k // 2
                    best_w1 = frozenset(elems[:mid])
                    best_w2 = frozenset(elems[mid:])

            w1 = get_wire(best_w1)
            w2 = get_wire(best_w2)
            if w1 == w2:
                result = zero_wire
            else:
                gates.append([w1, w2])
                result = next_wire
                next_wire += 1
            wire_cache[subset] = result
            return result

        # Process rows in given order
        for i in order:
            s = row_sets[i]
            if len(s) > 1 and s not in wire_cache:
                get_wire(s)

        return gates, wire_cache, next_wire

    def gate_drop_local_search(gates, wire_cache, next_wire, max_iters=3):
        """Try to remove redundant gates by checking if a gate's output can be
        replaced by an existing wire. Returns improved (gates, wire_cache, next_wire)."""
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # For each gate, check if its output wire is used by another gate
            # and if we can replace it
            # Build usage map: wire -> list of (gate_idx, position)
            usage = defaultdict(list)
            for gi, g in enumerate(gates):
                usage[g[0]].append((gi, 0))
                usage[g[1]].append((gi, 1))

            # Try to eliminate gates that produce a wire only used once
            # and whose inputs are both available (cached or input)
            # Actually, we try to replace a gate's output with an equivalent existing wire
            # This is complex; instead, try a simpler approach:
            # Check if any gate is redundant (its XOR equals an already-available wire)
            
            # Simpler: try to merge duplicate gates
            # Build map from (wire1, wire2) sorted -> gate_index
            gate_map = {}
            for gi, g in enumerate(gates):
                key = (min(g[0], g[1]), max(g[0], g[1]))
                if key not in gate_map:
                    gate_map[key] = gi

            # For gates that are duplicates, replace references
            dup_replacements = {}
            for key, first_gi in gate_map.items():
                # Find all gates with same key
                for gi in range(len(gates)):
                    g = gates[gi]
                    k2 = (min(g[0], g[1]), max(g[0], g[1]))
                    if k2 == key and gi != first_gi:
                        dup_replacements[gi] = first_gi

            if not dup_replacements:
                # Try another heuristic: for each gate, check if we can express
                # its output using two other available wires (not necessarily its inputs)
                # This is expensive; skip for now
                break

            # Apply replacements
            new_gates = []
            wire_remap = {}
            for gi, g in enumerate(gates):
                if gi in dup_replacements:
                    # This gate is a duplicate; map its output to the first gate's output
                    first_gi = dup_replacements[gi]
                    # The output wire of gate gi is n+1+gi
                    wire_remap[n + 1 + gi] = n + 1 + first_gi
                else:
                    new_gates.append(g)

            # Remap all wire references
            if wire_remap:
                remapped_gates = []
                for g in new_gates:
                    new_g = [wire_remap.get(g[0], g[0]), wire_remap.get(g[1], g[1])]
                    remapped_gates.append(new_g)
                
                # Remap wire_cache
                new_wire_cache = {}
                for k, v in wire_cache.items():
                    new_wire_cache[k] = wire_remap.get(v, v)
                
                # Remap outputs
                # We need to track outputs separately
                gates = remapped_gates
                wire_cache = new_wire_cache
                next_wire = n + 1 + len(gates)
                improved = True

            if not improved:
                break

        return gates, wire_cache, next_wire

    # Initial order: sort by popcount ascending
    non_zero_rows = [i for i in range(n_rows) if len(row_sets[i]) > 0]
    
    # Strategy 1: popcount ascending
    order1 = sorted(non_zero_rows, key=lambda i: len(row_sets[i]))
    gates1, cache1, nw1 = build_circuit(order1)
    
    # Strategy 2: popcount descending
    order2 = sorted(non_zero_rows, key=lambda i: -len(row_sets[i]))
    gates2, cache2, nw2 = build_circuit(order2)
    
    # Strategy 3: iterative greedy - repeatedly pick the row that can be
    # completed with fewest new gates (i.e., whose complement is most available)
    def greedy_order():
        """Iteratively pick rows that maximize one-gate completions."""
        remaining = set(non_zero_rows)
        order = []
        # Simulate cache building
        sim_cache = {frozenset([i]) for i in range(n)}
        sim_cache.add(frozenset())
        
        while remaining:
            if time.time() > DEADLINE:
                # Fallback to popcount
                order.extend(sorted(remaining, key=lambda i: len(row_sets[i])))
                break
            
            best_row = None
            best_score = float('inf')
            
            for i in remaining:
                s = row_sets[i]
                if s in sim_cache:
                    # Can be completed with 0 new gates
                    best_row = i
                    best_score = 0
                    break
                # Count how many sub-subsets are in cache
                elems = sorted(s)
                k = len(elems)
                # Estimate: check if any split has both parts in cache
                found_split = False
                if k <= 10:
                    for mask in range(1, (1 << k) - 1):
                        s1 = frozenset(elems[j] for j in range(k) if (mask >> j) & 1)
                        s2 = frozenset(elems[j] for j in range(k) if not ((mask >> j) & 1))
                        if s1 in sim_cache and s2 in sim_cache:
                            found_split = True
                            break
                if found_split:
                    best_row = i
                    best_score = 1
                    break
                # Score by popcount (smaller = easier)
                score = len(s)
                if score < best_score:
                    best_score = score
                    best_row = i
            
            if best_row is None:
                best_row = min(remaining, key=lambda i: len(row_sets[i]))
            
            order.append(best_row)
            remaining.discard(best_row)
            
            # Update sim_cache: add the row's subset
            sim_cache.add(row_sets[best_row])
            # Also add all sub-subsets that would be created
            # (approximate: just add the row itself)
        
        return order

    order3 = greedy_order()
    gates3, cache3, nw3 = build_circuit(order3)

    # Pick the best result
    candidates = [(len(gates1), gates1, cache1, nw1),
                  (len(gates2), gates2, cache2, nw2),
                  (len(gates3), gates3, cache3, nw3)]
    candidates.sort(key=lambda x: x[0])
    
    best_gates = candidates[0][1]
    best_cache = candidates[0][2]
    best_nw = candidates[0][3]

    # Apply gate-drop local search
    if time.time() < DEADLINE:
        best_gates, best_cache, best_nw = gate_drop_local_search(best_gates, best_cache, best_nw)

    # Build outputs
    outputs = []
    for i in range(n_rows):
        s = row_sets[i]
        if len(s) == 0:
            outputs.append(zero_wire)
        elif s in best_cache:
            outputs.append(best_cache[s])
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            # Should not happen, but fallback
            outputs.append(zero_wire)

    return {'gates': best_gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
