"""Shared XOR-circuit synthesis: global 2-bit edge sharing (Steiner-style)."""
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    import numpy as np
    import heapq
    from collections import defaultdict

    A = np.asarray(matrix)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    m, n = A.shape
    depth_limit = 16 if depth_limit is None else int(depth_limit)
    depth_limit = min(depth_limit, 16)

    # rows = outputs, cols = inputs (standard linear-map convention y = A x)
    row_masks = []
    for r in range(m):
        mk = 0
        row = A[r]
        for c in range(n):
            if row[c]:
                mk |= (1 << c)
        row_masks.append(mk)

    def build_independent():
        gates = []
        next_wire = n + 1
        outputs = []
        for mk in row_masks:
            bits = [c for c in range(n) if (mk >> c) & 1]
            if not bits:
                outputs.append(n)
                continue
            if len(bits) == 1:
                outputs.append(bits[0])
                continue
            cur = bits[0]
            for b in bits[1:]:
                gates.append([cur, b])
                cur = next_wire
                next_wire += 1
            outputs.append(cur)
        return gates, outputs

    def verify(gates, outputs):
        val = list(range(n)) + [0]
        for a, b in gates:
            val.append(val[a] ^ val[b])
        for j, ow in enumerate(outputs):
            if ow >= len(val) or val[ow] != row_masks[j]:
                return False
        depth = [0] * len(val)
        for k, (a, b) in enumerate(gates):
            depth[n + 1 + k] = max(depth[a], depth[b]) + 1
        if max(depth) > depth_limit:
            return False
        return True

    # ---- Dijkstra-based optimal XOR circuit synthesis ----
    # Each node is a bitmask representing a linear combination of inputs.
    # Wire 0..n-1 correspond to single-bit masks (1<<c).
    # Wire n is the zero mask (0).
    # Gate k produces wire n+1+k.
    
    # We want to find the minimum number of gates to realize all output masks.
    # Use Dijkstra from all input wires (and zero) to each output mask.
    # State: (cost, mask, depth) - but depth is constrained.
    # Since we want to minimize gate count, we use cost = number of gates used.
    # However, we need to track which wires have been created to avoid double-counting.
    # This is a Steiner tree problem on the hypergraph of XOR operations.
    
    # Simplified approach: Use Dijkstra where each state is a mask, and the cost is the minimum number of gates needed to produce that mask.
    # But this doesn't capture sharing across multiple outputs.
    
    # Better approach: Build a shared circuit using a greedy algorithm that prioritizes high-frequency subexpressions.
    
    # For now, use a frequency-based greedy approach:
    # 1. Count how many times each bit appears in all row masks.
    # 2. Identify common 2-bit subexpressions (pairs of bits that appear together frequently).
    # 3. Build gates for these shared subexpressions first.
    # 4. Then complete each output using the available wires.
    
    # Count bit frequencies
    bit_freq = [0] * n
    pair_freq = defaultdict(int)
    for mk in row_masks:
        bits = [c for c in range(n) if (mk >> c) & 1]
        for c in bits:
            bit_freq[c] += 1
        for i in range(len(bits)):
            for j in range(i+1, len(bits)):
                a, b = bits[i], bits[j]
                key = (min(a,b), max(a,b))
                pair_freq[key] += 1
    
    # Sort pairs by frequency (descending)
    sorted_pairs = sorted(pair_freq.items(), key=lambda x: -x[1])
    
    # Build shared gates for high-frequency pairs
    used_pairs = set()
    shared_gates = []  # (pair, wire_index)
    next_wire = n + 1
    pair_to_wire = {}
    
    # We need to be careful: using a pair gate changes the structure.
    # Let's use a different approach: iterative improvement.
    
    # Start with independent circuit
    gates, outputs = build_independent()
    
    # Try to merge gates that compute the same XOR
    # Group gates by their input pair
    from collections import Counter
    gate_pairs = Counter()
    for a, b in gates:
        key = (min(a,b), max(a,b))
        gate_pairs[key] += 1
    
    # For gates with same input pair, we can share them
    # But this requires restructuring the circuit.
    
    # Simpler: Use the existing greedy approach but with better pair selection
    used_edges = set()
    row_terms = []
    for mk in row_masks:
        bits = [c for c in range(n) if (mk >> c) & 1]
        terms = []
        rem = list(bits)
        # Sort remaining bits by frequency to prioritize high-frequency bits
        rem.sort(key=lambda c: -bit_freq[c])
        while len(rem) >= 2:
            i = rem[0]
            chosen = None
            # Prefer pairs that are high-frequency globally
            best_score = -1
            best_idx = None
            for idx in range(1, len(rem)):
                j = rem[idx]
                key = (i, j) if i < j else (j, i)
                score = pair_freq.get(key, 0)
                if key in used_edges:
                    score += 1000  # Strong preference for already-used edges
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                best_idx = 1
            idx, j = best_idx, rem[best_idx]
            key = (i, j) if i < j else (j, i)
            terms.append(('edge', key))
            used_edges.add(key)
            rem.pop(idx)
            rem.pop(0)
        if rem:
            terms.append(('single', rem[0]))
        row_terms.append(terms)

    edges = sorted(used_edges)
    edge_wire = {}
    gates = []
    next_wire = n + 1
    for e in edges:
        i, j = e
        edge_wire[e] = next_wire
        gates.append([i, j])
        next_wire += 1

    outputs = []
    for terms in row_terms:
        wires = []
        for t in terms:
            if t[0] == 'edge':
                wires.append(edge_wire[t[1]])
            else:
                wires.append(t[1])
        if not wires:
            outputs.append(n)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            cur = wires[0]
            for w in wires[1:]:
                gates.append([cur, w])
                cur = next_wire
                next_wire += 1
            outputs.append(cur)

    if verify(gates, outputs):
        return {'gates': gates, 'outputs': outputs}

    # fallback
    gates, outputs = build_independent()
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
