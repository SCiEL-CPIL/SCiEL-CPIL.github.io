"""Shared XOR-circuit synthesis: Global Dijkstra-based sequential synthesis."""
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    import numpy as np
    import heapq

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

    # Dijkstra-based sequential synthesis
    # State: vector (int)
    # Available vectors: set of ints
    # We process outputs in order of increasing Hamming weight (heuristic)
    
    outputs_indices = list(range(m))
    # Sort by Hamming weight of the target mask
    outputs_indices.sort(key=lambda i: bin(row_masks[i]).count('1'))
    
    available = set()
    # Inputs are available
    for i in range(n):
        available.add(1 << i)
    available.add(0)
    
    gates = []
    next_wire = n + 1
    wire_to_val = {} # wire index -> vector mask
    val_to_wire = {} # vector mask -> wire index
    
    # Initialize wires for inputs and 0
    for i in range(n):
        wire_to_wire = i
        val = 1 << i
        wire_to_val[i] = val
        val_to_wire[val] = i
    wire_to_val[n] = 0
    val_to_wire[0] = n
    
    outputs = [None] * m
    
    for idx in outputs_indices:
        target = row_masks[idx]
        if target in val_to_wire:
            outputs[idx] = val_to_wire[target]
            continue
        
        # Run Dijkstra from 0 to target using 'available' vectors
        # dist[v] = min gates to reach v
        # prev[v] = (prev_v, used_vec)
        size = 1 << n
        dist = [float('inf')] * size
        prev = [None] * size
        dist[0] = 0
        pq = [(0, 0)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if u == target:
                break
            if d > dist[u]:
                continue
            for v in available:
                next_u = u ^ v
                nd = d + 1
                if nd < dist[next_u]:
                    dist[next_u] = nd
                    prev[next_u] = (u, v)
                    heapq.heappush(pq, (nd, next_u))
        
        if dist[target] == float('inf'):
            # Should not happen if inputs are basis
            break
            
        # Reconstruct path
        # Path is a sequence of vectors.
        # We need to generate any new vectors in the path.
        path_vecs = []
        curr = target
        while curr != 0:
            prev_u, used_v = prev[curr]
            path_vecs.append(curr)
            curr = prev_u
        path_vecs.reverse()
        
        # Now we have a list of vectors to generate: path_vecs
        # For each new vector in path_vecs, we need to add a gate.
        # But we must ensure the operands are already available.
        # The path construction ensures that to reach `curr` from `prev_u`, we used `used_v`.
        # So `curr = prev_u ^ used_v`.
        # We need to generate `curr` if it's not already available.
        
        # The path is 0 -> v1 -> v2 -> ... -> target
        # v1 = 0 ^ avail_1
        # v2 = v1 ^ avail_2
        # ...
        # So we need to generate v1, v2, ..., target.
        
        # Iterate through the generated vectors in order
        curr = 0
        for vec in path_vecs:
            if vec not in val_to_wire:
                # Find the previous vector and the available vector used
                # We stored prev[vec] = (prev_u, used_v)
                prev_u, used_v = prev[vec]
                
                # Get wires for prev_u and used_v
                # prev_u is either 0 or a previously generated vector
                # used_v is in 'available'
                
                w_prev = val_to_wire[prev_u]
                w_used = val_to_wire[used_v]
                
                # Create new gate
                gates.append([w_prev, w_used])
                new_wire = next_wire
                next_wire += 1
                wire_to_val[new_wire] = vec
                val_to_wire[vec] = new_wire
                
                # Add to available
                available.add(vec)
            else:
                # Already available, just update available set if needed (it is)
                pass
        
        outputs[idx] = val_to_wire[target]

    if verify(gates, outputs):
        return {'gates': gates, 'outputs': outputs}

    # fallback
    gates, outputs = build_independent()
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
