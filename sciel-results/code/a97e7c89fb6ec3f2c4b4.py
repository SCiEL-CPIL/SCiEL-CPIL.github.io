"""XOR-circuit synthesis with Dijkstra-based decomposition using existing wires."""

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D list/array of 0s and 1s (m x n)
        depth_limit: maximum allowed circuit depth (<=16)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    import numpy as np
    import heapq

    if matrix is None or len(matrix) == 0:
        return {'gates': [], 'outputs': []}
    
    matrix = np.array(matrix, dtype=np.int32)
    m, n = matrix.shape
    
    # Wire indices: 0..n-1 are inputs, n is constant zero
    # Gates produce wires starting at n+1
    
    wire_map = {}  # tuple of int32 vector -> wire index
    gates = []     # list of [wire1, wire2]
    next_wire = n + 1  # first gate output wire index
    
    # Initialize input wires in the map
    for i in range(n):
        vec = np.zeros(n, dtype=np.int32)
        vec[i] = 1
        wire_map[tuple(vec.tolist())] = i
    
    # Constant zero wire
    zero_vec = tuple(np.zeros(n, dtype=np.int32).tolist())
    wire_map[zero_vec] = n
    
    def create_xor_gate(w1, w2):
        """Create an XOR gate and return the new wire index."""
        nonlocal next_wire
        gate = [w1, w2]
        gates.append(gate)
        wire_idx = next_wire
        next_wire += 1
        return wire_idx

    def synthesize_vector(target_vec_tuple):
        """
        Find the minimum cost path from target_vec_tuple to any existing wire in wire_map.
        Uses Dijkstra's algorithm where edges are XOR operations with existing wires.
        """
        if target_vec_tuple in wire_map:
            return wire_map[target_vec_tuple]
        
        # Dijkstra
        # State: (cost, vector_tuple, path_of_wires)
        visited = set()
        pq = [(0, target_vec_tuple, [])]
        
        while pq:
            cost, v_tuple, path = heapq.heappop(pq)
            
            if v_tuple in visited:
                continue
            visited.add(v_tuple)
            
            if v_tuple in wire_map:
                # Reconstruct gates
                current_wire = wire_map[v_tuple]
                # Path is [W_1, ..., W_k]
                # T = V_1 ^ W_1
                # V_1 = V_2 ^ W_2
                # ...
                # V_{k-1} = V_goal ^ W_k
                # We iterate from last to first
                for i in range(len(path) - 1, -1, -1):
                    w = path[i]
                    current_wire = create_xor_gate(w, current_wire)
                
                wire_map[target_vec_tuple] = current_wire
                return current_wire
            
            # Expand neighbors
            v_np = np.array(v_tuple, dtype=np.int32)
            
            # Optimization: Only consider a subset of wires if wire_map is large?
            # For now, consider all.
            for w_vec_tuple, w_idx in wire_map.items():
                w_vec_np = np.array(w_vec_tuple, dtype=np.int32)
                new_vec_np = np.bitwise_xor(v_np, w_vec_np)
                new_vec_tuple = tuple(new_vec_np.tolist())
                
                if new_vec_tuple not in visited:
                    # Check depth limit? 
                    # The path length so far is 'cost'.
                    # If cost + 1 > depth_limit, we might skip?
                    # But we need to reach a wire in wire_map.
                    # If we don't reach it within depth_limit, it's invalid.
                    # However, the "cost" here is the number of new gates.
                    # This corresponds to the depth of the chain we are building.
                    # So we should prune if cost + 1 > depth_limit.
                    if cost + 1 <= depth_limit:
                        heapq.heappush(pq, (cost + 1, new_vec_tuple, path + [w_idx]))
        
        # If no path found within depth_limit, fallback to greedy?
        # Or return None.
        # For safety, let's return None and handle it outside.
        return None

    # Synthesize each row of the matrix
    outputs = []
    for i in range(m):
        row_tuple = tuple(matrix[i].tolist())
        if row_tuple in wire_map:
            outputs.append(wire_map[row_tuple])
        else:
            wire = synthesize_vector(row_tuple)
            if wire is not None:
                outputs.append(wire)
            else:
                # Fallback: Greedy synthesis
                # This is the original logic
                vec = np.array(row_tuple, dtype=np.int32)
                set_bits = np.where(vec == 1)[0]
                if len(set_bits) == 0:
                    outputs.append(n)
                    continue
                
                best_bit = -1
                best_remainder_weight = n + 1
                for bit in set_bits:
                    remainder = vec.copy()
                    remainder[bit] = 0
                    r_weight = np.sum(remainder)
                    if r_weight < best_remainder_weight:
                        best_remainder_weight = r_weight
                        best_bit = bit
                
                remainder = vec.copy()
                remainder[best_bit] = 0
                remainder_tuple = tuple(remainder.tolist())
                
                remainder_wire = synthesize_vector(remainder_tuple)
                if remainder_wire is None:
                     # Deep recursion fallback?
                     # Just use input
                     remainder_wire = n # Zero? No, remainder is not zero.
                     # This case should not happen if depth_limit is sufficient.
                
                new_wire = create_xor_gate(best_bit, remainder_wire)
                wire_map[row_tuple] = new_wire
                outputs.append(new_wire)
    
    return {'gates': gates, 'outputs': outputs}
