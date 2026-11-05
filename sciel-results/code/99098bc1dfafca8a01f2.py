    def synthesize_vector(vec_tuple):
        if vec_tuple in wire_map:
            return wire_map[vec_tuple]
        
        vec = np.array(vec_tuple, dtype=np.int32)
        if not np.any(vec):
            return wire_map[zero_vec]
            
        best_cost = float('inf')
        best_a_wire = -1
        best_b_tuple = None
        
        # Check if v can be formed by XOR of two existing wires
        for a_vec_tuple, a_wire in wire_map.items():
            b_vec = vec ^ np.array(a_vec_tuple, dtype=np.int32)
            b_tuple = tuple(b_vec.tolist())
            if b_tuple in wire_map:
                # Found a decomposition into two existing wires
                # Cost is 1 gate. This is optimal.
                new_wire = create_xor_gate(a_wire, wire_map[b_tuple])
                wire_map[vec_tuple] = new_wire
                return new_wire
        
        # No decomposition into two existing wires found.
        # Find the best a such that b = v XOR a has low weight.
        for a_vec_tuple, a_wire in wire_map.items():
            b_vec = vec ^ np.array(a_vec_tuple, dtype=np.int32)
            b_tuple = tuple(b_vec.tolist())
            w = np.sum(b_vec)
            if w < best_cost:
                best_cost = w
                best_a_wire = a_wire
                best_b_tuple = b_tuple
        
        # Synthesize the best b
        b_wire = synthesize_vector(best_b_tuple)
        
        # Create gate for a XOR b
        new_wire = create_xor_gate(best_a_wire, b_wire)
        wire_map[vec_tuple] = new_wire
        return new_wire
