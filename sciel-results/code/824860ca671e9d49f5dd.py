    def xor_wires(wires):
        if not wires:
            return n
        if len(wires) == 1:
            return wires[0]
        # Balanced binary tree
        mid = len(wires) // 2
        left = xor_wires(wires[:mid])
        right = xor_wires(wires[mid:])
        return new_gate(left, right)
