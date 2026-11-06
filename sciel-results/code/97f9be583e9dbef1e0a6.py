def synthesize(matrix, depth_limit):
    import numpy as np

    mat = np.array(matrix, dtype=np.uint8)
    m, n = mat.shape

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Get input indices for each output row
    output_sets = []
    for i in range(m):
        idx = np.where(mat[i])[0].tolist()
        output_sets.append(idx)

    # Build circuit with memoization on input subsets
    gates = []
    wire_cache = {}
    next_wire = n + 1

    def compute_xor(inputs):
        nonlocal next_wire
        if not inputs:
            return n  # constant zero
        if len(inputs) == 1:
            return inputs[0]

        key = frozenset(inputs)
        if key in wire_cache:
            return wire_cache[key]

        # Chain: XOR first input with XOR of the rest
        first = inputs[0]
        rest = inputs[1:]
        rest_wire = compute_xor(rest)

        wire = next_wire
        gates.append([first, rest_wire])
        next_wire += 1
        wire_cache[key] = wire
        return wire

    outputs = []
    for idx in output_sets:
        wire = compute_xor(idx)
        outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}
