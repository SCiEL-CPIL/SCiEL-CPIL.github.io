import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0  # 54s safety margin out of 60s

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    A = np.array(matrix, dtype=np.uint8) % 2

    # Compute RREF with row tracking
    M = A.copy()
    T = np.eye(m, dtype=np.uint8)

    rank = 0
    pivot_cols = []

    for col in range(n):
        if time.time() > deadline:
            break
        pivot = -1
        for row in range(rank, m):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue

        M[[rank, pivot]] = M[[pivot, rank]]
        T[[rank, pivot]] = T[[pivot, rank]]

        for row in range(m):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                T[row] ^= T[rank]

        pivot_cols.append(col)
        rank += 1
        if rank == m:
            break

    # Compute T^{-1}
    aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
    for col in range(m):
        if time.time() > deadline:
            break
        pivot = -1
        for row in range(col, m):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(m):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    T_inv = aug[:, m:]

    # Build XOR gates with global caching
    gates = []
    wire_cache = {}  # frozenset of input wires -> output wire number

    def get_xor_wire(wire_set):
        """Get or create a wire that is the XOR of all wires in wire_set."""
        if not wire_set:
            return n
        if len(wire_set) == 1:
            return wire_set.pop() if isinstance(wire_set, set) else wire_set[0]

        key = frozenset(wire_set)
        if key in wire_cache:
            return wire_cache[key]

        # Try to find a cached subset to reduce work
        # For small sets, just build a balanced tree
        wire_list = sorted(wire_set) if isinstance(wire_set, set) else list(wire_set)

        if len(wire_list) <= 2:
            if len(wire_list) == 1:
                result = wire_list[0]
            else:
                g = len(gates)
                gates.append([wire_list[0], wire_list[1]])
                result = n + 1 + g
            wire_cache[key] = result
            return result

        # Divide and conquer
        mid = len(wire_list) // 2
        left_set = set(wire_list[:mid])
        right_set = set(wire_list[mid:])

        left = get_xor_wire(left_set)
        right = get_xor_wire(right_set)

        g = len(gates)
        gates.append([left, right])
        result = n + 1 + g
        wire_cache[key] = result
        return result

    # Step 1: For each pivot row, compute the wire from input wires
    pivot_wires = []
    for i in range(rank):
        if time.time() > deadline:
            # Fallback: simple linear XOR
            ones = [j for j in range(n) if M[i, j] == 1]
            if not ones:
                pivot_wires.append(n)
            elif len(ones) == 1:
                pivot_wires.append(ones[0])
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    g = len(gates)
                    gates.append([current, ones[j]])
                    current = n + 1 + g
                pivot_wires.append(current)
        else:
            ones = [j for j in range(n) if M[i, j] == 1]
            pivot_wires.append(get_xor_wire(ones))

    # Step 2: Express each output as XOR of pivot wires
    outputs = []
    for j in range(m):
        if time.time() > deadline:
            # Fallback: simple linear XOR
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            if not contributing:
                outputs.append(n)
            elif len(contributing) == 1:
                outputs.append(pivot_wires[contributing[0]])
            else:
                current = pivot_wires[contributing[0]]
                for i in range(1, len(contributing)):
                    g = len(gates)
                    gates.append([current, pivot_wires[contributing[i]]])
                    current = n + 1 + g
                outputs.append(current)
        else:
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            if not contributing:
                outputs.append(n)
            elif len(contributing) == 1:
                outputs.append(pivot_wires[contributing[0]])
            else:
                outputs.append(get_xor_wire([pivot_wires[i] for i in contributing]))

    return {'gates': gates, 'outputs': outputs}
