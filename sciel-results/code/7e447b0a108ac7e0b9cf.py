def synthesize(matrix, depth_limit):
    import time
    start_time = time.time()
    deadline = start_time + 80

    # Parse matrix
    if isinstance(matrix, list):
        rows = [list(row) for row in matrix]
        n_cols = len(rows[0]) if rows else 0
        n_rows = len(rows)
    else:
        import numpy as np
        mat = np.array(matrix)
        n_rows, n_cols = mat.shape
        rows = mat.tolist()

    n = n_cols
    targets = []
    for row in rows:
        v = 0
        for i, bit in enumerate(row):
            if bit:
                v |= (1 << i)
        targets.append(v)

    # Available wires
    wire_to_vec = {i: (1 << i) for i in range(n)}
    wire_to_vec[n] = 0
    vec_to_wire = {(1 << i): i for i in range(n)}
    vec_to_wire[0] = n

    gates = []
    next_wire = n + 1

    produced = {}
    for i, t in enumerate(targets):
        if t in vec_to_wire:
            produced[i] = vec_to_wire[t]

    remaining = [i for i in range(len(targets)) if i not in produced]

    while remaining:
        if time.time() > deadline:
            break

        available_wires = list(wire_to_vec.keys())

        best_c = None
        best_score = -1
        best_pair = None

        for i in range(len(available_wires)):
            a = available_wires[i]
            va = wire_to_vec[a]
            for j in range(i + 1, len(available_wires)):
                b = available_wires[j]
                vb = wire_to_vec[b]
                c = va ^ vb

                if c == 0:
                    continue

                score = 0
                for ti in remaining:
                    t = targets[ti]
                    if t == c:
                        score += 1000
                    else:
                        old_pop = t.bit_count()
                        new_pop = (t ^ c).bit_count()
                        if new_pop < old_pop:
                            score += (old_pop - new_pop)

                if score > best_score:
                    best_score = score
                    best_c = c
                    best_pair = (a, b)

        if best_c is None or best_score <= 0:
            for ti in remaining:
                t = targets[ti]
                bits = [k for k in range(n) if (t >> k) & 1]
                if not bits:
                    produced[ti] = n
                elif len(bits) == 1:
                    produced[ti] = bits[0]
                else:
                    current_wire = bits[0]
                    for k in range(1, len(bits)):
                        if time.time() > deadline:
                            break
                        gates.append([current_wire, bits[k]])
                        new_vec = wire_to_vec[current_wire] ^ (1 << bits[k])
                        vec_to_wire[new_vec] = next_wire
                        wire_to_vec[next_wire] = new_vec
                        current_wire = next_wire
                        next_wire += 1
                    produced[ti] = current_wire
            break

        a, b = best_pair
        if best_c not in vec_to_wire:
            gates.append([a, b])
            vec_to_wire[best_c] = next_wire
            wire_to_vec[next_wire] = best_c
            next_wire += 1

        new_remaining = []
        for ti in remaining:
            t = targets[ti]
            if t in vec_to_wire:
                produced[ti] = vec_to_wire[t]
            else:
                new_remaining.append(ti)
        remaining = new_remaining

    outputs = []
    for i in range(len(targets)):
        if i in produced:
            outputs.append(produced[i])
        else:
            t = targets[i]
            bits = [k for k in range(n) if (t >> k) & 1]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                current_wire = bits[0]
                for k in range(1, len(bits)):
                    gates.append([current_wire, bits[k]])
                    new_vec = wire_to_vec[current_wire] ^ (1 << bits[k])
                    vec_to_wire[new_vec] = next_wire
                    wire_to_vec[next_wire] = new_vec
                    current_wire = next_wire
                    next_wire += 1
                outputs.append(current_wire)

    return {'gates': gates, 'outputs': outputs}
