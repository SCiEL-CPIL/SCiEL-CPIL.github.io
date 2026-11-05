def synthesize(matrix, depth_limit):
    import time
    start = time.time()
    DEADLINE = 84.0

    m = len(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}
    n = len(matrix[0])

    targets = []
    for row in matrix:
        mask = 0
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        targets.append(mask)

    wires = []      # list of (mask, depth)
    wire_index = {} # mask -> (index, depth)
    for j in range(n):
        wires.append((1 << j, 0))
        wire_index[1 << j] = (len(wires)-1, 0)
    wires.append((0, 0))
    wire_index[0] = (n, 0)

    gates = []
    def create_gate(a, b):
        ma, da = wires[a]
        mb, db = wires[b]
        mm = ma ^ mb
        dd = max(da, db) + 1
        if mm in wire_index:
            return wire_index[mm][0]
        idx = len(wires)
        wires.append((mm, dd))
        wire_index[mm] = (idx, dd)
        gates.append((a, b))
        return idx

    def avail_indices():
        return list(range(len(wires)))

    # Precompute which targets contain a given mask as subset: we'll compute on the fly.

    # Phase 1: proactive shared wires
    rem = set(targets)
    for t in list(rem):
        if t in wire_index:
            rem.discard(t)

    max_proactive = 60
    for _ in range(max_proactive):
        if time.time() > DEADLINE:
            break
        if not rem:
            break
        avail = avail_indices()
        best_net = 0
        best_pair = None
        rem_list = list(rem)
        for i in range(len(avail)):
            ai = avail[i]
            mi = wires[ai][0]
            if mi == 0:
                continue
            for j in range(i+1, len(avail)):
                aj = avail[j]
                mj = wires[aj][0]
                if mj == 0:
                    continue
                w = mi ^ mj
                if w == 0 or w in wire_index:
                    continue
                # net benefit
                benefit = 0
                for t in rem_list:
                    if w == t:
                        benefit += (t.bit_count() - 1) * 2
                    elif (w & ~t) == 0:
                        benefit += (w.bit_count() - 1)
                net = benefit - 1
                if net > best_net:
                    best_net = net
                    best_pair = (ai, aj, w)
        if best_pair is None:
            break
        ai, aj, w = best_pair
        create_gate(ai, aj)
        for t in list(rem):
            if t in wire_index:
                rem.discard(t)

    # Phase 2: build each target sequentially
    # sort by popcount desc
    order = sorted(range(len(targets)), key=lambda i: targets[i].bit_count(), reverse=True)
    for ti in order:
        t = targets[ti]
        if t in wire_index:
            continue
        if time.time() > DEADLINE:
            break
        remaining = t
        chosen = []
        while remaining != 0:
            # if remaining available, use it
            if remaining in wire_index:
                chosen.append(wire_index[remaining][0])
                break
            # find best available subset wire
            best_s = None
            best_pc = 0
            for idx in range(len(wires)):
                s = wires[idx][0]
                if s == 0:
                    continue
                if (s & ~remaining) == 0 and s != remaining:
                    pc = s.bit_count()
                    if pc > best_pc:
                        best_pc = pc
                        best_s = idx
            if best_s is None:
                # use a basis input in remaining
                # find lowest set bit
                lb = (remaining & -remaining).bit_length() - 1
                best_s = lb  # wire index for input lb
            chosen.append(best_s)
            remaining ^= wires[best_s][0]
        # build t from chosen
        if len(chosen) == 1:
            pass  # already available
        else:
            acc = chosen[0]
            for s in chosen[1:]:
                acc = create_gate(acc, s)
            # acc should be t
            # ensure t in wire_index
            if t not in wire_index:
                # fallback
                pass

    # Ensure all targets have wires
    outputs = []
    for t in targets:
        if t in wire_index:
            outputs.append(wire_index[t][0])
        else:
            # build from basis
            bits = [j for j in range(n) if (t >> j) & 1]
            if not bits:
                outputs.append(n)
            else:
                acc = wires[bits[0]][0]
                for b in bits[1:]:
                    acc = create_gate(acc, wires[b][0])
                outputs.append(acc)

    # Depth fix
    for idx in range(len(outputs)):
        t = targets[idx]
        widx = outputs[idx]
        if wires[widx][1] > depth_limit:
            bits = [j for j in range(n) if (t >> j) & 1]
            if not bits:
                outputs[idx] = n
            else:
                acc = wires[bits[0]][0]
                for b in bits[1:]:
                    acc = create_gate(acc, wires[b][0])
                outputs[idx] = acc

    return {'gates': gates, 'outputs': outputs}
