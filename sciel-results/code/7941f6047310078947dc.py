def synthesize(matrix, depth_limit):
    import time
    from collections import defaultdict

    start_time = time.time()
    DEADLINE = 85.0

    n = matrix.shape[0] if hasattr(matrix, 'shape') else len(matrix)

    rows = []
    for i in range(n):
        s = set()
        for j in range(n):
            if matrix[i][j] != 0:
                s.add(j)
        rows.append(frozenset(s))

    row_terms = [set(r) for r in rows]
    gates = []

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def total_cost():
        cost = len(gates)
        for terms in row_terms:
            if len(terms) > 1:
                cost += len(terms) - 1
        return cost

    baseline_cost = sum(len(r) - 1 for r in rows if len(r) > 1)

    best_cost = baseline_cost
    best_gates = []
    best_row_terms = [set(r) for r in rows]

    while time_left() > 5.0:
        pair_count = defaultdict(int)
        for terms in row_terms:
            if len(terms) < 2:
                continue
            tlist = sorted(terms)
            for i in range(len(tlist)):
                for j in range(i + 1, len(tlist)):
                    pair_count[(tlist[i], tlist[j])] += 1

        best_pair = None
        best_count = 1
        for pair, cnt in pair_count.items():
            if cnt > best_count:
                best_count = cnt
                best_pair = pair

        if best_pair is None or best_count < 2:
            break

        a, b = best_pair
        new_wire = n + len(gates)
        gates.append([a, b])

        for i in range(n):
            if a in row_terms[i] and b in row_terms[i]:
                row_terms[i].discard(a)
                row_terms[i].discard(b)
                row_terms[i].add(new_wire)

        current_cost = total_cost()
        if current_cost < best_cost:
            best_cost = current_cost
            best_gates = [g[:] for g in gates]
            best_row_terms = [set(t) for t in row_terms]

    gates = best_gates
    row_terms = best_row_terms

    outputs = []
    for i in range(n):
        if len(row_terms[i]) == 0:
            outputs.append(n)
        elif len(row_terms[i]) == 1:
            outputs.append(next(iter(row_terms[i])))
        else:
            terms = sorted(row_terms[i])
            current = terms[0]
            for t in terms[1:]:
                new_wire = n + len(gates)
                gates.append([current, t])
                current = new_wire
            outputs.append(current)

    return {'gates': gates, 'outputs': outputs}
