import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    full = (1 << n) - 1

    # adjacency bitmask
    adj = [0] * n
    es = P['edges']
    for a, b in es:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute for each subset: is it a clique? independent set?
    N = 1 << n
    is_clique = [False] * N
    is_indep = [False] * N
    is_clique[0] = True
    is_indep[0] = True
    for mask in range(1, N):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    valid = [False] * N
    for m in range(N):
        if is_clique[m] or is_indep[m]:
            valid[m] = True

    # Quick check: is the full graph a clique or independent set?
    if valid[full]:
        return [list(range(n))]

    # DP: f[mask] = min number of valid parts to partition mask
    INF = 10**9
    f = [INF] * N
    f[0] = 0

    for mask in range(1, N):
        if time.time() - t0 > DEADLINE:
            break
        best = INF
        s = mask
        while s:
            if valid[s]:
                val = f[mask ^ s] + 1
                if val < best:
                    best = val
                    if best == 1:
                        break
            s = (s - 1) & mask
            if time.time() - t0 > DEADLINE:
                break
        f[mask] = best

    # If we finished the DP, we can reconstruct the solution
    if f[full] != INF:
        parts = []
        mask = full
        while mask:
            s = mask
            while s:
                if valid[s] and f[mask ^ s] + 1 == f[mask]:
                    parts.append(s)
                    mask ^= s
                    break
                s = (s - 1) & mask
            if not s and not parts:
                break
            if time.time() - t0 > DEADLINE:
                break
        
        if len(parts) == f[full]:
            final_parts = []
            for p in parts:
                vertices = []
                for i in range(n):
                    if (p >> i) & 1:
                        vertices.append(i)
                final_parts.append(vertices)
            return final_parts

    # Fallback: DFS search for minimal k
    coadj = [0] * n
    all_mask = (1 << n) - 1
    for u in range(n):
        coadj[u] = all_mask ^ adj[u] ^ (1 << u)

    CL = 0
    IND = 1
    BOTH = 2

    def can_add(v, part_mask, part_type):
        if part_type == CL:
            return (part_mask & ~adj[v]) == 0
        elif part_type == IND:
            return (part_mask & adj[v]) == 0
        elif part_type == BOTH:
            return True
        return False

    def get_new_type(v, part_mask, part_type):
        if part_type == BOTH:
            if (part_mask & adj[v]) != 0:
                return CL
            else:
                return IND
        return part_type

    def dfs_sol(unassigned, parts, k_rem, deadline):
        if unassigned == 0:
            return parts[:]
        if time.time() - t0 > deadline:
            return None

        v = (unassigned & -unassigned).bit_length() - 1
        new_unassigned = unassigned ^ (1 << v)

        # Try adding v to existing parts
        for i in range(len(parts)):
            p_mask, p_type = parts[i]
            if can_add(v, p_mask, p_type):
                new_type = get_new_type(v, p_mask, p_type)
                new_mask = p_mask | (1 << v)
                new_parts = parts[:]
                new_parts[i] = (new_mask, new_type)
                res = dfs_sol(new_unassigned, new_parts, k_rem, deadline)
                if res is not None:
                    return res

        # Try creating a new part
        if k_rem > 0:
            new_parts = parts[:] + [(1 << v, BOTH)]
            res = dfs_sol(new_unassigned, new_parts, k_rem - 1, deadline)
            if res is not None:
                return res

        return None

    for k in range(2, n + 1):
        if time.time() - t0 > DEADLINE:
            break
        res = dfs_sol(full, [], k, DEADLINE)
        if res is not None:
            final_parts = []
            for mask, _ in res:
                verts = []
                for v in range(n):
                    if mask & (1 << v):
                        verts.append(v)
                final_parts.append(verts)
            return final_parts

    # Final fallback: Greedy partition
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            part_clique = True
            part_indep = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, w = part[i], part[j]
                    if (adj[u] >> w) & 1:
                        part_indep = False
                    else:
                        part_clique = False
            if part_clique:
                ok = all((adj[v] >> u) & 1 for u in part)
                if ok:
                    part.append(v)
                    placed = True
                    break
            elif part_indep:
                ok = all(not ((adj[v] >> u) & 1) for u in part)
                if ok:
                    part.append(v)
                    placed = True
                    break
        if not placed:
            parts.append([v])
    return parts
