"""Cochromatic partition via exact chromatic/independence number computation."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    _clique_memo = {}
    _indep_memo = {}

    def is_clique(mask):
        if mask in _clique_memo:
            return _clique_memo[mask]
        if mask == 0:
            _clique_memo[mask] = True
            return True
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        result = is_clique(rest) and ((rest & ~adj[v]) == 0)
        _clique_memo[mask] = result
        return result

    def is_independent(mask):
        if mask in _indep_memo:
            return _indep_memo[mask]
        if mask == 0:
            _indep_memo[mask] = True
            return True
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        result = is_independent(rest) and ((rest & adj[v]) == 0)
        _indep_memo[mask] = result
        return result

    def is_valid_part(mask):
        if mask == 0:
            return False
        return is_clique(mask) or is_independent(mask)

    def find_chromatic_number():
        order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
        color = [-1] * n
        for v in order:
            used = set()
            for u in range(n):
                if color[u] >= 0 and (adj[v] & (1 << u)):
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        ub = max(color) + 1 if color else 0
        best = [ub]
        color = [-1] * n
        def dfs(idx, used_colors):
            if time.time() > deadline:
                return
            if used_colors >= best[0]:
                return
            if idx == n:
                best[0] = used_colors
                return
            v = order[idx]
            for c in range(used_colors):
                conflict = False
                for u in range(n):
                    if color[u] == c and (adj[v] & (1 << u)):
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    dfs(idx + 1, used_colors)
                    color[v] = -1
                    if time.time() > deadline:
                        return
            if used_colors < best[0]:
                color[v] = used_colors
                dfs(idx + 1, used_colors + 1)
                color[v] = -1
        dfs(0, 0)
        return best[0]

    chi = find_chromatic_number()

    def find_max_independent_set():
        dp = [0] * (1 << n)
        for mask in range(1, 1 << n):
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            dp[mask] = dp[rest]
            if (rest & adj[v]) == 0:
                dp[mask] = max(dp[mask], 1 + dp[rest & ~adj[v]])
        return dp[full_mask]

    alpha = find_max_independent_set()

    # Precompute all valid subsets (cliques or independent sets)
    valid_subs = set()
    for mask in range(1, 1 << n):
        if is_valid_part(mask):
            valid_subs.add(mask)
        if time.time() > deadline - 10:
            break

    best_parts = None

    # Try k=1
    if full_mask in valid_subs:
        best_parts = [full_mask]
    else:
        # Memoized DP: min number of valid parts to cover 'remaining'
        # Use canonical form: always include the lowest set bit in the first part
        _dp_memo = {}

        def min_parts(remaining):
            if time.time() > deadline - 5:
                return None
            if remaining == 0:
                return 0
            if remaining in _dp_memo:
                return _dp_memo[remaining]
            if remaining in valid_subs:
                _dp_memo[remaining] = 1
                return 1
            # Canonical: lowest bit must be in the first sub-part
            lsb = remaining & (-remaining)
            best = n + 1
            # Enumerate sub-masks of 'remaining' that contain 'lsb'
            # sub must be a valid part
            rest = remaining ^ lsb
            # Iterate over all sub-masks of rest, add lsb
            sub = rest
            while True:
                candidate = sub | lsb
                if candidate in valid_subs:
                    comp = remaining ^ candidate
                    sub_cost = min_parts(comp)
                    if sub_cost is not None and sub_cost + 1 < best:
                        best = sub_cost + 1
                        if best == 2:
                            break
                if sub == 0:
                    break
                sub = (sub - 1) & rest
            _dp_memo[remaining] = best
            return best

        k = min_parts(full_mask)
        if k is not None and k <= n:
            # Reconstruct the partition
            parts = []
            remaining = full_mask
            while remaining != 0:
                lsb = remaining & (-remaining)
                rest = remaining ^ lsb
                sub = rest
                found_part = None
                while True:
                    candidate = sub | lsb
                    if candidate in valid_subs:
                        comp = remaining ^ candidate
                        if comp == 0:
                            if min_parts(remaining) == 1:
                                found_part = candidate
                                break
                        else:
                            c = min_parts(comp)
                            if c is not None and c + 1 == min_parts(remaining):
                                found_part = candidate
                                break
                    if sub == 0:
                        break
                    sub = (sub - 1) & rest
                if found_part is None:
                    # Fallback: take a single vertex
                    found_part = lsb
                parts.append(found_part)
                remaining ^= found_part
            best_parts = parts

    # Fallback greedy
    if best_parts is None:
        es = set()
        for u, v in edges:
            es.add((u, v))
            es.add((v, u))
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                is_c = True
                is_i = True
                for u in part:
                    if (u, v) in es:
                        is_i = False
                    else:
                        is_c = False
                if is_c or is_i:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        best_parts = parts

    # Fallback greedy
    if best_parts is None:
        es = set()
        for u, v in edges:
            es.add((u, v))
            es.add((v, u))
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                is_c = True
                is_i = True
                for u in part:
                    if (u, v) in es:
                        is_i = False
                    else:
                        is_c = False
                if is_c or is_i:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        best_parts = parts

    result = []
    for part_mask in best_parts:
        verts = []
        for v in range(n):
            if part_mask & (1 << v):
                verts.append(v)
        result.append(verts)

    def validate(parts):
        seen = [False] * n
        for part in parts:
            if not part:
                return False
            for v in part:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
                if seen[v]:
                    return False
                seen[v] = True
            is_c = True
            is_i = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    if adj[part[i]] & (1 << part[j]):
                        is_i = False
                    else:
                        is_c = False
            if not (is_c or is_i):
                return False
        return all(seen)

    if not validate(result):
        result = [[v] for v in range(n)]

    return result
