import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    # Perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Diagonal constraint: 2*v must not be a perfect square
    forbidden = set()
    for v in range(1, N + 1):
        if 2 * v in squares:
            forbidden.add(v)

    valid_vertices = [v for v in range(1, N + 1) if v not in forbidden]
    M = len(valid_vertices)
    vertex_map = valid_vertices  # index -> value

    # Build adjacency (conflict) as bitsets for speed
    adj_mask = [0] * M
    for i in range(M):
        vi = vertex_map[i]
        for j in range(i + 1, M):
            vj = vertex_map[j]
            if vi + vj in squares:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def can_add(selected_mask, v):
        return (selected_mask & adj_mask[v]) == 0

    # ---------- LP relaxation for seeding ----------
    lp_solution = None
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix

        edges = []
        for i in range(M):
            m = adj_mask[i]
            j = i + 1
            while m:
                if m & 1:
                    edges.append((i, j))
                m >>= 1
                j += 1

        E = len(edges)
        c = [-1.0] * M
        A_ub = []
        for (i, j) in edges:
            row = [0.0] * M
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
        A_ub = csr_matrix(A_ub)
        b_ub = [1.0] * E
        bounds = [(0.0, 1.0)] * M

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            lp_solution = res.x
    except Exception:
        lp_solution = None

    # ---------- Local search with 1-opt and 2-opt swaps ----------
    def local_search(initial_mask, budget_sec=2.0):
        selected = initial_mask
        improved = True
        end_time = start_time + budget_sec
        while improved and time.time() < end_time and time.time() < DEADLINE:
            improved = False
            # 1-opt: add any non-conflicting vertex
            for v in range(M):
                if not (selected >> v) & 1:
                    if can_add(selected, v):
                        selected |= (1 << v)
                        improved = True
            # 2-opt: remove 1, try to add 2
            sel_list = [v for v in range(M) if (selected >> v) & 1]
            for v in sel_list:
                without = selected & ~(1 << v)
                # Try adding two vertices that both fit
                added = 0
                for u in range(M):
                    if added >= 2:
                        break
                    if (without >> u) & 1:
                        continue
                    if can_add(without, u):
                        without |= (1 << u)
                        added += 1
                if bin(without).count('1') > bin(selected).count('1'):
                    selected = without
                    improved = True
            # 2-opt: remove 2, try to add 3
            for a in range(len(sel_list)):
                for b in range(a + 1, len(sel_list)):
                    va, vb = sel_list[a], sel_list[b]
                    without = selected & ~(1 << va) & ~(1 << vb)
                    added = 0
                    for u in range(M):
                        if added >= 3:
                            break
                        if (without >> u) & 1:
                            continue
                        if can_add(without, u):
                            without |= (1 << u)
                            added += 1
                    if bin(without).count('1') > bin(selected).count('1'):
                        selected = without
                        improved = True
                        break
                if improved:
                    break
        return selected

    def mask_to_set(mask):
        return [vertex_map[i] for i in range(M) if (mask >> i) & 1]

    # ---------- Branch and Bound for Maximum Independent Set ----------
    # Upper bound: |C| - |max_matching(G[C])|
    # We compute max matching via a greedy + augmenting path approach on the induced subgraph.
    
    def max_matching_bound(candidate_mask):
        """Return upper bound on MIS size: |C| - |max matching in G[C]|"""
        c_list = [i for i in range(M) if (candidate_mask >> i) & 1]
        c_set = set(c_list)
        # Build adjacency within candidate
        # Greedy matching first
        match = {}  # vertex -> matched partner
        for i in c_list:
            if i in match:
                continue
            # Find an unmatched neighbor in candidate
            m = adj_mask[i] & candidate_mask
            m &= ~(1 << i)
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                if j not in match:
                    match[i] = j
                    match[j] = i
                    break
                m ^= lsb
        # Augmenting paths (BFS) to improve matching
        changed = True
        while changed:
            changed = False
            # Find augmenting path from each unmatched vertex
            for start in c_list:
                if start in match:
                    continue
                # BFS to find augmenting path
                parent = {start: None}
                queue = [start]
                found = False
                while queue and not found:
                    u = queue.pop(0)
                    m = adj_mask[u] & candidate_mask
                    m &= ~(1 << u)
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        if v not in parent:
                            parent[v] = u
                            if v not in match:
                                # Found augmenting path
                                # Trace back and flip
                                cur = v
                                while cur != start:
                                    prev = parent[cur]
                                    if cur in match:
                                        del match[cur]
                                    if prev in match:
                                        del match[prev]
                                    match[cur] = prev
                                    match[prev] = cur
                                    cur = prev
                                found = True
                                changed = True
                                break
                            else:
                                # v is matched, go to its partner
                                partner = match[v]
                                if partner not in parent:
                                    parent[partner] = v
                                    queue.append(partner)
                        m ^= lsb
        return len(c_list) - len(match)

    # Greedy coloring bound on the complement (clique bound)
    # Actually, for MIS, we need clique cover of G[C], which equals chromatic number of complement.
    # Simpler: use the matching bound which is tighter in practice.

    best_mask = 0
    best_size = 0

    def consider(mask):
        nonlocal best_mask, best_size
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # ---------- Seed solutions ----------
    # Seed 1: LP-guided greedy
    if lp_solution is not None:
        order = sorted(range(M), key=lambda i: -lp_solution[i])
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        consider(mask)
        mask = local_search(mask, budget_sec=5.0)
        consider(mask)

    # Seed 2: degree-based greedy (pick low-degree first)
    deg = [bin(adj_mask[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda i: deg[i])
    mask = 0
    for v in order:
        if can_add(mask, v):
            mask |= (1 << v)
    consider(mask)
    mask = local_search(mask, budget_sec=5.0)
    consider(mask)

    # Seed 3: random restarts
    for r in range(300):
        if elapsed() > DEADLINE - 15:
            break
        random.seed(r * 7919 + 13)
        order = list(range(M))
        random.shuffle(order)
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        mask = local_search(mask, budget_sec=0.3)
        consider(mask)

    # ---------- Branch and Bound ----------
    # Start from the best solution found so far.
    # We'll do a depth-first branch-and-bound starting with all vertices as candidates.
    # Branch on a vertex: include it (remove its neighbors) or exclude it.
    
    def branch_and_bound(candidate_mask, current_mask, current_size, depth=0):
        nonlocal best_mask, best_size
        
        if time.time() > DEADLINE:
            return
        
        # Compute upper bound
        ub = max_matching_bound(candidate_mask)
        if current_size + ub <= best_size:
            return
        
        if candidate_mask == 0:
            return
        
        # If we can add all candidates (no edges among them), take them all
        # Quick check: if no edges in candidate subgraph
        has_edge = False
        c_list = [i for i in range(M) if (candidate_mask >> i) & 1]
        for i in c_list:
            if adj_mask[i] & candidate_mask & ~(1 << i):
                has_edge = True
                break
        if not has_edge:
            new_mask = current_mask | candidate_mask
            new_size = current_size + len(c_list)
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
            return
        
        # Choose branching vertex: highest degree in candidate subgraph
        best_v = -1
        best_deg = -1
        for i in c_list:
            d = bin(adj_mask[i] & candidate_mask & ~(1 << i)).count('1')
            if d > best_deg:
                best_deg = d
                best_v = i
        
        if best_v < 0:
            return
        
        # Branch 1: Include best_v
        # Remove best_v and its neighbors from candidate
        new_candidate = candidate_mask & ~(1 << best_v) & ~adj_mask[best_v]
        new_mask = current_mask | (1 << best_v)
        branch_and_bound(new_candidate, new_mask, current_size + 1, depth + 1)
        
        if time.time() > DEADLINE:
            return
        
        # Branch 2: Exclude best_v
        new_candidate = candidate_mask & ~(1 << best_v)
        branch_and_bound(new_candidate, current_mask, current_size, depth + 1)
    
    # Run branch and bound with a time limit
    # Start with all valid vertices as candidates
    all_mask = (1 << M) - 1
    if elapsed() < DEADLINE - 5:
        branch_and_bound(all_mask, 0, 0)

    # ---------- Final polish ----------
    final_mask = local_search(best_mask, budget_sec=max(1.0, DEADLINE - elapsed() - 1))
    consider(final_mask)

    result = mask_to_set(best_mask)
    result.sort()

    # Safety: if somehow empty, return a single valid vertex
    if not result:
        for v in range(1, N + 1):
            if 2 * v not in squares:
                return [v]
        return [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    valid = True
    for a in res:
        if 2 * a in squares:
            valid = False
            print(f"Invalid diagonal: {a}")
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if res[i] + res[j] in squares:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    print("Valid" if valid else "Invalid")
