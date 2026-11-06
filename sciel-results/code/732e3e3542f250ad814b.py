"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_pairs)}

    # ---- Precompute Q3 4x4 patterns ----
    # A Q3 is a 3-regular bipartite graph on 4+4 vertices.
    # There are 6 such graphs up to isomorphism (the 6 ways to pair 4+4 with degree 3).
    # We generate all 576 permutations of row/col indices for the 12 edges.
    # The base Q3 edges (0-3 vs 4-7) are derived from bit flips:
    # V0: 000(0), 011(3), 101(5), 110(6) -> indices 0, 1, 2, 3
    # V1: 001(1), 010(2), 100(4), 111(7) -> indices 0, 1, 2, 3
    base_q3_edges = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,3), (2,0), (2,2), (2,3), (3,1), (3,2), (3,3)]
    
    q3_patterns = set()
    for perm_r in itertools.permutations(range(4)):
        for perm_c in itertools.permutations(range(4)):
            mask = 0
            for r, c in base_q3_edges:
                pr = perm_r[r]
                pc = perm_c[c]
                mask |= (1 << (pr * 4 + pc))
            q3_patterns.add(mask)
    
    # Precompute: for each 16-bit mask, does it contain a Q3 pattern?
    mask_is_q3_superset = [False] * (1 << 16)
    for m in range(1 << 16):
        for p in q3_patterns:
            if (m & p) == p:
                mask_is_q3_superset[m] = True
                break

    # ---- Precompute relevant (A,B) pairs for each edge ----
    # For each edge (u,v), find all (A,B) such that u in A, v in B or u in B, v in A.
    # We store the list of local edge indices for each (A,B) pair.
    all_verts = list(range(n))
    edge_to_ab_pairs = {e: [] for e in all_pairs}
    
    for A in itertools.combinations(all_verts, 4):
        Aset = set(A)
        for B in itertools.combinations(all_verts, 4):
            if Aset & set(B):
                continue
            # Precompute local edge indices
            local_edges = []
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    e = (a, b) if a < b else (b, a)
                    local_edges.append((edge_index[e], i*4+j))
            
            for a in A:
                for b in B:
                    e = (a, b) if a < b else (b, a)
                    edge_to_ab_pairs[e].append(local_edges)

    # ---- Build graph as bitsets ----
    adj = [0] * n
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    def edge_count():
        return sum(bin(adj[i]).count("1") for i in range(n)) // 2

    # ---- Seed: K7 + K7 + K2 (57 edges, Q3-free) ----
    for i in range(7):
        for j in range(i+1, 7):
            add_edge(i, j)
    for i in range(7, 14):
        for j in range(i+1, 14):
            add_edge(i, j)
    add_edge(14, 15)

    best_edges = edge_count()
    best_adj = adj[:]

    def time_left():
        return deadline - time.time()

    # ---- Fast Q3 check for adding (u,v) ----
    def creates_q3_fast(u, v):
        e = (u, v) if u < v else (v, u)
        for local_edges in edge_to_ab_pairs[e]:
            mask = 0
            for ge_idx, bit_pos in local_edges:
                a, b = all_pairs[ge_idx]
                if adj[a] & (1 << b):
                    mask |= (1 << bit_pos)
                if (a == u and b == v) or (a == v and b == u):
                    mask |= (1 << bit_pos)
            if mask_is_q3_superset[mask]:
                return True
        return False

    # ---- Count Q3s in current graph (for SA penalty) ----
    def count_q3s():
        cnt = 0
        for A in itertools.combinations(all_verts, 4):
            Aset = set(A)
            for B in itertools.combinations(all_verts, 4):
                if Aset & set(B):
                    continue
                mask = 0
                for i, a in enumerate(A):
                    for j, b in enumerate(B):
                        if adj[a] & (1 << b):
                            mask |= (1 << (i*4+j))
                if mask_is_q3_superset[mask]:
                    cnt += 1
        return cnt

    # ---- Greedy: try to add every missing edge ----
    missing = [e for e in all_pairs if not has_edge(*e)]
    random.seed(42)
    random.shuffle(missing)

    for u, v in missing:
        if time_left() < 20:
            break
        if not creates_q3_fast(u, v):
            add_edge(u, v)
            if edge_count() > best_edges:
                best_edges = edge_count()
                best_adj = adj[:]

    # ---- Simulated Annealing with Q3 penalty ----
    # Structural change: allow temporary Q3 violations to escape local optima
    if time_left() > 30:
        adj[:] = best_adj[:]
        current_edges = edge_count()
        current_q3 = count_q3s()  # should be 0
        
        T = 10.0
        cooling = 0.999
        min_T = 0.05
        
        while time_left() > 10 and T > min_T:
            # Pick a random edge to toggle
            u, v = random.choice(all_pairs)
            
            if has_edge(u, v):
                # Try removing
                remove_edge(u, v)
                new_edges = current_edges - 1
                new_q3 = count_q3s()
                # Score: maximize edges, penalize Q3s
                old_score = current_edges - 1000 * current_q3
                new_score = new_edges - 1000 * new_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    current_edges = new_edges
                    current_q3 = new_q3
                    if current_q3 == 0 and current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
            else:
                # Try adding
                add_edge(u, v)
                new_edges = current_edges + 1
                new_q3 = count_q3s()
                old_score = current_edges - 1000 * current_q3
                new_score = new_edges - 1000 * new_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    current_edges = new_edges
                    current_q3 = new_q3
                    if current_q3 == 0 and current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    remove_edge(u, v)
            
            T *= cooling

        # After SA, ensure we have a valid Q3-free graph
        if count_q3s() > 0:
            # Greedily remove edges that are in Q3s
            for _ in range(200):
                if count_q3s() == 0:
                    break
                if time_left() < 5:
                    break
                found = False
                for u, v in all_pairs:
                    if not has_edge(u, v):
                        continue
                    remove_edge(u, v)
                    if count_q3s() < count_q3s():
                        found = True
                        break
                    add_edge(u, v)
                if not found:
                    break

    # ---- Restore best state and final validation ----
    adj[:] = best_adj[:]
    
    # Final validity check
    if time_left() > 5:
        found = False
        for A in itertools.combinations(all_verts, 4):
            Aset = set(A)
            for B in itertools.combinations(all_verts, 4):
                if Aset & set(B):
                    continue
                mask = 0
                for i, a in enumerate(A):
                    for j, b in enumerate(B):
                        if adj[a] & (1 << b):
                            mask |= (1 << (i*4+j))
                if mask_is_q3_superset[mask]:
                    found = True
                    break
            if found:
                break
        if found:
            # Fallback to seed
            adj = [0]*n
            for i in range(7):
                for j in range(i+1, 7):
                    add_edge(i, j)
            for i in range(7, 14):
                for j in range(i+1, 14):
                    add_edge(i, j)
            add_edge(14, 15)

    return [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
# EVOLVE-BLOCK-END
