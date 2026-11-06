import time
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    N = 128
    UNIVERSE = list(range(1, N + 1))
    n = N  # vertices 0..127 correspond to numbers 1..128

    # Precompute gcd matrix (1-indexed)
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Build list of bad triples: (a,b,c) with a<b<c, gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            if gab == 1:
                continue  # gcd 1 means coprime pair; check if all three are 1
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed

    # For each vertex, precompute the set of vertices it's in a bad triple with
    # conflict_pairs[u] = set of v > u such that (u,v,w) is a bad triple for some w
    # Actually, for B&B we need: given a set S, is S independent (no bad triple fully in S)?
    # Precompute for each vertex v, the list of bad triples containing v
    triples_by_vertex = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(bad_triples):
        triples_by_vertex[a].append(idx)
        triples_by_vertex[b].append(idx)
        triples_by_vertex[c].append(idx)

    # Greedy lower bound: process vertices in order of fewest conflicts first
    # Count for each vertex how many bad triples it participates in
    vertex_degree = [len(triples_by_vertex[v]) for v in range(n)]
    
    # Greedy: pick vertex with minimum remaining conflicts
    def greedy_lower_bound():
        selected = set()
        remaining = list(range(n))
        # Sort by degree ascending (fewest constraints first)
        remaining.sort(key=lambda v: vertex_degree[v])
        
        for v in remaining:
            # Check if adding v creates a bad triple with two already-selected vertices
            conflict = False
            for t_idx in triples_by_vertex[v]:
                a, b, c = bad_triples[t_idx]
                # Check if the other two vertices in this triple are both in selected
                others = [x for x in (a, b, c) if x != v]
                if others[0] in selected and others[1] in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
        return selected

    # Also try greedy in reverse order (high degree first)
    def greedy_lower_bound_rev():
        selected = set()
        remaining = list(range(n))
        remaining.sort(key=lambda v: -vertex_degree[v])
        
        for v in remaining:
            conflict = False
            for t_idx in triples_by_vertex[v]:
                a, b, c = bad_triples[t_idx]
                others = [x for x in (a, b, c) if x != v]
                if others[0] in selected and others[1] in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
        return selected

    # Try multiple greedy orderings
    best_greedy = set()
    best_greedy_size = 0
    
    for ordering in ["min_degree", "max_degree", "natural", "reverse"]:
        if time.time() > DEADLINE - 5:
            break
        if ordering == "min_degree":
            cand = greedy_lower_bound()
        elif ordering == "max_degree":
            cand = greedy_lower_bound_rev()
        elif ordering == "natural":
            selected = set()
            for v in range(n):
                conflict = False
                for t_idx in triples_by_vertex[v]:
                    a, b, c = bad_triples[t_idx]
                    others = [x for x in (a, b, c) if x != v]
                    if others[0] in selected and others[1] in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(v)
            cand = selected
        else:  # reverse
            selected = set()
            for v in range(n - 1, -1, -1):
                conflict = False
                for t_idx in triples_by_vertex[v]:
                    a, b, c = bad_triples[t_idx]
                    others = [x for x in (a, b, c) if x != v]
                    if others[0] in selected and others[1] in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(v)
            cand = selected
        
        if len(cand) > best_greedy_size:
            best_greedy_size = len(cand)
            best_greedy = cand

    # Branch and bound for maximum independent set in 3-uniform hypergraph
    # State: (selected_set, candidate_set, best_size)
    # We process vertices in a fixed order; at each step, either include or exclude
    
    # Use a recursive B&B with bitsets for speed
    # candidates: bitmask of remaining vertices to consider
    # selected: bitmask of vertices already in the solution
    # We want to maximize |selected|
    
    # For fast conflict check: when considering adding vertex v to selected,
    # check if any bad triple has v and two other vertices both in selected
    
    # Precompute for each vertex v, a bitmask of "forbidden pairs" - but that's complex.
    # Instead: for each v, precompute list of (u, w) pairs such that (v, u, w) is a bad triple
    # Then when adding v, check if any (u,w) both in selected
    
    forbidden_pairs = [[] for _ in range(n)]
    for a, b, c in bad_triples:
        # For vertex a: forbidden pair is (b, c)
        forbidden_pairs[a].append((b, c))
        # For vertex b: forbidden pair is (a, c)
        forbidden_pairs[b].append((a, c))
        # For vertex c: forbidden pair is (a, b)
        forbidden_pairs[c].append((a, b))

    def can_add(v, selected_mask):
        """Check if vertex v can be added to selected (represented as bitmask)."""
        for u, w in forbidden_pairs[v]:
            if (selected_mask >> u) & 1 and (selected_mask >> w) & 1:
                return False
        return True

    # B&B: process vertices in order 0, 1, ..., n-1
    # At position pos, we decide to include or exclude vertex pos
    # Upper bound: current_size + (n - pos) [optimistic: all remaining can be added]
    # Better bound: current_size + number of remaining vertices that can still be added
    
    best_solution_mask = 0
    best_solution_size = best_greedy_size
    
    # Convert best_greedy to mask
    for v in best_greedy:
        best_solution_mask |= (1 << v)

    def branch_and_bound(pos, selected_mask, current_size):
        nonlocal best_solution_mask, best_solution_size
        
        if time.time() > DEADLINE - 2:
            return
        
        if pos == n:
            if current_size > best_solution_size:
                best_solution_size = current_size
                best_solution_mask = selected_mask
            return
        
        # Upper bound: current_size + (n - pos)
        if current_size + (n - pos) <= best_solution_size:
            return
        
        # Option 1: exclude vertex pos
        branch_and_bound(pos + 1, selected_mask, current_size)
        
        # Option 2: include vertex pos (if valid)
        if can_add(pos, selected_mask):
            branch_and_bound(pos + 1, selected_mask | (1 << pos), current_size + 1)

    # Run B&B
    branch_and_bound(0, 0, 0)

    # Convert best_solution_mask to list
    result = [v + 1 for v in range(n) if (best_solution_mask >> v) & 1]
    
    if not result:
        result = [1]
    
    return result

# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
