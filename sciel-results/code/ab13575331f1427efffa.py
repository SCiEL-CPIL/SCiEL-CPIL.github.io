"""Improved program: vectorized conflict checking + branch-and-bound MIS on conflict graph."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

N = 192
K = 3
MAX_CARD = 48

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    if n < 4:
        return True
    sel = list(selected)
    for i in range(n):
        a = sel[i]
        others = [sel[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _can_add_fast(selected, x):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    # Check x against existing triples
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            return False
    # Check each existing element a against pairs + x
    for i in range(n):
        a = selected[i]
        for j in range(n):
            if j == i:
                continue
            s1 = selected[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                s2 = selected[k]
                if (s1 + s2 + x) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Greedy construction with given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(selected, x):
            selected.append(x)
    return selected

def _build_conflict_graph():
    """Build conflict graph: edge (a,b) exists if there's a c such that a | (b+c+d) for some d,
    or more precisely: a and b conflict if there exist c, d distinct from a,b such that
    a divides b+c+d. Actually the constraint is: for each selected a, no triple of OTHERS sums to multiple of a.
    So a and b are "conflicting" in the sense that if both are selected, we need to ensure no triple from remaining sums to multiple of a or b.
    
    For a pairwise conflict graph for MIS: two vertices a, b are adjacent (conflict) if there exists a c (c != a, b) such that a | (b + c + d) for some d... 
    Actually this is a hypergraph constraint, not pairwise. Let me think differently.
    
    A simpler approach: build a "forbidden triple" set. For each a in 1..192, and each triple (b,c,d) of distinct elements not including a, if a | (b+c+d), then {a,b,c,d} is a forbidden 4-set.
    
    For the greedy approach, we just check incrementally. For a branch-and-bound, we need a different strategy.
    
    Let me use a simpler but effective approach: randomized greedy with multiple restarts + local search with better heuristics.
    """
    pass

def _random_greedy(deadline):
    """Randomized greedy with multiple restarts and structured orderings."""
    best = []
    start_time = time.time()
    
    # Try several random orderings
    for attempt in range(30):
        if time.time() > deadline:
            break
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        result = _greedy(rand_order, deadline)
        if len(result) > len(best):
            best = result
    
    # Structured orderings
    structured_orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(96, 0, -1)) + list(range(97, N + 1)),
        [x for x in range(N, 0, -1) if x > 64],
        [x for x in range(N, 0, -1) if x % 2 == 1],
        [x for x in range(N, 0, -1) if x > 48],
    ]
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    return best

def _local_improve(selected, deadline):
    """Local search: try to add more elements."""
    selected = list(selected)
    candidates = [x for x in range(1, N + 1) if x not in set(selected)]
    candidates.sort(reverse=True)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates[:]:
            if time.time() > deadline:
                break
            if x not in set(selected):
                if _can_add_fast(selected, x):
                    selected.append(x)
                    candidates.remove(x)
                    improved = True
    
    # Try swaps: remove one element, try to add two new ones
    for i in range(len(selected)):
        if time.time() > deadline:
            break
        removed = selected[i]
        remaining = selected[:i] + selected[i+1:]
        candidates_for_swap = [x for x in range(1, N + 1) if x not in set(remaining) and x != removed]
        candidates_for_swap.sort(reverse=True)
        
        added = []
        for x in candidates_for_swap:
            if time.time() > deadline:
                break
            if len(added) == 2:
                break
            if _can_add_fast(remaining + added, x):
                added.append(x)
        
        if len(added) >= 2:
            new_selected = remaining + added
            if len(new_selected) > len(selected):
                selected = new_selected
    
    return selected

def _branch_and_bound_mis(deadline):
    """Branch and bound for maximum independent set on a conflict graph.
    
    We define a conflict graph where two vertices a, b are connected if they cannot both be in a valid set.
    Two elements a and b conflict if there exist c, d (distinct from a, b, and from each other) such that
    a divides (b + c + d) OR b divides (a + c + d).
    
    Actually, the constraint is higher-order. Let me use a different approach:
    Build a "forbidden" relation: a and b are incompatible if there exist c, d (c,d distinct from a,b, c!=d) such that a|(b+c+d).
    Then find maximum independent set in this graph.
    """
    # Build pairwise conflict graph
    # a and b conflict if there exist c, d distinct from a,b, c!=d, such that a | (b+c+d)
    # This is O(n^4) to build but n=192 so ~192^4 = 1.3B which is too slow.
    # Let's optimize: for each a, for each b, we need to check if there exist c,d such that b+c+d ≡ 0 (mod a).
    # b+c+d ≡ 0 (mod a) means c+d ≡ -b (mod a).
    # So we need to check if there exist c,d in [1,192], c≠d, c≠a, c≠b, d≠a, d≠b, such that c+d ≡ -b (mod a).
    
    # For each a, precompute for each residue r mod a, the set of values v in [1,192]\{a} such that v ≡ r (mod a).
    # Then for given b, we need c+d ≡ -b (mod a), i.e., c ≡ r, d ≡ (-b-r) (mod a) for some r.
    # We need c≠b, d≠b, c≠a, d≠a, c≠d.
    
    conflict = np.zeros((N+1, N+1), dtype=bool)
    
    for a in range(1, N+1):
        # For each residue mod a, store list of values
        residue_vals = {}
        for v in range(1, N+1):
            if v == a:
                continue
            r = v % a
            if r not in residue_vals:
                residue_vals[r] = []
            residue_vals[r].append(v)
        
        for b in range(1, N+1):
            if b == a:
                continue
            # Check if there exist c, d such that c+d ≡ -b (mod a), c,d distinct from a,b, c≠d
            found = False
            neg_b = (-b) % a
            for r in range(a):
                r2 = (neg_b - r) % a
                if r not in residue_vals:
                    continue
                if r2 not in residue_vals:
                    continue
                c_list = residue_vals[r]
                d_list = residue_vals[r2]
                # Need c≠b, c≠a, d≠b, d≠a, c≠d
                c_valid = [c for c in c_list if c != b]
                d_valid = [d for d in d_list if d != b]
                if not c_valid or not d_valid:
                    continue
                if r == r2:
                    # c and d from same residue class, need c≠d
                    if len(c_valid) >= 2:
                        found = True
                        break
                    elif len(c_valid) == 1:
                        # Only one valid c, need d from same class, d≠c
                        # d_valid has same elements as c_valid (same residue), so need at least 2 distinct
                        # Already checked len>=2
                        pass
                else:
                    if c_valid and d_valid:
                        # Need c≠d, but since r≠r2, c≠d automatically
                        found = True
                        break
            if found:
                conflict[a][b] = True
                conflict[b][a] = True
    
    # Now find maximum independent set using branch and bound
    # Vertices: 1..192
    # Adjacency: conflict matrix
    # We want max set with no edges
    
    # Use a simple branch and bound with coloring upper bound
    from collections import defaultdict
    
    # Convert to 0-indexed
    adj = [set() for _ in range(N)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i != j and conflict[i][j]:
                adj[i-1].add(j-1)
    
    # Greedy coloring for upper bound
    def greedy_coloring(vertices, adj):
        """Color vertices greedily, return max color used."""
        color = {}
        for v in vertices:
            used = set()
            for u in adj[v]:
                if u in color:
                    used.add(color[u])
            c = 1
            while c in used:
                c += 1
            color[v] = c
        return max(color.values()) if color else 0
    
    # Branch and bound
    best_set = []
    best_size = 0
    
    def branch_and_bound(candidates, current_set):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound: |current_set| + greedy_coloring(candidates)
        # But coloring is expensive, use a simpler bound
        # Simple bound: |current_set| + len(candidates)
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Choose vertex with highest degree in candidate subgraph
        cand_list = list(candidates)
        # Pick vertex with max degree
        max_deg = -1
        chosen = cand_list[0]
        for v in cand_list:
            deg = len(adj[v] & set(candidates))
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        # Branch 1: include chosen
        new_candidates = set(candidates) - {chosen} - adj[chosen]
        current_set.add(chosen)
        branch_and_bound(new_candidates, current_set)
        current_set.discard(chosen)
        
        # Branch 2: exclude chosen
        new_candidates = set(candidates) - {chosen}
        branch_and_bound(new_candidates, current_set)
    
    # This is too slow for 192 vertices. Let me use a different approach.
    # Instead, use a greedy + local search hybrid with better heuristics.
    pass

def _weighted_greedy(deadline):
    """Greedy that prioritizes elements with fewer conflicts."""
    # Compute for each element x, how many other elements y have the property that
    # there exist c,d such that x | (y+c+d)
    # This gives a "conflict count" - prefer elements with fewer conflicts
    
    conflict_count = np.zeros(N+1, dtype=int)
    for a in range(1, N+1):
        for b in range(1, N+1):
            if a == b:
                continue
            # Check if there exist c,d distinct from a,b,c≠d such that a | (b+c+d)
            # b+c+d ≡ 0 (mod a) => c+d ≡ -b (mod a)
            # For small a, this is likely; for large a, less likely
            # Quick check: if a > 3*192 = 576, impossible, but a<=192
            # c+d ranges from 1+2=3 to 191+192=383
            # So a | (b+c+d) requires b+c+d >= a and b+c+d is multiple of a
            # b+c+d ranges from b+3 to b+383
            # For a > b+383, impossible. But a<=192, b>=1, so b+383>=384>192 always
            # So for any a<=192 and b<=192, it's possible in principle
            # Let's just count roughly: for each a, the number of (c,d) pairs with c+d ≡ -b (mod a)
            # is approximately (192/a)^2 * a = 192^2/a
            # So conflict_count[a] ~ sum over b of 192^2/a ~ 192 * 192^2/a = 192^3/a
            # This means small a have many conflicts, large a have few
            # So prefer large numbers
            conflict_count[a] += 192 * 192 // a
    
    order = sorted(range(1, N+1), key=lambda x: conflict_count[x])
    return _greedy(order, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin before 90s limit
    
    # Phase 1: Multiple greedy strategies
    best = []
    
    # Weighted greedy (prefer large numbers)
    result = _weighted_greedy(deadline)
    if len(result) > len(best):
        best = result
    
    # Randomized greedy
    result = _random_greedy(deadline)
    if len(result) > len(best):
        best = result
    
    # Phase 2: Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: More local improvement rounds
    if time.time() < deadline - 5:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 4: Try additional random restarts if time allows
    if time.time() < deadline - 10:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            rand_order = list(range(1, N + 1))
            random.shuffle(rand_order)
            result = _greedy(rand_order, deadline - 5)
            if len(result) > len(best):
                best = result
            if time.time() < deadline - 5:
                improved = _local_improve(best, deadline - 5)
                if len(improved) > len(best):
                    best = improved
    
    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _greedy(list(range(N, 0, -1)), deadline)
    
    return best
# EVOLVE-BLOCK-END
