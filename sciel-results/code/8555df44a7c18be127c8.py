"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to stay under 90s wall-clock
    
    n = 128
    # Precompute LCM matrix
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all "bad" triples: three distinct elements a < b < c such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structure: for each element, which other elements form a bad triple with it
    # Actually, we need a hypergraph: each bad triple is a hyperedge
    # We want the maximum independent set in this 3-uniform hypergraph
    
    # For branch and bound, we need:
    # - For each element, the list of hyperedges it belongs to
    # - A way to quickly check if adding an element creates a conflict
    
    # Let's use a different approach: branch and bound with pruning
    # State: current selected set, candidate elements that can still be added
    # A candidate x can be added if no two elements in the current set form a bad triple with x
    
    # Precompute for each pair (a, b) whether they are in any bad triple together
    # and if so, which third elements complete the triple
    pair_to_thirds = {}
    for a, b, c in bad_triples:
        key = (min(a,b), max(a,b))
        if key not in pair_to_thirds:
            pair_to_thirds[key] = set()
        pair_to_thirds[key].add(c)
    
    # Also for each element, the set of elements that form a bad triple with it (in any pair)
    # Actually, let's precompute: for each element x, and each pair (a,b) already selected,
    # check if (a,b,x) is a bad triple. But this is expensive.
    
    # Better: for each element x, precompute which pairs (a,b) with a<b<x or similar 
    # would form a bad triple. Actually, let's just precompute for each x the set of
    # pairs (a,b) such that {a,b,x} is a bad triple.
    
    # For each x, store the set of (a,b) pairs that with x form a bad triple
    x_conflict_pairs = {x: set() for x in range(1, n + 1)}
    for a, b, c in bad_triples:
        x_conflict_pairs[a].add((b, c))
        x_conflict_pairs[b].add((a, c))
        x_conflict_pairs[c].add((a, b))
    
    # Branch and bound
    best_set = []
    
    def can_add(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        # For each pair in selected, check if (pair, x) is a bad triple
        # We can use x_conflict_pairs: for x, the set of pairs that conflict with x
        # A conflict exists if any pair in selected is in x_conflict_pairs[x]
        conflict_pairs = x_conflict_pairs[x]
        for a, b in selected:
            if (a, b) in conflict_pairs or (b, a) in conflict_pairs:
                return False
        return True
    
    # This is too slow. Let's optimize.
    
    # Alternative: precompute for each x, a bitmask or set representation
    # Since n=128, we can use bitmasks.
    
    # Let's use a different strategy: greedy with local search, or a better B&B
    
    # Actually, let's first try a greedy approach to get a good initial solution,
    # then use local search to improve.
    
    # Greedy: add elements in order of least conflict
    # For each element, compute how many bad triples it's in
    triple_count = [0] * (n + 1)
    for a, b, c in bad_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    # Sort by triple count (ascending) - elements in fewer triples are "safer"
    order = sorted(range(1, n + 1), key=lambda x: (triple_count[x], x))
    
    selected = []
    for x in order:
        if can_add(x, selected):
            selected.append(x)
    
    best_set = selected[:]
    
    # Local search: try to swap elements or add/remove
    # Try to add any element not in selected
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        # Try adding each element not in current_set
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, list(current_set)):
                best_set.append(x)
                current_set.add(x)
                improved = True
                break
        # Try removing one element and adding two
        # This is complex; let's try a simpler local search
        # Try swapping: remove one, add one
        if time.time() < deadline:
            for i in range(len(best_set)):
                removed = best_set[i]
                temp_set = best_set[:i] + best_set[i+1:]
                temp_set_set = set(temp_set)
                for x in range(1, n + 1):
                    if x in temp_set_set:
                        continue
                    if can_add(x, temp_set):
                        # Check if this improves (it doesn't change size, so only if we can add more later)
                        # For now, just accept if it allows more additions
                        new_set = temp_set + [x]
                        # Try to add more
                        can_add_more = False
                        for y in range(1, n + 1):
                            if y in set(new_set):
                                continue
                            if can_add(y, new_set):
                                can_add_more = True
                                break
                        if can_add_more:
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break
    
    # Verify the solution
    def is_valid(s):
        s_set = set(s)
        for a, b, c in combinations(s, 3):
            if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                return False
        return True
    
    if not is_valid(best_set):
        # Fallback to greedy
        best_set = []
        for x in range(1, n + 1):
            if can_add(x, best_set):
                best_set.append(x)
    
    return best_set
# EVOLVE-BLOCK-END
