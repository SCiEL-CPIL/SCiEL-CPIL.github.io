"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations
from functools import lru_cache

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM matrix
    lcm_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_mat[i][j] = _lcm(i, j)
            lcm_mat[j][i] = lcm_mat[i][j]
    
    def is_bad_triple(a, b, c):
        return lcm_mat[a][b] == lcm_mat[a][c] == lcm_mat[b][c]
    
    # Build hypergraph: for each element x, find all pairs (a,b) in [1..n]\{x}
    # such that (a,b,x) is a bad triple
    # Then x conflicts with pair (a,b)
    
    # For efficient checking: precompute for each x, the set of "conflict pairs"
    # conflict_pairs[x] = list of (a,b) where a<b, a!=x, b!=x, and (a,b,x) is bad
    # But this could be large. Instead, for membership testing:
    # For a candidate set S and new element x, x can be added iff
    # for all pairs (a,b) in S with a<b, (a,b,x) is NOT a bad triple.
    
    # Precompute: bad_triple_set = set of frozenset({a,b,c}) for all bad triples
    # This might be too large. Let's check the count first.
    
    # Alternative: for each x, precompute a bitset of "conflicting" elements
    # Two elements a and b "conflict through x" if (a,b,x) is bad
    # But we need triple-level conflicts, not pair-level.
    
    # Let's use a different approach: for each x, compute the set of all
    # elements y such that there exists z with (x,y,z) bad.
    # Actually, the key insight: x can be added to set S iff
    # no pair in S forms a bad triple with x.
    
    # Precompute for each x: conflict_set[x] = set of y such that
    # there exists some z where (x,y,z) is a bad triple.
    # This is NOT sufficient because we need PAIR conflicts with x.
    
    # Better approach: For each x, precompute a boolean array can_coexist_with_x
    # where can_coexist_with_x[a][b] = True if (a,b,x) is NOT a bad triple.
    # But this is O(n^3) memory which is 128^3 = 2M, acceptable.
    
    # Actually, let's just precompute all bad triples as a set of sorted tuples.
    # Count them first to see if feasible.
    
    bad_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = lcm_mat[a][b]
            for c in range(b + 1, n + 1):
                if lcm_mat[a][c] == L and lcm_mat[b][c] == L:
                    bad_triples.add((a, b, c))
    
    # Now build for each element x: the set of pairs that form bad triples with x
    # pairs_conflicting_with[x] = set of (a,b) where (a,b,x) is a bad triple
    pairs_conflicting_with = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        # (a,b,c) is a bad triple, so for element c, pair (a,b) conflicts
        # for element a, pair (b,c) conflicts
        # for element b, pair (a,c) conflicts
        pairs_conflicting_with[c].add((a, b))
        pairs_conflicting_with[a].add((b, c))
        pairs_conflicting_with[b].add((a, c))
    
    # For faster lookup: convert to a set of encoded pairs
    # Encode pair (a,b) with a<b as a*n+b
    def encode_pair(a, b):
        if a > b:
            a, b = b, a
        return a * (n + 1) + b
    
    pairs_conflicting_enc = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for (a, b) in pairs_conflicting_with[x]:
            pairs_conflicting_enc[x].add(encode_pair(a, b))
    
    def can_add_to_set(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        sel_list = sorted(selected)
        if len(sel_list) < 2:
            return True
        conflict_set = pairs_conflicting_enc[x]
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                if encode_pair(sel_list[i], sel_list[j]) in conflict_set:
                    return False
        return True
    
    def greedy_add(order):
        selected = set()
        for x in order:
            if can_add_to_set(x, selected):
                selected.add(x)
        return selected
    
    random.seed(42)
    
    best_set = set()
    
    # Multiple greedy starts
    orders = [
        elements,
        list(reversed(elements)),
        # Primality-based ordering
        sorted(elements, key=lambda x: (len(math.gcd(x, 1) and [i for i in range(2, x+1) if x%i==0]) == 0 and 0 or 1, x)),
    ]
    
    for order in orders:
        s = greedy_add(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Random orders
    for _ in range(20):
        if time.time() > deadline:
            break
        order = elements[:]
        random.shuffle(order)
        s = greedy_add(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search with 1-add and 1-remove-2-add moves
    def local_search(current_set, max_iters=20000):
        current = set(current_set)
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try adding each unselected element
            for x in range(1, n + 1):
                if x in current:
                    continue
                if can_add_to_set(x, current):
                    current.add(x)
                    improved = True
                    break
            
            # Try 1-for-2 swaps
            if not improved:
                current_list = sorted(current)
                removed = None
                for x in current_list:
                    temp = current - {x}
                    # Try to add two elements
                    candidates = [y for y in range(1, n + 1) if y not in current]
                    found = False
                    for i in range(len(candidates)):
                        y1 = candidates[i]
                        if not can_add_to_set(y1, temp):
                            continue
                        temp_with_y1 = temp | {y1}
                        for j in range(i + 1, len(candidates)):
                            y2 = candidates[j]
                            if can_add_to_set(y2, temp_with_y1):
                                current = temp_with_y1 | {y2}
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        
        return current
    
    # Apply local search
    if time.time() < deadline:
        best_set = local_search(best_set, max_iters=30000)
    
    # More random restarts with local search
    for _ in range(15):
        if time.time() > deadline:
            break
        order = elements[:]
        random.shuffle(order)
        s = greedy_add(order)
        if len(s) > len(best_set):
            best_set = s
        if time.time() < deadline:
            s = local_search(s, max_iters=15000)
            if len(s) > len(best_set):
                best_set = s
    
    # Final: try all single removals and see if we can add more
    if time.time() < deadline:
        current = set(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in list(current):
                temp = current - {x}
                # Try to add elements
                for y in range(1, n + 1):
                    if y in temp:
                        continue
                    if can_add_to_set(y, temp):
                        new_set = temp | {y}
                        if len(new_set) > len(current):
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
    
    return sorted(current)
# EVOLVE-BLOCK-END
