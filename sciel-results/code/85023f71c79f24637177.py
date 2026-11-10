"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    # Find all bad triples: {a,b,c} where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For each pair (a,b), find witnesses z
    # bad_triples: list of frozensets of 3 elements
    bad_triples = []
    triple_set = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    t = frozenset((a, b, c))
                    if t not in triple_set:
                        triple_set.add(t)
                        bad_triples.append(t)
    
    # For each element x, precompute:
    # - conflict_pairs[x]: set of (a,b) pairs such that {a,b,x} is a bad triple
    #   (i.e., x is a witness for pair (a,b))
    # - For fast checking: for a selected set S, x can be added iff
    #   no pair in S is in conflict_pairs[x]
    conflict_pairs = [set() for _ in range(n + 1)]
    for t in bad_triples:
        a, b, c = sorted(t)
        # a is witness for (b,c), b for (a,c), c for (a,b)
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    # For even faster checking, for each x, build a bitmask of elements
    # such that if any two of those elements are both in S, x cannot be added
    # Actually, the conflict is per-pair, so we need to check pairs.
    # Alternative: for each x, precompute a list of "forbidden pairs"
    # and use a fast pair-membership test.
    
    # Build a set of all forbidden pairs for each x
    # forbidden_pairs[x] = set of (a,b) with a<b, such that {a,b,x} is bad
    forbidden_pairs = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for (a, b) in conflict_pairs[x]:
            forbidden_pairs[x].add((a, b))
    
    # Precompute: for each pair (a,b), which elements are witnesses
    # witness_map[(a,b)] = set of x such that {a,b,x} is bad
    witness_map = defaultdict(set)
    for (a, b) in set(p for x in range(1, n+1) for p in forbidden_pairs[x]):
        pass  # We'll build this differently
    
    # Actually, let's just use the forbidden_pairs structure directly.
    # For can_add(x, S): check if any pair in S is in forbidden_pairs[x]
    # To speed this up, we can precompute for each x a sorted list of forbidden pairs
    # and use a different approach.
    
    # Better approach: for each x, build a bitset of "conflict partners"
    # If x is in S, then for any other y in S, the pair (x,y) must not be forbidden for any z in S
    # This is getting complex. Let's use a simpler but effective approach:
    # For each x, precompute a set of "conflicting pairs" and check membership.
    
    # Even better: use the fact that the constraint is 3-uniform.
    # When adding x to S, we need to check all pairs (a,b) in S where a,b < x (or just all pairs)
    # to see if {a,b,x} is a bad triple.
    
    # Precompute: for each x, a set of pairs (a,b) with a<b, a!=x, b!=x
    # such that {a,b,x} is bad. This is forbidden_pairs[x].
    # can_add(x, S): for each pair (a,b) in S with a<b, check if (a,b) in forbidden_pairs[x]
    
    # To make this faster, for each x, precompute a "conflict graph" where
    # edges represent pairs that conflict with x. Then we need to check if S
    # contains any edge from this graph.
    
    # Let's use a different strategy: backtracking with branch and bound
    # on the hypergraph independent set problem.
    
    # First, let's get a good initial solution using greedy
    def can_add(x, selected):
        """Check if x can be added to selected without forming a bad triple."""
        sel_list = sorted(selected)
        k = len(sel_list)
        for i in range(k):
            a = sel_list[i]
            for j in range(i + 1, k):
                b = sel_list[j]
                # Check if {a,b,x} is a bad triple
                # We need to check if (a,b) is in forbidden_pairs[x]
                # or (a,x) is in forbidden_pairs[b] or (b,x) is in forbidden_pairs[a]
                # But since we're adding x to selected, we only need to check
                # if any pair in selected forms a bad triple with x.
                # That is: is there (a,b) in forbidden_pairs[x]?
                if (a, b) in forbidden_pairs[x]:
                    return False
        return True
    
    def greedy_add(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    # Precompute conflict counts for ordering
    conflict_counts = [0] * (n + 1)
    for x in range(1, n + 1):
        conflict_counts[x] = len(forbidden_pairs[x])
    
    # Get initial solution via greedy with multiple orderings
    random.seed(42)
    best_set = set()
    
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: conflict_counts[x]),
        sorted(range(1, n + 1), key=lambda x: -conflict_counts[x]),
    ]
    for _ in range(5):
        o = list(range(1, n + 1))
        random.shuffle(o)
        orderings.append(o)
    
    for order in orderings:
        s = greedy_add(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Now use branch-and-bound backtracking to improve
    # We'll work with the complement: try to find a larger independent set
    # in the 3-uniform hypergraph.
    
    # For branch and bound, we need:
    # 1. A good upper bound on how many more elements we can add
    # 2. Forward checking to prune
    
    # Let's implement a local search that tries to swap elements
    def local_search(initial_set, max_iters=500):
        """Local search: try to improve by removing one and adding another."""
        current_set = set(initial_set)
        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 3 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try to add an element first
            for in_elem in range(1, n + 1):
                if in_elem in current_set:
                    continue
                if can_add(in_elem, current_set):
                    current_set.add(in_elem)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            for out_elem in list(current_set):
                if time.time() > deadline - 3:
                    break
                trial_base = current_set - {out_elem}
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add(in_elem, trial_base):
                        current_set = trial_base | {in_elem}
                        improved = True
                        break
                if improved:
                    break
        return current_set
    
    best_set = local_search(best_set)
    
    # Multiple restarts with local search
    for i in range(10):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = greedy_add(order)
        s = local_search(s, max_iters=200)
        if len(s) > len(best_set):
            best_set = s
    
    # Final pass: try to add any remaining elements
    if time.time() < deadline - 2:
        for in_elem in range(1, n + 1):
            if in_elem in best_set:
                continue
            if can_add(in_elem, best_set):
                best_set.add(in_elem)
    
    # Verify and return
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
