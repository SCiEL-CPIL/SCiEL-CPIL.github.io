"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    max_n = 128
    nums = list(range(1, max_n + 1))
    
    # Precompute lcm table
    lcm_table = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    for i in range(1, max_n + 1):
        for j in range(i, max_n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict hyperedges: for each triple (a,b,c), check if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # A triple is "bad" if all three pairwise LCMs are equal.
    # We'll represent conflicts as: for each element x, a set of pairs (a,b) in the current selection
    # that would form a bad triple with x.
    
    # Alternative approach: for each element x, precompute which pairs (a,b) with a,b < x 
    # (in some ordering) would conflict with x.
    
    # Let's use a different representation:
    # For each x, compute the set of "conflict pairs" - pairs (a,b) such that 
    # lcm(a,b) == lcm(a,x) == lcm(b,x).
    
    # Actually, let's think about it differently.
    # Three distinct elements a, b, c form a bad triple iff:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Let L = lcm(a,b) = lcm(a,c) = lcm(b,c)
    # This means a|L, b|L, c|L, and for each pair, the lcm is exactly L.
    # So a, b, c are all divisors of L, and no proper divisor of L contains two of them with lcm < L... 
    # Actually more precisely: lcm(a,b)=L means that for every prime power p^k || L, 
    # at least one of a or b has p^k as its p-adic valuation.
    
    # Let's just precompute all bad triples.
    bad_triples = set()
    for a in range(1, max_n + 1):
        for b in range(a + 1, max_n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, max_n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.add((a, b, c))
    
    # Now we want to find the maximum subset with no bad triple.
    # This is a 3-uniform hypergraph independent set problem.
    
    # Use randomized greedy with multiple orderings.
    
    def greedy_select(order):
        selected = set()
        # For efficiency, track for each candidate x, how many pairs in selected would conflict
        # Actually, let's just check directly.
        for x in order:
            # Check if adding x creates any bad triple
            ok = True
            selected_list = list(selected)
            for i in range(len(selected_list)):
                a = selected_list[i]
                for j in range(i + 1, len(selected_list)):
                    b = selected_list[j]
                    if (min(a,b,x), min(max(a,b),x), max(a,b,x)) in bad_triples:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.add(x)
        return selected
    
    # Better: precompute for each x, the list of pairs (a,b) with a<b that conflict with x
    # i.e., (a,b,x) is a bad triple.
    
    conflict_pairs = {x: [] for x in range(1, max_n + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    def greedy_select_fast(order):
        selected = set()
        # For each pair in selected, track it
        selected_pairs = set()
        for x in order:
            # Check if any conflict pair for x is a subset of selected
            ok = True
            for (a, b) in conflict_pairs[x]:
                if a in selected and b in selected:
                    ok = False
                    break
            if ok:
                selected.add(x)
        return selected
    
    best = set()
    
    # Try ascending order first
    order = list(range(1, max_n + 1))
    result = greedy_select_fast(order)
    if len(result) > len(best):
        best = result
    
    # Try descending order
    order = list(range(max_n, 0, -1))
    result = greedy_select_fast(order)
    if len(result) > len(best):
        best = result
    
    # Try random orders
    rng = random.Random(42)
    for trial in range(2000):
        if time.time() > deadline:
            break
        order = list(range(1, max_n + 1))
        rng.shuffle(order)
        result = greedy_select_fast(order)
        if len(result) > len(best):
            best = result
    
    # Local search: try to improve by removing one element and adding others
    # Or try a simple hill climbing: remove a random element, re-greedy with that removed
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best)
        # Try removing each element and re-running greedy
        for x in list(current):
            if time.time() > deadline:
                break
            remaining = [y for y in range(1, max_n + 1) if y not in current or y == x]
            # Actually, let's try: remove x, then greedily add elements not in current
            base = set(current) - {x}
            candidates = [y for y in range(1, max_n + 1) if y not in base]
            rng.shuffle(candidates)
            # Greedily add from candidates
            for y in candidates:
                ok = True
                for (a, b) in conflict_pairs[y]:
                    if a in base and b in base:
                        ok = False
                        break
                if ok:
                    base.add(y)
            if len(base) > len(current):
                current = base
                improved = True
                break
        best = current
    
    # Final local search: try swapping
    if time.time() < deadline - 5:
        current = set(best)
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try removing one and adding another
            for x in list(current):
                if time.time() > deadline - 5:
                    break
                base = set(current) - {x}
                candidates = [y for y in range(1, max_n + 1) if y not in base]
                rng.shuffle(candidates)
                for y in candidates:
                    ok = True
                    for (a, b) in conflict_pairs[y]:
                        if a in base and b in base:
                            ok = False
                            break
                    if ok:
                        base.add(y)
                        break
                if len(base) > len(current):
                    current = base
                    improved = True
                    break
            if not improved:
                break
        if len(current) > len(best):
            best = current
    
    return list(best)
# EVOLVE-BLOCK-END
