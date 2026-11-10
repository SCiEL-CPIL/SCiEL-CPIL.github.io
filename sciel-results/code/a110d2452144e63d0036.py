"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict, Counter

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Build mapping: element -> set of bad triples it's in
    elem_triples = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples[x].add(t)
    
    # bad_pairs_for[x] = set of (a, b) such that (x, a, b) is a bad triple (a < b)
    bad_pairs_for = {i: set() for i in range(1, n + 1)}
    for (a, b, c) in bad_triples:
        bad_pairs_for[a].add((b, c))
        bad_pairs_for[b].add((a, c))
        bad_pairs_for[c].add((a, b))
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without creating a bad triple."""
        for (a, b) in bad_pairs_for[x]:
            if a in current_set and b in current_set:
                return False
        return True
    
    def local_improve(selected):
        """Improve by adding elements and 1-for-2 swaps."""
        selected = list(selected)
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            in_set = set(selected)
            
            # Try adding any missing element
            for x in range(1, n + 1):
                if x not in in_set:
                    if can_add(x, in_set):
                        selected.append(x)
                        in_set.add(x)
                        improved = True
                        if time.time() > deadline - 3:
                            return selected
            
            # Try 1-for-2 swaps: remove one element, try to add two
            if time.time() > deadline - 3:
                break
            for i in range(len(selected)):
                if time.time() > deadline - 3:
                    break
                a = selected[i]
                temp = selected[:i] + selected[i+1:]
                temp_set = set(temp)
                added = 0
                for x in range(1, n + 1):
                    if x not in temp_set:
                        if can_add(x, temp_set):
                            temp.append(x)
                            temp_set.add(x)
                            added += 1
                            if added >= 2:
                                break
                if len(temp) > len(selected):
                    selected = temp
                    improved = True
                    break
        return selected
    
    def hitting_set_removal(order_hint=None):
        """Greedy hitting set: remove elements to cover all bad triples."""
        in_set = [True] * (n + 1)
        current_set = set(range(1, n + 1))
        remaining_bad = set(bad_triples)
        
        while remaining_bad:
            if time.time() > deadline - 5:
                break
            
            # Count occurrences of each element in remaining_bad
            counts = Counter()
            for t in remaining_bad:
                for x in t:
                    if in_set[x]:
                        counts[x] += 1
            
            if not counts:
                break
                
            max_count = max(counts.values())
            candidates = [x for x, c in counts.items() if c == max_count]
            
            if order_hint and len(candidates) > 1:
                # Use order_hint to break ties
                cand_set = set(candidates)
                for x in order_hint:
                    if x in cand_set:
                        x = x
                        break
                else:
                    x = random.choice(candidates)
            else:
                x = random.choice(candidates)
            
            in_set[x] = False
            current_set.remove(x)
            remaining_bad = {t for t in remaining_bad if x not in t}
        
        return current_set
    
    best_size = 0
    best_set = set()
    
    # Precompute some orderings for tie-breaking
    def num_divisors(x):
        count = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count
    
    order_by_divisors = sorted(range(1, n + 1), key=num_divisors)
    order_by_divisors_rev = list(reversed(order_by_divisors))
    order_natural = list(range(1, n + 1))
    order_natural_rev = list(reversed(order_natural))
    
    # Multiple restarts with different strategies
    random.seed(42)
    num_restarts = 80
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Vary the removal strategy
        if r % 4 == 0:
            # Pure random
            result = hitting_set_removal()
        elif r % 4 == 1:
            # Tie-break by fewest divisors
            result = hitting_set_removal(order_by_divisors)
        elif r % 4 == 2:
            # Tie-break by most divisors
            result = hitting_set_removal(order_by_divisors_rev)
        else:
            # Random shuffle hint
            shuffled = list(range(1, n + 1))
            random.shuffle(shuffled)
            result = hitting_set_removal(shuffled)
        
        # Local improvement
        if time.time() < deadline - 5:
            result = local_improve(list(result))
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    # Final local improvement on best
    if time.time() < deadline - 3:
        final = local_improve(list(best_set))
        if len(final) > best_size:
            best_set = set(final)
    
    if not best_set:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END
