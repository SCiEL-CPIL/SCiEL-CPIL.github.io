"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

P = {'n': 12, 'm': 40}

def _generate_family(n, m, seed=42):
    """
    Generate a family of m distinct subsets of {0..n-1} that maximizes
    the number of comparable (strict subset) pairs.
    
    Strategy: Use a Lagrangian-inspired min-cut formulation.
    1. Enumerate all 2^n subsets
    2. Precompute comparable pairs
    3. Use a flow-based / local search hybrid to find optimal selection
    """
    n = P['n']
    m = P['m']
    
    # Enumerate all subsets as bitmasks
    all_subsets = list(range(1 << n))
    total = 1 << n  # 4096
    
    # Precompute size of each subset
    sizes = [bin(x).count('1') for x in all_subsets]
    
    # Precompute comparable pairs: for each pair (i, j), check if one is subset of other
    # Two subsets A, B are comparable if A & B == A or A & B == B (i.e., A⊆B or B⊆A)
    # They are strictly comparable if A⊂B or B⊂A (not equal)
    
    # Build adjacency: for each node, list of nodes it's strictly comparable with
    # This is O(total^2) = ~8.4M operations, should be fast
    
    # For efficiency, group by size
    by_size = [[] for _ in range(n + 1)]
    for i in range(total):
        by_size[sizes[i]].append(i)
    
    # comparable[i] = list of j such that i ⊂ j or j ⊂ i (strict)
    comparable = [[] for _ in range(total)]
    
    for i in range(total):
        si = sizes[i]
        for j in range(total):
            if i == j:
                continue
            sj = sizes[j]
            # Check if i ⊂ j or j ⊂ i
            # i ⊂ j iff (i & j) == i and si < sj
            # j ⊂ i iff (i & j) == j and sj < si
            if si < sj and (i & j) == i:
                comparable[i].append(j)
            elif sj < si and (i & j) == j:
                comparable[i].append(j)
    
    def count_comparable(family_mask):
        """Count comparable pairs in the selected family."""
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                a = family_mask[i]
                b = family_mask[j]
                sa = sizes[a]
                sb = sizes[b]
                if sa < sb and (a & b) == a:
                    count += 1
                elif sb < sa and (a & b) == b:
                    count += 1
        return count
    
    def local_search(family_mask, rng, iterations=200):
        """Improve the family via swap-based local search."""
        family_set = set(family_mask)
        selected = list(family_mask)
        
        def get_score(sel):
            cnt = 0
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    a = sel[i]
                    b = sel[j]
                    sa = sizes[a]
                    sb = sizes[b]
                    if sa < sb and (a & b) == a:
                        cnt += 1
                    elif sb < sa and (a & b) == b:
                        cnt += 1
            return cnt
        
        current_score = get_score(selected)
        
        for _ in range(iterations):
            improved = False
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    new_family = selected[:]
                    # Swap: remove selected[i], add j if not already present
                    if j in family_set:
                        continue
                    old_i = selected[i]
                    new_family[i] = j
                    new_score = get_score(new_family)
                    if new_score > current_score:
                        selected = new_family
                        family_set.discard(old_i)
                        family_set.add(j)
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return selected, current_score
    
    # Initial construction: use greedy with Lagrangian-inspired scoring
    def greedy_construct(m, seed):
        rng = random.Random(seed)
        remaining = list(range(total))
        rng.shuffle(remaining)
        selected = []
        selected_set = set()
        
        for step in range(m):
            best_score = -1
            best_candidate = None
            for candidate in remaining:
                if candidate in selected_set:
                    continue
                # Score: number of comparable pairs with already selected
                score = 0
                for existing in selected:
                    sc = sizes[candidate]
                    se = sizes[existing]
                    if sc < se and (candidate & existing) == candidate:
                        score += 1
                    elif se < sc and (candidate & existing) == existing:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is None:
                idx = rng.randrange(len(remaining))
                best_candidate = remaining[idx]
                remaining.pop(idx)
            else:
                remaining.remove(best_candidate)
            
            selected.append(best_candidate)
            selected_set.add(best_candidate)
        
        return selected
    
    # Try multiple seeds, do local search on each, keep best
    best_family = None
    best_score = -1
    
    for seed in range(30):
        family = greedy_construct(m, seed)
        family, score = local_search(family, random.Random(seed), iterations=100)
        if score > best_score:
            best_score = score
            best_family = family
    
    # Convert to lists of integers
    result = []
    for mask in best_family:
        elems = [i for i in range(n) if (mask >> i) & 1]
        result.append(elems)
    
    return result, best_score

def construct():
    family, score = _generate_family(P['n'], P['m'])
    # Ensure exactly m distinct subsets
    assert len(family) == P['m'], f"Expected {P['m']} subsets, got {len(family)}"
    # Ensure all are distinct
    assert len(set(tuple(s) for s in family)) == P['m'], "Subsets are not distinct"
    # Ensure all elements are integers in range
    for s in family:
        for x in s:
            assert isinstance(x, int) and not isinstance(x, bool), f"Non-int element: {x}"
            assert 0 <= x < P['n'], f"Element out of range: {x}"
        assert len(s) == len(set(s)), "Duplicate elements in subset"
    return family
# EVOLVE-BLOCK-END
