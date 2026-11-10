"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

P = {'n': 12, 'm': 40}

def _generate_family(n, m, seed=42):
    """
    Generate a family of m distinct subsets of {0..n-1} that maximizes
    the number of comparable (strict subset) pairs.
    
    Strategy: 
    1. Initialize with a "Ball" (down-set) of size m.
    2. Use Local Search (Hill Climbing) with swaps to optimize the count.
    """
    all_subsets = []
    for r in range(n + 1):
        for comb in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(comb))
            
    # Map subset to index for quick lookup? Or just use set of frozensets.
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        n_fam = len(family)
        for i in range(n_fam):
            for j in range(i + 1, n_fam):
                if family[i] < family[j] or family[j] < family[i]:
                    count += 1
        return count

    def calculate_delta(family_set, a, b):
        """
        Calculate the change in comparable pairs if we remove 'a' and add 'b'.
        family_set is the set of current elements (including a, excluding b).
        """
        # Pairs involving 'a' are lost. Pairs involving 'b' are gained.
        # Note: 'a' and 'b' are not in the set of 'other' elements.
        delta = 0
        for v in family_set:
            if v == a:
                continue
            # Check if a is comparable with v
            comp_a = 1 if (a < v or v < a) else 0
            # Check if b is comparable with v
            comp_b = 1 if (b < v or v < b) else 0
            delta += comp_b - comp_a
        return delta

    def local_search(family, max_iterations=1000):
        """
        Perform hill climbing on the family.
        """
        family_set = set(family)
        current_score = count_comparable(family)
        
        for _ in range(max_iterations):
            best_delta = 0
            best_a = None
            best_b = None
            
            # Try all swaps: remove a from family, add b from outside
            for a in family_set:
                for b in all_subsets:
                    if b in family_set:
                        continue
                    delta = calculate_delta(family_set, a, b)
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_a is not None and best_delta > 0:
                family_set.remove(best_a)
                family_set.add(best_b)
                current_score += best_delta
            else:
                break
        
        return list(family_set), current_score

    def initial_ball(m):
        """
        Construct a down-set (ball) of size m.
        Priority: smaller sizes first.
        """
        # Sort subsets by size, then lexicographically (for determinism)
        sorted_subsets = sorted(all_subsets, key=lambda x: (len(x), x))
        family = []
        for s in sorted_subsets:
            if len(family) == m:
                break
            family.append(s)
        return family

    # Generate initial family
    family = initial_ball(m)
    
    # Run local search
    optimized_family, score = local_search(family)
    
    # Convert to lists of integers
    result = [sorted(list(s)) for s in optimized_family]
    return result, score

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
