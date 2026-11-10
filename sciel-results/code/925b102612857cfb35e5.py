"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

P = {'n': 12, 'm': 40}

def _generate_family(n, m, seed=42):
    """
    Generate a family of m distinct subsets of {0..n-1} that maximizes
    the number of comparable (strict subset) pairs.
    
    Strategy: Use a greedy construction that prioritizes adding sets
    which have high comparability with existing sets. We also try
    multiple random restarts and keep the best.
    """
    all_subsets = []
    for r in range(n + 1):
        for comb in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(comb))
    
    # Precompute comparability: for each pair, check if one is strict subset of other
    # But that's 4096^2 / 2 which is ~8 million, manageable
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        n_fam = len(family)
        for i in range(n_fam):
            for j in range(i + 1, n_fam):
                if family[i] < family[j] or family[j] < family[i]:
                    count += 1
        return count
    
    def greedy_construct(m, seed):
        rng = random.Random(seed)
        # Start with empty set and full set as they are comparable with many
        # Actually, let's use a smarter greedy:
        # At each step, pick the subset that maximizes the number of comparable
        # pairs it forms with already selected subsets.
        
        selected = []
        remaining = list(all_subsets)
        rng.shuffle(remaining)
        
        # Precompute subset relations for efficiency
        # subset_map[s] = list of indices in remaining that are strict subsets of s
        # superset_map[s] = list of indices in remaining that are strict supersets of s
        
        # Actually, let's just do a direct greedy with incremental scoring
        # For m=40, this is fast enough
        
        selected_set = set()
        family = []
        
        for step in range(m):
            best_score = -1
            best_candidate = None
            for candidate in remaining:
                if candidate in selected_set:
                    continue
                # Count how many existing sets are comparable with candidate
                score = 0
                for existing in family:
                    if candidate < existing or existing < candidate:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is None:
                # Pick randomly
                idx = rng.randrange(len(remaining))
                best_candidate = remaining[idx]
                remaining.pop(idx)
            else:
                remaining.remove(best_candidate)
            
            family.append(best_candidate)
            selected_set.add(best_candidate)
        
        return family
    
    # Try multiple seeds and keep the best
    best_family = None
    best_score = -1
    
    for seed in range(50):
        family = greedy_construct(m, seed)
        score = count_comparable(family)
        if score > best_score:
            best_score = score
            best_family = family
    
    # Convert to lists of integers
    result = [sorted(list(s)) for s in best_family]
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
