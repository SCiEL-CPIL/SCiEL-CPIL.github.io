"""
Maximize the number of triples on 16 points with no K(2,2,2) transversal.
We use a greedy randomized construction with backtracking-lite local search,
plus a time-bounded simulated annealing phase.
"""
import time
import random
import itertools
from collections import defaultdict

def _all_triples():
    """Generate all C(16,3) triples as sorted tuples."""
    return [tuple(t) for t in itertools.combinations(range(16), 3)]

def _build_pattern_incidence():
    """
    Precompute all K(2,2,2) patterns and their incidence with triples.
    Returns:
        patterns: list of lists of 8 triples (tuples)
        triple_to_patterns: dict mapping triple -> list of pattern indices
    """
    patterns = []
    triple_to_patterns = defaultdict(list)
    
    # Generate all 6-point subsets
    for six in itertools.combinations(range(16), 6):
        a, b, c, d, e, f = six
        # 15 ways to partition 6 elements into 3 unordered pairs
        pairings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in pairings:
            pat_triples = []
            for x in (0, 1):
                for y in (0, 1):
                    for z in (0, 1):
                        t = tuple(sorted((p1[x], p2[y], p3[z])))
                        pat_triples.append(t)
            
            # Check for duplicates in pattern (should not happen for distinct points)
            # But just in case, ensure unique
            unique_pat = list(set(pat_triples))
            if len(unique_pat) != 8:
                continue # Should not happen
                
            pat_idx = len(patterns)
            patterns.append(unique_pat)
            for t in unique_pat:
                triple_to_patterns[t].append(pat_idx)
                
    return patterns, triple_to_patterns

def _greedy_construct(seed=42, patterns=None, triple_to_patterns=None):
    """Greedy construction: add triples in random order, skip if it creates K(2,2,2)."""
    rng = random.Random(seed)
    all_triples = list(itertools.combinations(range(16), 3))
    rng.shuffle(all_triples)
    
    selected = []
    selected_set = set()
    pattern_counts = [0] * len(patterns)
    
    for t in all_triples:
        # Check if adding t would complete any K(2,2,2)
        valid = True
        affected_patterns = triple_to_patterns.get(t, [])
        for p_idx in affected_patterns:
            if pattern_counts[p_idx] == 7:
                valid = False
                break
        
        if valid:
            selected.append(list(t))
            selected_set.add(t)
            for p_idx in affected_patterns:
                pattern_counts[p_idx] += 1
        # else: skip t
    
    return selected

def _local_search(triples, patterns, triple_to_patterns, max_iters=50000, seed=42, deadline=None):
    """Try to improve by swapping triples or adding new ones."""
    rng = random.Random(seed)
    current = set(map(tuple, triples))
    current_list = list(current)
    best = len(current_list)
    
    all_triples = list(itertools.combinations(range(16), 3))
    not_in_current = [t for t in all_triples if t not in current]
    
    # Initialize pattern counts
    pattern_counts = [0] * len(patterns)
    for t in current:
        for p_idx in triple_to_patterns.get(t, []):
            pattern_counts[p_idx] += 1
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try to add a random triple not in current
        if not_in_current:
            idx = rng.randrange(len(not_in_current))
            t = not_in_current[idx]
            
            # Check validity
            valid = True
            affected = triple_to_patterns.get(t, [])
            for p_idx in affected:
                if pattern_counts[p_idx] == 7:
                    valid = False
                    break
            
            if valid:
                # Commit
                current.add(t)
                current_list.append(t)
                not_in_current.pop(idx)
                for p_idx in affected:
                    pattern_counts[p_idx] += 1
                best = max(best, len(current_list))
        
        # Try to remove a random triple and add two others (swap)
        elif current_list:
            idx = rng.randrange(len(current_list))
            removed = current_list[idx]
            
            # Remove 'removed'
            current.remove(removed)
            current_list.pop(idx)
            
            # Update counts
            affected_rem = triple_to_patterns.get(removed, [])
            for p_idx in affected_rem:
                pattern_counts[p_idx] -= 1
            
            # Try to add two random triples
            candidates = [t for t in all_triples if t not in current]
            if len(candidates) >= 2:
                t1, t2 = rng.sample(candidates, 2)
                
                # Check if t1 and t2 can be added
                # We need to check if adding t1 creates a conflict, then t2.
                # But they might interact.
                # Simple check:
                # 1. Check t1 validity against current
                # 2. Check t2 validity against current + t1
                
                valid_1 = True
                aff_1 = triple_to_patterns.get(t1, [])
                for p_idx in aff_1:
                    if pattern_counts[p_idx] == 7:
                        valid_1 = False
                        break
                
                if valid_1:
                    # Tentatively add t1
                    for p_idx in aff_1:
                        pattern_counts[p_idx] += 1
                    
                    valid_2 = True
                    aff_2 = triple_to_patterns.get(t2, [])
                    for p_idx in aff_2:
                        if pattern_counts[p_idx] == 7:
                            valid_2 = False
                            break
                    
                    if valid_2:
                        # Commit t1 and t2
                        current.add(t1)
                        current.add(t2)
                        current_list.append(t1)
                        current_list.append(t2)
                        
                        for p_idx in aff_2:
                            pattern_counts[p_idx] += 1
                        
                        # Update not_in_current
                        if t1 in not_in_current: not_in_current.remove(t1)
                        if t2 in not_in_current: not_in_current.remove(t2)
                        
                        best = max(best, len(current_list))
                    else:
                        # Revert t1
                        for p_idx in aff_1:
                            pattern_counts[p_idx] -= 1
                else:
                    # Revert removal of 'removed'
                    current.add(removed)
                    current_list.append(removed)
                    for p_idx in affected_rem:
                        pattern_counts[p_idx] += 1

    return current_list

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute structures
    patterns, triple_to_patterns = _build_pattern_incidence()
    
    best_triples = []
    best_len = 0
    
    # Multiple random greedy constructions
    seed = 42
    while time.time() < deadline - 5:
        triples = _greedy_construct(seed, patterns, triple_to_patterns)
        if len(triples) > best_len:
            best_len = len(triples)
            best_triples = triples
        seed += 1
    
    # Local search on the best
    if time.time() < deadline - 2:
        improved = _local_search(best_triples, patterns, triple_to_patterns, max_iters=1000000, seed=123, deadline=deadline - 1)
        if len(improved) > best_len:
            best_len = len(improved)
            best_triples = improved
    
    return best_triples
