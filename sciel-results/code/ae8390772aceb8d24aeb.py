"""
Linear triple systems on 16 points with small independence number.

Strategy:
  1. Generate a pool of candidate triples from the affine geometry AG(4,2):
     all 2-dimensional affine subspaces (4-flats) and their 4 constituent
     3-subsets, plus all 1-dimensional affine subspaces (lines, size 3).
     Also include random triples to broaden the search space.
  2. Greedily select triples to maximize coverage (minimize independence number)
     while maintaining the linearity constraint (any two triples share at most
     one point).
  3. Local search: try to swap in a different triple to further reduce alpha.
  4. Use bitmask-based exact independence number computation for evaluation.
"""

import time
import random
import itertools
from functools import lru_cache

# Deadline guard: must finish well before 150s
_DEADLINE = 140.0  # seconds
_start_time = None


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _triple_mask(triple):
    """Convert a triple (a,b,c) to a 16-bit mask."""
    m = 0
    for x in triple:
        m |= (1 << x)
    return m


def _mask_to_triple(m):
    """Convert a 16-bit mask back to a sorted tuple of 3 elements."""
    pts = [i for i in range(16) if (m >> i) & 1]
    return tuple(pts)


def _share_count(m1, m2):
    """Number of shared points between two masks."""
    return bin(m1 & m2).count('1')


def _is_linear(triples):
    """Check if any two triples share at most one point."""
    n = len(triples)
    masks = [_triple_mask(t) for t in triples]
    for i in range(n):
        for j in range(i + 1, n):
            if _share_count(masks[i], masks[j]) > 1:
                return False
    return True


def _calc_alpha(triples):
    """
    Exact independence number: largest subset of {0..15} containing no triple.
    Uses bitmask superset marking.
    """
    if not triples:
        return 16

    # Mark all masks that contain at least one triple
    # A set S is independent iff it contains no triple
    # We want the maximum |S| such that no triple is a subset of S.

    # Efficient approach: for each subset, check if it contains any triple.
    # With 16 points, there are 2^16 = 65536 subsets.
    # For each subset, check if any triple is contained in it.

    triple_masks = [_triple_mask(t) for t in triples]

    # For each subset mask, determine if it contains any triple
    # A subset S contains triple T iff T & ~S == 0 (T is subset of S)
    # i.e., T | S == S, or T & S == T

    best = 0
    # Iterate over all subsets by size, from largest to smallest
    # We can use the property: if a subset of size k contains a triple,
    # then all supersets also contain a triple.
    # So we want the largest subset with NO triple.

    # Precompute: for each triple mask, mark all its supersets as "covered"
    covered = [False] * (1 << 16)

    for tm in triple_masks:
        # All supersets of tm are covered
        # Supersets of tm: tm | extra where extra is any subset of ~tm
        complement = ((1 << 16) - 1) ^ tm
        # Enumerate all submasks of complement
        sub = complement
        while True:
            covered[tm | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & complement

    # Now find the largest subset that is NOT covered
    for size in range(16, -1, -1):
        if _remaining() < 5:
            break
        # Check all subsets of this size
        # Use combinatorial generation
        for subset in itertools.combinations(range(16), size):
            m = 0
            for p in subset:
                m |= (1 << p)
            if not covered[m]:
                return size

    return 0


def _calc_alpha_fast(triples):
    """
    Faster alpha computation using a different approach:
    For each point, track which triples it's in.
    Use the fact that alpha is the max size of a set with no complete triple.
    """
    if not triples:
        return 16

    triple_masks = [_triple_mask(t) for t in triples]
    n_triples = len(triple_masks)

    # For each subset of points, check if it contains any triple.
    # We'll iterate subsets from large to small.
    # Optimization: use the covered array approach but more efficiently.

    # Alternative: for each subset S, S is independent iff for all triples T, T not subset of S.
    # T not subset of S iff there exists x in T with x not in S, i.e., T & ~S != 0.

    # Let's use the covered marking but optimize the enumeration.
    covered = bytearray(1 << 16)  # Use bytearray for memory efficiency

    for tm in triple_masks:
        complement = ((1 << 16) - 1) ^ tm
        sub = complement
        while True:
            covered[tm | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & complement

    # Find largest uncovered subset
    for size in range(16, -1, -1):
        if _remaining() < 3:
            break
        for subset in itertools.combinations(range(16), size):
            m = 0
            for p in subset:
                m |= (1 << p)
            if not covered[m]:
                return size

    return 0


def _generate_ag42_triples():
    """
    Generate candidate triples from AG(4,2).
    Points are vectors in F_2^4, identified with 0..15.
    
    Lines (1-dim affine subspaces): {v, v+u} for u != 0, but that's size 2.
    Actually in AG(4,2), lines have 2 points. So we need higher dimensions.
    
    2-dim affine subspaces (planes) have 4 points. Each 4-set gives 4 triples.
    3-dim affine subspaces (hyperplanes) have 8 points.
    
    Let's generate all 4-point affine subspaces and extract their 3-subsets.
    Also generate all 3-point collinear sets... but in AG(4,2) lines are size 2.
    
    So the natural triples come from 4-point planes: each plane {a,b,c,d} 
    gives 4 triples: {a,b,c}, {a,b,d}, {a,c,d}, {b,c,d}.
    """
    triples = set()
    
    # All 2-dim affine subspaces of F_2^4
    # A 2-dim affine subspace is {v, v+u, v+w, v+u+w} where u,w are linearly independent
    # There are C(4,2) = 6 choices for the 2-dim subspace direction, and 2^2 = 4 cosets
    # Total: 6 * 4 = 24 planes, each with 4 points, giving 4 triples each = 96 triples
    
    # Enumerate all pairs of distinct nonzero vectors (u, w) that are linearly independent
    for u in range(1, 16):
        for w in range(u + 1, 16):
            # Check linear independence: u & w != 0 doesn't mean dependent in F_2
            # In F_2, u and w are dependent iff u == w (since only scalar is 1)
            # Wait, in F_2^4, two nonzero vectors are linearly dependent iff they are equal.
            # Because the only nonzero scalar is 1. So any two distinct nonzero vectors are independent.
            # But we need to avoid duplicates: the plane {v, v+u, v+w, v+u+w} is the same
            # as {v', v'+u', v'+w', v'+u'+w'} for different (u',w') from the same 2-dim subspace.
            
            # To avoid duplicates, fix a canonical form: u < w and u & w == 0? No, that's not right.
            # Actually, in F_2, a 2-dim subspace is spanned by any two independent vectors.
            # The number of 2-dim subspaces in F_2^4 is C(4,2) = 6.
            # Let's just enumerate all 2-dim subspaces properly.
            
            # Two vectors u, w are independent in F_2 iff u != w and u != 0 and w != 0.
            # Since we have u < w and both nonzero, they're independent.
            # But the same 2-dim subspace can be spanned by multiple pairs.
            # For a 2-dim subspace S with 3 nonzero vectors, there are 3 pairs.
            # To get each subspace once, we can require that u is the smallest element.
            
            # Simpler: just generate all and use a set to deduplicate.
            for v in range(16):
                plane = {v, v ^ u, v ^ w, v ^ u ^ w}
                # Generate all 4 triples from this 4-set
                pts = sorted(plane)
                for i in range(4):
                    for j in range(i + 1, 4):
                        for k in range(j + 1, 4):
                            triples.add((pts[i], pts[j], pts[k]))
    
    return list(triples)


def _greedy_construct(candidates, max_triples=32):
    """
    Greedily select triples to minimize alpha while maintaining linearity.
    """
    selected = []
    selected_masks = []
    used_points = 0  # bitmask of points that appear in selected triples
    
    # Sort candidates by some heuristic: prefer triples that cover "new" points
    # or that are in high-traffic areas
    
    for triple in candidates:
        if _remaining() < 2:
            break
        tm = _triple_mask(triple)
        
        # Check linearity with all selected triples
        valid = True
        for sm in selected_masks:
            if _share_count(tm, sm) > 1:
                valid = False
                break
        
        if valid:
            selected.append(triple)
            selected_masks.append(tm)
            used_points |= tm
    
    return selected


def _local_search(triples, candidates, max_iterations=200):
    """
    Try to improve by swapping triples.
    """
    if not triples:
        return triples, _calc_alpha_fast(triples)
    
    current_alpha = _calc_alpha_fast(triples)
    best_triples = list(triples)
    best_alpha = current_alpha
    
    selected_masks = [_triple_mask(t) for t in triples]
    
    for iteration in range(max_iterations):
        if _remaining() < 5:
            break
        
        improved = False
        n = len(triples)
        
        # Try removing each triple and replacing with a better one
        for i in range(n):
            if _remaining() < 5:
                break
            
            # Remove triple i
            remaining = [t for j, t in enumerate(triples) if j != i]
            remaining_masks = [m for j, m in enumerate(selected_masks) if j != i]
            
            # Try adding each candidate
            for cand in candidates:
                cm = _triple_mask(cand)
                
                # Check if cand is already in remaining
                if cm in remaining_masks:
                    continue
                
                # Check linearity
                valid = True
                for rm in remaining_masks:
                    if _share_count(cm, rm) > 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_triples = remaining + [cand]
                new_alpha = _calc_alpha_fast(new_triples)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples
                    selected_masks = [_triple_mask(t) for t in best_triples]
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_triples, best_alpha


def _random_greedy(candidates, n_triples_target=24, seed=None):
    """
    Randomized greedy construction.
    """
    rng = random.Random(seed)
    selected = []
    selected_masks = []
    
    cand_list = list(candidates)
    rng.shuffle(cand_list)
    
    for triple in cand_list:
        if len(selected) >= n_triples_target:
            break
        if _remaining() < 2:
            break
        
        tm = _triple_mask(triple)
        
        valid = True
        for sm in selected_masks:
            if _share_count(tm, sm) > 1:
                valid = False
                break
        
        if valid:
            selected.append(triple)
            selected_masks.append(tm)
    
    return selected


def construct():
    global _start_time
    _start_time = time.time()
    
    # Step 1: Generate candidate triples
    ag42_triples = _generate_ag42_triples()
    
    # Also add all possible triples for broader search (subset)
    # Too many (C(16,3) = 560), so add a random subset
    all_triples = list(itertools.combinations(range(16), 3))
    random.seed(42)
    random_subset = random.sample(all_triples, 200)
    
    candidates = list(set(ag42_triples + random_subset))
    
    # Step 2: Initial greedy construction
    initial = _greedy_construct(candidates, max_triples=32)
    if not initial:
        # Fallback: just use disjoint triples
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Evaluate
    alpha = _calc_alpha_fast(initial)
    best_triples = list(initial)
    best_alpha = alpha
    
    # Step 3: Local search
    if _remaining() > 10:
        best_triples, best_alpha = _local_search(initial, candidates, max_iterations=100)
    
    # Step 4: Try multiple random greedy starts
    if _remaining() > 15:
        for trial in range(20):
            if _remaining() < 10:
                break
            
            seed = 1000 + trial
            trial_triples = _random_greedy(candidates, n_triples_target=24, seed=seed)
            if not trial_triples:
                continue
            
            trial_alpha = _calc_alpha_fast(trial_triples)
            
            if trial_alpha < best_alpha:
                best_alpha = trial_alpha
                best_triples = trial_triples
            
            # Local search on this trial
            if _remaining() > 15 and trial_alpha <= best_alpha:
                ls_triples, ls_alpha = _local_search(trial_triples, candidates, max_iterations=50)
                if ls_alpha < best_alpha:
                    best_alpha = ls_alpha
                    best_triples = ls_triples
    
    # Step 5: Final polish - try to remove triples that don't help
    if _remaining() > 5:
        for i in range(len(best_triples)):
            if _remaining() < 3:
                break
            without_i = [t for j, t in enumerate(best_triples) if j != i]
            if not without_i:
                continue
            a = _calc_alpha_fast(without_i)
            if a <= best_alpha:
                best_triples = without_i
                best_alpha = a
    
    # Convert to list of lists
    result = [list(t) for t in best_triples]
    
    # Final validation
    if not _is_linear(result):
        # Fallback to known valid solution
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return result
