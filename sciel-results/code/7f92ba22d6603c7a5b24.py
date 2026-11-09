"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _independence_number_fast(triples, n=16):
    """Compute independence number using bitmask DP over all 2^16 subsets."""
    if not triples:
        return n
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # For each subset, check if it contains any triple
    # independent[mask] = True if no triple is fully contained in mask
    # We want the largest mask with independent[mask] = True
    full = (1 << n) - 1
    # Precompute which masks contain each triple
    # A mask m contains triple tm iff (m & tm) == tm
    # We'll iterate masks from full down to 0
    
    # Better approach: for each mask, check all triples
    # But that's 2^16 * |triples| which could be slow
    # Alternative: use inclusion or just iterate
    
    # Actually, let's use a different approach:
    # A subset S is independent iff for every triple t, S does not contain all of t
    # Equivalently, S is independent iff S is not a superset of any triple
    
    # For each triple, mark all supersets as non-independent
    # non_indep[mask] = True if mask contains some triple
    non_indep = [False] * (1 << n)
    for tm in triple_masks:
        # All masks that are supersets of tm
        # mask must have all bits of tm set
        # Iterate over all masks that contain tm
        remaining = full ^ tm  # bits not in tm
        # For each subset of remaining, add to tm
        r = remaining
        while True:
            mask = tm | r
            non_indep[mask] = True
            if r == 0:
                break
            r = (r - 1) & remaining
    
    # Find largest independent set
    for size in range(n, 0, -1):
        found = False
        for mask in range(1 << n):
            if bin(mask).count('1') == size and not non_indep[mask]:
                return size
        if found:
            return size
    return 0

def _independence_number(triples, n=16):
    """Compute the largest subset of points containing no triple using bitmasks."""
    if not triples:
        return n
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check sizes from n down to 1
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            # Check if any triple is fully contained
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                return size
    return 0

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _affine_lines_f24():
    """Generate all affine lines in F2^4 (16 points).
    
    In F2^4, an affine line through points a and b is {a, b, a^b}.
    Each line has 3 points. There are C(16,2)/3 = 120 lines total,
    but each pair determines exactly one line, and each line has 3 pairs.
    Total distinct lines = C(16,2)/3 = 120.
    
    However, we can't use all 120 because two lines might share 2 points
    (which would violate linearity). In F2^4, two distinct affine lines
    intersect in at most 1 point, so ALL 120 lines form a linear system!
    """
    triples = []
    seen = set()
    for a in range(16):
        for b in range(a + 1, 16):
            c = a ^ b
            if c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                triples.append(list(t))
    return triples

def _greedy_improve(triples, n=16, deadline=None):
    """Greedy: try to add triples that reduce independence number."""
    triples = [list(t) for t in triples]
    
    # Build pair-to-triple-index map
    pair_to_idx = {}
    for i, t in enumerate(triples):
        for a in range(3):
            for b in range(a+1, 3):
                pair_to_idx[(t[a], t[b])] = i
    
    current_indep = _independence_number(triples, n)
    
    # Try adding triples greedily
    for _ in range(200):
        if deadline is not None and time.time() > deadline:
            break
        if current_indep <= 4:
            break
        
        # Find a triple that covers the most "independent" points
        best_triple = None
        best_gain = 0
        
        # Try all possible triples
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    t = (a, b, c)
                    # Check linearity
                    linear = True
                    for i in range(3):
                        for j in range(i+1, 3):
                            if (t[i], t[j]) in pair_to_idx:
                                linear = False
                                break
                        if not linear:
                            break
                    if not linear:
                        continue
                    
                    # Estimate gain: how many points in this triple are
                    # in the current maximum independent set?
                    # Approximate: just add it and check
                    new_triples = triples + [list(t)]
                    new_indep = _independence_number(new_triples, n)
                    gain = current_indep - new_indep
                    if gain > best_gain:
                        best_gain = gain
                        best_triple = t
        
        if best_triple is not None and best_gain > 0:
            t = best_triple
            triples.append(list(t))
            idx = len(triples) - 1
            for i in range(3):
                for j in range(i+1, 3):
                    pair_to_idx[(t[i], t[j])] = idx
            current_indep -= best_gain
        else:
            break
    
    return triples

def _swap_improve(triples, n=16, deadline=None, max_swaps=50):
    """Try swapping triples to reduce independence number."""
    triples = [list(t) for t in triples]
    
    pair_to_idx = {}
    for i, t in enumerate(triples):
        for a in range(3):
            for b in range(a+1, 3):
                pair_to_idx[(t[a], t[b])] = i
    
    current_indep = _independence_number(triples, n)
    
    for _ in range(max_swaps):
        if deadline is not None and time.time() > deadline:
            break
        if current_indep <= 4:
            break
        
        best_swap = None
        best_new_indep = current_indep
        
        # Try removing each triple and adding a new one
        for idx in range(len(triples)):
            old = triples[idx]
            # Remove old
            del pair_to_idx[(old[0], old[1])]
            del pair_to_idx[(old[0], old[2])]
            del pair_to_idx[(old[1], old[2])]
            
            # Try adding new triples
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        t = (a, b, c)
                        linear = True
                        for i in range(3):
                            for j in range(i+1, 3):
                                if (t[i], t[j]) in pair_to_idx:
                                    linear = False
                                    break
                            if not linear:
                                break
                        if not linear:
                            continue
                        
                        new_triples = triples[:idx] + [list(t)] + triples[idx+1:]
                        new_indep = _independence_number(new_triples, n)
                        if new_indep < best_new_indep:
                            best_new_indep = new_indep
                            best_swap = (idx, t)
            
            # Restore old
            pair_to_idx[(old[0], old[1])] = idx
            pair_to_idx[(old[0], old[2])] = idx
            pair_to_idx[(old[1], old[2])] = idx
        
        if best_swap is not None:
            idx, t = best_swap
            old = triples[idx]
            del pair_to_idx[(old[0], old[1])]
            del pair_to_idx[(old[0], old[2])]
            del pair_to_idx[(old[1], old[2])]
            triples[idx] = list(t)
            pair_to_idx[(t[0], t[1])] = idx
            pair_to_idx[(t[0], t[2])] = idx
            pair_to_idx[(t[1], t[2])] = idx
            current_indep = best_new_indep
        else:
            break
    
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20 second safety margin
    
    # Start with all affine lines in F2^4 - this is a complete linear system
    # with 120 triples covering all pairs exactly once
    all_lines = _affine_lines_f24()
    
    if not _is_linear(all_lines):
        # Fallback: should not happen
        all_lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    current_indep = _independence_number(all_lines, 16)
    best_triples = [list(t) for t in all_lines]
    best_indep = current_indep
    
    # The full affine line system already has a low independence number.
    # In F2^4, the maximum cap (set with no 3 collinear points) has size 6.
    # So independence number should be 6 or less.
    
    # Try to further reduce by adding non-affine triples if possible
    # But the affine lines already cover all pairs, so no more triples can be added.
    # Instead, try swaps to see if we can do better.
    
    improved = _swap_improve(best_triples, 16, deadline=deadline, max_swaps=20)
    imp_indep = _independence_number(improved, 16)
    if imp_indep < best_indep:
        best_indep = imp_indep
        best_triples = improved
    
    if best_triples is None:
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    
    return best_triples
# EVOLVE-BLOCK-END
