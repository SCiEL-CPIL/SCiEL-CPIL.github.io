"""Combined program: Paley 67 + vertex removal + fast bitmask local search."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _make_paley(p):
    """Return orientation bits for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    return [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]

def _build_out_masks(orientation, n):
    """Build out-neighborhood bitmask list from orientation."""
    out_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                out_masks[i] |= (1 << j)
            else:
                out_masks[j] |= (1 << i)
    return out_masks

def _count_violations_fast(out_masks, n):
    """
    Count 4-sets without a common dominator using bitmask subset checks.
    A 4-set {a,b,c,d} is valid if some v has S ⊆ out_masks[v].
    Since v cannot dominate itself, if S ⊆ out_masks[v] then v ∉ S.
    """
    violations = 0
    # Precompute for speed: iterate over all 4-sets
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    S = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    # Check if any vertex dominates S
                    found = False
                    for v in range(n):
                        if (out_masks[v] & S) == S:
                            found = True
                            break
                    if not found:
                        violations += 1
    return violations

def _is_valid_fast(out_masks, n):
    """Check validity using bitmask subset test."""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    S = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    found = False
                    for v in range(n):
                        if (out_masks[v] & S) == S:
                            found = True
                            break
                    if not found:
                        return False
    return True

def _remove_vertex_from_masks(out_masks, n, v):
    """
    Remove vertex v from the tournament represented by out_masks.
    Returns new out_masks list for n-1 vertices with indices remapped.
    """
    new_n = n - 1
    # Map old index to new index: indices < v stay same, indices > v decrease by 1
    new_masks = [0] * new_n
    for i in range(n):
        if i == v:
            continue
        new_i = i if i < v else i - 1
        mask = out_masks[i]
        # Remove bit v
        mask &= ~(1 << v)
        # Remap bits: for each bit position j in mask, if j > v, shift down by 1
        # We need to reconstruct the mask with remapped indices
        new_mask = 0
        for j in range(n):
            if j == v:
                continue
            if mask & (1 << j):
                new_j = j if j < v else j - 1
                new_mask |= (1 << new_j)
        new_masks[new_i] = new_mask
    return new_masks

def _remove_vertex_from_orientation(orientation, n, v):
    """Remove vertex v from orientation list, return new orientation and new n."""
    new_n = n - 1
    new_orientation = []
    idx = 0
    for i in range(n):
        if i == v:
            continue
        for j in range(i + 1, n):
            if j == v:
                continue
            new_orientation.append(orientation[idx])
            idx += 1
    return new_orientation, new_n

def _flip_pair(orientation, n, i, j):
    """Flip the edge between i and j. Returns new orientation list."""
    # Compute index of pair (i,j) in lexicographic order
    # Index = sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
    # = i*(n-1) - i*(i-1)//2 + (j - i - 1)
    # Let me verify: for i=0: indices 0..n-2 (j=1..n-1)
    # for i=1: indices (n-1)..(n-1)+(n-2-1) = n-1..2n-4
    # General: offset for row i = i*(n-1) - i*(i-1)//2
    offset = i * (n - 1) - i * (i - 1) // 2
    pos = offset + (j - i - 1)
    new_orient = orientation[:]
    new_orient[pos] = 1 - new_orient[pos]
    return new_orient

def _local_search(orientation, n, deadline):
    """
    Local search: flip edges to reduce violation count.
    Returns (orientation, n) of best found.
    """
    out_masks = _build_out_masks(orientation, n)
    current_violations = _count_violations_fast(out_masks, n)
    best_orient = orientation[:]
    best_violations = current_violations
    
    # If already valid, we're done
    if current_violations == 0:
        return orientation, n
    
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Try random edge flips
        improved = False
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            new_orient = _flip_pair(orientation, n, i, j)
            new_masks = _build_out_masks(new_orient, n)
            new_violations = _count_violations_fast(new_masks, n)
            
            if new_violations < current_violations:
                orientation = new_orient
                out_masks = new_masks
                current_violations = new_violations
                improved = True
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_orient = orientation[:]
                if current_violations == 0:
                    return orientation, n
                break
        if not improved:
            # Accept some worsening moves to escape local minima
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                new_orient = _flip_pair(orientation, n, i, j)
                new_masks = _build_out_masks(new_orient, n)
                new_violations = _count_violations_fast(new_masks, n)
                # Accept if not too bad
                if new_violations <= current_violations + 5:
                    orientation = new_orient
                    out_masks = new_masks
                    current_violations = new_violations
                    break
    
    return best_orient, n

def construct():
    """
    Construct a tournament where every 4-set has a common dominator.
    Strategy: 
    1. Try Paley tournaments at prime sizes 67, 71, 79, 83.
    2. For the smallest valid one, attempt vertex removal.
    3. Use bitmask local search to repair violations after removal.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try Paley tournaments in order of increasing n
    for p in [67, 71, 79, 83]:
        if time.time() > deadline:
            break
        orientation = _make_paley(p)
        n = p
        out_masks = _build_out_masks(orientation, n)
        
        if _is_valid_fast(out_masks, n):
            # Valid! Now try to remove vertices to minimize n
            # Greedy removal with local search repair
            for removal_attempt in range(10):
                if time.time() > deadline - 10:
                    break
                removed_this_round = False
                for v in range(n):
                    if time.time() > deadline - 10:
                        break
                    # Try removing vertex v
                    new_masks = _remove_vertex_from_masks(out_masks, n, v)
                    new_n = n - 1
                    if new_n < 5:
                        continue
                    
                    # Count violations in the reduced tournament
                    new_violations = _count_violations_fast(new_masks, new_n)
                    
                    if new_violations == 0:
                        # Clean removal
                        orientation, n = _remove_vertex_from_orientation(orientation, n, v)
                        out_masks = new_masks
                        removed_this_round = True
                        break
                    elif new_violations <= 3:
                        # Near-valid: try to repair with local search
                        new_orient, _ = _remove_vertex_from_orientation(orientation, n, v)
                        repaired_orient, repaired_n = _local_search(new_orient, new_n, deadline)
                        if repaired_n == new_n:
                            rep_masks = _build_out_masks(repaired_orient, repaired_n)
                            rep_violations = _count_violations_fast(rep_masks, repaired_n)
                            if rep_violations == 0:
                                orientation = repaired_orient
                                n = repaired_n
                                out_masks = rep_masks
                                removed_this_round = True
                                break
                if not removed_this_round:
                    break
            
            # Final verification
            final_masks = _build_out_masks(orientation, n)
            if _is_valid_fast(final_masks, n):
                return {'n': n, 'orientation': orientation}
            else:
                # Fallback to the original valid Paley
                return {'n': p, 'orientation': _make_paley(p)}
    
    # Fallback: return Paley 83
    return {'n': 83, 'orientation': _make_paley(83)}

# EVOLVE-BLOCK-END
