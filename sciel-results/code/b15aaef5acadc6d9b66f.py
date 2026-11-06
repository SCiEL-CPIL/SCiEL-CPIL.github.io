import time
import itertools
import random

# EVOLVE-BLOCK-START
def _paleyn_orientation(p):
    """
    Construct Paley tournament on p vertices.
    Returns list of bits for pairs (i, j) i < j.
    0 means i -> j, 1 means j -> i.
    """
    squares = set(x * x % p for x in range(1, p))
    orientation = []
    append = orientation.append
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            append(0 if diff in squares else 1)
    return orientation

def _in_bits_from_orientation(n, orientation):
    """Build in_bits list from orientation list."""
    in_bits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                in_bits[j] |= (1 << i)
            else:
                in_bits[i] |= (1 << j)
            idx += 1
    return in_bits

def _verify_4_dominator_exact(n, in_bits):
    """Exact verification: every 4-set has a common dominator."""
    ib = in_bits
    for a, b, c, d in itertools.combinations(range(n), 4):
        if (ib[a] & ib[b] & ib[c] & ib[d]) == 0:
            return False
    return True

def _verify_fast(n, in_bits):
    """Faster verification using pairwise intersections."""
    ib = in_bits
    # Precompute pair_int[a][b] = ib[a] & ib[b] for a < b
    pair_int = {}
    for a in range(n):
        for b in range(a + 1, n):
            pair_int[(a, b)] = ib[a] & ib[b]
    
    # For each 4-set (a,b,c,d), check if any vertex dominates all 4
    # A vertex v dominates {a,b,c,d} iff v is in ib[a] & ib[b] & ib[c] & ib[d]
    # i.e., (ib[a] & ib[b]) & (ib[c] & ib[d]) != 0
    # But we need the common dominator to be one of the 4 vertices? No, any vertex.
    # Actually, the common dominator can be any vertex in the tournament.
    # So we check if (ib[a] & ib[b] & ib[c] & ib[d]) != 0
    
    # Actually the direct check is the same. Let me think differently.
    # For each 4-set S, we need some v such that v->a, v->b, v->c, v->d.
    # That's ib[a] & ib[b] & ib[c] & ib[d] != 0.
    # The pairwise approach: (ib[a]&ib[b]) & (ib[c]&ib[d]) is the same as the 4-way AND.
    # So precomputing pair_int doesn't help much unless we cache.
    
    # Let's just do the direct check but with local variables for speed
    for a, b, c, d in itertools.combinations(range(n), 4):
        if (ib[a] & ib[b] & ib[c] & ib[d]) == 0:
            return False
    return True

def _try_prune(n, in_bits, deadline):
    """
    Try to prune vertices that are not needed.
    A vertex v is needed if there exists a 4-set S such that v is the UNIQUE dominator of S.
    If v is not unique dominator of any 4-set, we can try removing it.
    After removing, we need to re-verify.
    """
    # For each vertex v, determine if it's a unique dominator of some 4-set
    # v is dominator of S = {a,b,c,d} iff v in ib[a] & ib[b] & ib[c] & ib[d]
    # v is UNIQUE dominator of S iff for all w != v, w not in ib[a] & ib[b] & ib[c] & ib[d]
    
    # This is expensive. Let's use a simpler heuristic:
    # Try removing each vertex one at a time (in order of lowest out-degree first)
    # and check if the remaining tournament still satisfies the property.
    
    # Compute out-degree for each vertex
    out_deg = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (in_bits[i] >> j) & 1:
                out_deg[i] += 1
    
    # Sort vertices by out-degree ascending (try removing low-degree vertices first)
    order = sorted(range(n), key=lambda v: out_deg[v])
    
    kept = set(range(n))
    current_in_bits = in_bits[:]
    current_n = n
    
    for v in order:
        if time.time() > deadline:
            break
        if v not in kept:
            continue
        
        # Try removing v
        new_kept = kept - {v}
        
        # Build new in_bits for remaining vertices
        # Map old indices to new indices
        old_to_new = {}
        new_idx = 0
        for u in sorted(new_kept):
            old_to_new[u] = new_idx
            new_idx += 1
        
        new_n = len(new_kept)
        if new_n < 4:
            break
        
        new_in_bits = [0] * new_n
        for u in new_kept:
            for w in new_kept:
                if u != w and (in_bits[u] >> w) & 1:
                    new_in_bits[old_to_new[u]] |= (1 << old_to_new[w])
        
        # Verify
        if _verify_4_dominator_exact(new_n, new_in_bits):
            kept = new_kept
            current_in_bits = new_in_bits
            current_n = new_n
    
    return current_n, current_in_bits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Start with the best known Paley tournament
    # 67 is the smallest known, but let's also try 71, 79, 83
    candidates = [67, 71, 79, 83]
    
    best_n = 83
    best_orientation = _paleyn_orientation(83)
    best_in_bits = _in_bits_from_orientation(83, best_orientation)
    
    for p in candidates:
        if time.time() > deadline - 10:
            break
        
        orientation = _paleyn_orientation(p)
        in_bits = _in_bits_from_orientation(p, orientation)
        
        if not _verify_4_dominator_exact(p, in_bits):
            continue
        
        # Try to prune
        pruned_n, pruned_in_bits = _try_prune(p, in_bits, deadline - 5)
        
        if pruned_n < best_n:
            best_n = pruned_n
            best_in_bits = pruned_in_bits
            # Reconstruct orientation from in_bits
            best_orientation = []
            for i in range(pruned_n):
                for j in range(i + 1, pruned_n):
                    best_orientation.append(0 if (pruned_in_bits[i] >> j) & 1 else 1)
    
    return {'n': best_n, 'orientation': best_orientation}
# EVOLVE-BLOCK-END
