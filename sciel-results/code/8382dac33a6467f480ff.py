"""
Snake-in-the-box in Q_9: maximize induced path length.
Strategy: beam search with bitmask conflict tracking, multi-restart with different
bit-orderings and starting vertices, plus local improvement (2-opt style).
"""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N
TIME_LIMIT = 115  # safety margin before 120s kill

def _bit_count(x):
    return x.bit_count()

def _validate_path(path):
    """Validate that path is an induced path in Q_N."""
    if not path:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if _bit_count(path[i] ^ path[i+1]) != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if _bit_count(path[i] ^ path[j]) == 1:
                return False
    return True

def _greedy_extend(path, used, forbidden_mask, bit_order):
    """
    Extend path greedily. forbidden_mask[v] is a bitmask of vertices that would
    create a chord if v were added (i.e., vertices at distance 1 from v that are
    already in the path but not the last vertex).
    Returns (new_path, new_used, new_forbidden_mask) or None if stuck.
    """
    cur = path[-1]
    for b in bit_order:
        v = cur ^ (1 << b)
        if v in used:
            continue
        # Check if v would create a chord: any vertex in path[:-1] at distance 1 from v
        if forbidden_mask[v] & (1 << 0):  # placeholder, we'll compute properly
            pass
        # Actually check: for each p in path[:-1], if bit_count(v ^ p) == 1, chord
        # But that's O(path_length). Let's use a different approach.
        # Precompute: for each vertex v, which path vertices are at distance 1?
        # We maintain a 'neighbors_in_path' set.
        # Simpler: just check against the last few vertices and use a set of neighbors.
        # Let's just do the direct check but optimize with a neighbor set.
        # Actually, let's maintain: for each vertex in path, its neighbors in the hypercube.
        # A new vertex v creates a chord iff any of its hypercube-neighbors is in path[:-1].
        # So we need: is any neighbor of v in (used - {cur})?
        # Neighbors of v: v ^ (1<<b) for b in range(N)
        # Check if any of those are in used and != cur.
        ok = True
        for b2 in range(N):
            nb = v ^ (1 << b2)
            if nb in used and nb != cur:
                ok = False
                break
        if not ok:
            continue
        # Also need to check: does adding v create a chord with path[:-1]?
        # v is at distance 1 from cur (guaranteed). For p in path[:-1],
        # we need bit_count(v ^ p) >= 2.
        # bit_count(v ^ p) == 1 means v is a neighbor of p, which we just checked.
        # So the neighbor check above is sufficient!
        path.append(v)
        used.add(v)
        return path, used
    return None

def _greedy_full(start, bit_order):
    """Run full greedy from start with given bit order."""
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        extended = False
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check no chord: no neighbor of v (other than cur) is in used
            ok = True
            for b2 in range(N):
                nb = v ^ (1 << b2)
                if nb in used and nb != cur:
                    ok = False
                    break
            if not ok:
                continue
            path.append(v)
            used.add(v)
            extended = True
            break
        if not extended:
            break
    return path

def _beam_search(start, bit_order, beam_width=50, max_depth=250):
    """Beam search to find a longer snake."""
    # Each beam entry: (path, used_set, score)
    # We'll use a list of (path_tuple, used_set)
    beams = [([start], {start})]
    best_path = [start]
    
    for depth in range(max_depth):
        if time.time() > TIME_LIMIT - 5:
            break
        new_beams = []
        for path, used in beams:
            cur = path[-1]
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check chord condition
                ok = True
                for b2 in range(N):
                    nb = v ^ (1 << b2)
                    if nb in used and nb != cur:
                        ok = False
                        break
                if not ok:
                    continue
                new_path = path + [v]
                new_used = used | {v}
                new_beams.append((new_path, new_used))
        
        if not new_beams:
            break
        
        # Keep top beam_width by length (all same length at this depth, so by some heuristic)
        # Since all have same length, we need a different criterion.
        # Use: prefer paths that have more "open" options at the end.
        # Score: number of valid extensions from the last vertex.
        scored = []
        for path, used in new_beams:
            cur = path[-1]
            ext_count = 0
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for b2 in range(N):
                    nb = v ^ (1 << b2)
                    if nb in used and nb != cur:
                        ok = False
                        break
                if ok:
                    ext_count += 1
            scored.append((ext_count, path, used))
        
        scored.sort(key=lambda x: -x[0])
        beams = [(p, u) for _, p, u in scored[:beam_width]]
        
        if len(beams[0][0]) > len(best_path):
            best_path = beams[0][0]
    
    return best_path

def _local_improve(path):
    """Try to improve path by removing internal vertices and re-extending."""
    if not path:
        return path
    best = list(path)
    n = len(path)
    
    # Try removing each internal vertex and see if we can extend further
    for i in range(1, n - 1):
        if time.time() > TIME_LIMIT - 5:
            break
        # Remove vertex i
        new_path = path[:i] + path[i+1:]
        # Check if new_path is still valid
        if not _validate_path(new_path):
            continue
        # Try to extend from the end
        used = set(new_path)
        cur = new_path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for b2 in range(N):
                nb = v ^ (1 << b2)
                if nb in used and nb != cur:
                    ok = False
                    break
            if not ok:
                continue
            new_path.append(v)
            used.add(v)
            extended = True
            break
        if extended and len(new_path) > len(best):
            best = new_path
    
    return best

def construct():
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Generate multiple bit orderings for diversity
    bit_orders = []
    for _ in range(20):
        order = list(range(N))
        random.shuffle(order)
        bit_orders.append(order)
    
    # Also try natural order and reverse
    bit_orders.insert(0, list(range(N)))
    bit_orders.insert(0, list(range(N-1, -1, -1)))
    
    # Phase 1: Greedy with different starts and bit orders
    for order in bit_orders:
        if time.time() > TIME_LIMIT - 30:
            break
        for start in range(0, TOTAL_VERTICES, 17):  # sample starts
            if time.time() > TIME_LIMIT - 30:
                break
            path = _greedy_full(start, order)
            if len(path) > best_len:
                best_path = path
                best_len = len(path)
    
    # Phase 2: Beam search from best starting points
    if time.time() < TIME_LIMIT - 20:
        # Try beam search from the start that gave best greedy
        best_start = best_path[0] if best_path else 0
        for order in bit_orders[:5]:
            if time.time() > TIME_LIMIT - 15:
                break
            path = _beam_search(best_start, order, beam_width=30, max_depth=250)
            if len(path) > best_len:
                best_path = path
                best_len = len(path)
    
    # Phase 3: Local improvement
    if time.time() < TIME_LIMIT - 5:
        improved = _local_improve(best_path)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
    
    # Final validation
    if best_path and not _validate_path(best_path):
        # Fallback to a known valid greedy
        best_path = _greedy_full(0, list(range(N)))
    
    return best_path
