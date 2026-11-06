"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
TOTAL_VERTS = 1 << N  # 512

# Precompute adjacency structure for speed
# neighbors[v] = list of 9 neighbors of vertex v
# We also precompute bit_count differences for all pairs? That's 512^2 = 262144, feasible.
# But we can compute on the fly efficiently.

def _precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    return neighbors

_NEIGHBORS = _precompute_neighbors()

def _is_valid_extension(v, path, used):
    """Check if vertex v can be appended to path.
    v must be adjacent to path[-1] and not adjacent to any path[i] for i < len(path)-2.
    Also v must not be in used.
    """
    if v in used:
        return False
    # Check non-consecutive adjacency: v must not differ by exactly 1 bit from any path[i] where i <= len(path)-3
    # Actually, the condition is: non-consecutive vertices differ in at least two bits.
    # Consecutive vertices (path[-2], path[-1]) are already handled by construction.
    # New vertex v is consecutive to path[-1]. So v must NOT be consecutive to path[0..len(path)-2].
    # i.e., for all i in [0, len(path)-2], popcount(v ^ path[i]) != 1.
    # Note: v is adjacent to path[-1] by construction (we only try neighbors).
    # But we also need to ensure v is not adjacent to path[-2]? No, path[-2] and path[-1] are consecutive,
    # but v and path[-2] are non-consecutive (distance 2 in path), so they must differ in >= 2 bits.
    # So yes, check all i from 0 to len(path)-2.
    
    # Optimization: we can maintain a set of "forbidden" vertices that are adjacent to any path vertex.
    # But for now, let's just check.
    
    limit = len(path) - 1  # we check indices 0..limit-1, i.e., up to path[-2]
    for i in range(limit):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True

def _greedy_seed():
    """Original greedy approach to get a baseline."""
    path, used = [0], {0}
    while True:
        cur = path[-1]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _is_valid_extension(v, path, used):
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path

def _random_greedy(seed_val=42):
    """Greedy with random bit ordering."""
    rng = random.Random(seed_val)
    path, used = [0], {0}
    while True:
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        found = False
        for b in bits:
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _is_valid_extension(v, path, used):
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path

def _try_insert(path, used, v):
    """Try to insert v at some position in the path. Returns new path or None."""
    # This is expensive, so we only use it sparingly.
    # Check if v can be inserted between path[i] and path[i+1]
    # Conditions:
    # 1. v adjacent to path[i] and path[i+1]
    # 2. v not adjacent to any path[j] where j is not i or i+1
    # 3. path[i] and path[i+1] are not adjacent to v's other neighbors... actually just the above.
    
    L = len(path)
    for i in range(L - 1):
        a, b = path[i], path[i+1]
        if (v ^ a).bit_count() != 1 or (v ^ b).bit_count() != 1:
            continue
        # Check v against all other path vertices
        ok = True
        for j in range(L):
            if j == i or j == i+1:
                continue
            if (v ^ path[j]).bit_count() == 1:
                ok = False
                break
        if not ok:
            continue
        # Also need to check that a and b are not adjacent to v in a way that creates issues?
        # No, the induced path condition is just pairwise.
        # Create new path
        new_path = path[:i+1] + [v] + path[i+1:]
        return new_path
    return None

def construct():
    """
    Construct the longest induced path in Q_9.
    Strategy:
    1. Start with a greedy seed.
    2. Use local search: try to extend, insert, or rearrange.
    3. Time-limited search.
    """
    deadline = time.time() + 115  # 115s to be safe
    
    # Get initial seed
    best_path = _greedy_seed()
    best_len = len(best_path)
    
    # Try a few random greedy seeds
    for s in range(1, 20):
        if time.time() > deadline - 5:
            break
        p = _random_greedy(s)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
    
    # Local search: try to extend from both ends, and insert vertices
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        path = best_path
        used = set(path)
        
        # Try to extend from the end
        cur = path[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _is_valid_extension(v, path, used):
                path = path + [v]
                used.add(v)
                improved = True
                break
        
        # Try to extend from the beginning
        if not improved:
            cur = path[0]
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check if v can be prepended: v adjacent to path[0], not adjacent to path[2:]
                ok = True
                for i in range(2, len(path)):
                    if (v ^ path[i]).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    path = [v] + path
                    used.add(v)
                    improved = True
                    break
        
        # Try to insert a vertex
        if not improved:
            # Find candidate vertices adjacent to some edge in the path
            candidates = set()
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                # Vertices adjacent to both a and b (other than a and b)
                # In a hypercube, if a and b differ in bit k, then vertices adjacent to both
                # are those that differ from a in bit k and one other bit, and from b in one other bit.
                # Actually, the common neighbors of a and b in Q_N: 
                # a and b differ in exactly one bit (since they're consecutive).
                # A vertex v is adjacent to both if v differs from a in one bit and from b in one bit.
                # Since a and b differ in bit k, v must differ from a in bit k (to be adjacent to b? no)
                # Let's think: a = ...0..., b = ...1... (differ in bit k)
                # v adjacent to a: v = a ^ (1<<j) for some j.
                # v adjacent to b: v = b ^ (1<<m) for some m.
                # So a ^ (1<<j) = b ^ (1<<m) => a ^ b = (1<<j) ^ (1<<m).
                # Since a ^ b = (1<<k), we need (1<<j) ^ (1<<m) = (1<<k).
                # This implies {j, m} = {k, l} for some l != k.
                # So v = a ^ (1<<k) ^ (1<<l) = b ^ (1<<l).
                # So for each l != k, v = b ^ (1<<l) is a common neighbor.
                
                diff_bit = (a ^ b).bit_length() - 1  # the bit that differs
                for l in range(N):
                    if l == diff_bit:
                        continue
                    v = b ^ (1 << l)
                    if v not in used:
                        candidates.add(v)
            
            # Try to insert each candidate
            for v in list(candidates):
                if time.time() > deadline - 1:
                    break
                new_path = _try_insert(path, used, v)
                if new_path is not None:
                    path = new_path
                    used = set(path)
                    improved = True
                    break
        
        if improved:
            best_path = path
            best_len = len(path)
    
    return best_path

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the block structure)
# Actually, the task says to keep the function names and signatures. The original had construct() in the EVOLVE-BLOCK.
# Let me restructure to put everything in the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
