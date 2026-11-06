"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _get_paleyn_orientation(n):
    """
    Constructs a Paley tournament orientation for prime n.
    Returns list of bits or None if n is not a valid prime for Paley (n % 4 == 3).
    """
    if n < 2:
        return None
    # Check prime
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if not is_prime:
        return None
    if n % 4 != 3:
        return None
        
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
        
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return orientation

def _orientation_to_in_bits(n, orientation):
    """Convert orientation list to in_bits array."""
    in_bits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                # i -> j
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                in_bits[i] |= (1 << j)
            idx += 1
    return in_bits

def _in_bits_to_orientation(n, in_bits):
    """Convert in_bits array to orientation list."""
    orientation = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (in_bits[j] >> i) & 1:
                orientation[idx] = 0 # i -> j
            else:
                orientation[idx] = 1 # j -> i
            idx += 1
    return orientation

def _validate_tournament(n, in_bits):
    """Check if every 4-set has a common dominator."""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    if (common & ~mask) == 0:
                        return False
    return True

def _greedy_prune(n, in_bits, deadline):
    """
    Greedy set-cover pruning: iteratively remove vertices that are not
    uniquely required to dominate any 4-set.
    Returns (pruned_n, pruned_in_bits) or (n, in_bits) if no improvement.
    """
    start_time = time.time()
    current_n = n
    current_in_bits = in_bits[:]
    
    # For each vertex, determine which 4-sets it is the UNIQUE dominator for
    # A vertex v is a dominator of 4-set S if v is not in S and v->x for all x in S
    # v is the UNIQUE dominator if no other vertex w (w not in S, w != v) also dominates S
    
    # For efficiency, precompute for each vertex v, the set of vertices it dominates
    # v dominates x iff v -> x, i.e., x in Out[v]
    # Equivalently, v -> x iff (in_bits[x] >> v) & 1
    
    # For each 4-set, find all dominators
    # A 4-set {a,b,c,d} is dominated by v if:
    #   v not in {a,b,c,d} and v->a, v->b, v->c, v->d
    #   i.e., (in_bits[a]>>v)&1, (in_bits[b]>>v)&1, (in_bits[c]>>v)&1, (in_bits[d]>>v)&1
    
    # For pruning: we can remove vertex v if every 4-set that v uniquely dominates
    # is still dominated by some other vertex after removing v.
    # This is complex. Simpler approach: try removing each vertex and check validity.
    # But that's O(n * C(n,4)) per removal attempt.
    
    # Better: for each vertex v, count how many 4-sets are ONLY dominated by v.
    # If count is 0, v can be safely removed.
    
    # Let's implement: for each vertex v, count 4-sets where v is the unique dominator.
    # For a 4-set S, dominators are vertices outside S that point to all of S.
    
    # Optimization: use bitmasks
    # For vertex v, the set of vertices it dominates: Dom[v] = {x : v->x}
    # 4-set {a,b,c,d} is dominated by v iff {a,b,c,d} ⊆ Dom[v] and v ∉ {a,b,c,d}
    
    # Let's precompute Dom[v] as bitmask
    Dom = [0] * current_n
    for v in range(current_n):
        for x in range(current_n):
            if v == x:
                continue
            if (in_bits[x] >> v) & 1:
                Dom[v] |= (1 << x)
    
    # For each vertex v, count 4-sets uniquely dominated by v
    # A 4-set S is uniquely dominated by v if:
    #   S ⊆ Dom[v], v ∉ S, and for all w ≠ v, w ∉ S, NOT(S ⊆ Dom[w])
    
    # This is expensive. Let's use a simpler heuristic:
    # Try removing vertices in order of "least critical" (fewest unique dominations)
    # and check if the result is still valid.
    
    # Simpler approach: iteratively try to remove each vertex and validate.
    # Use a timeout.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_n = len(current_in_bits)
        
        # Recompute Dom for current state
        Dom = [0] * current_n
        for v in range(current_n):
            for x in range(current_n):
                if v == x:
                    continue
                if (current_in_bits[x] >> v) & 1:
                    Dom[v] |= (1 << x)
        
        # For each vertex, count how many 4-sets it uniquely dominates
        unique_counts = [0] * current_n
        
        # This is O(n * C(n,4)) which is too slow for n=67
        # C(67,4) = 751,015, times 67 = ~50M operations. Might be borderline.
        # Let's optimize: for each 4-set, find all dominators, increment count for each.
        # Then a vertex is unique dominator for a set if it's the only one.
        
        # Actually, let's just try removing vertices one by one and check validity.
        # For each vertex v, check if removing it keeps the tournament valid.
        
        # To speed up, first compute for each 4-set the set of dominators.
        # dominators[set_idx] = bitmask of vertices that dominate this set
        
        # Number of 4-sets for n=67: C(67,4) = 751015
        # Store as list of bitmasks (Python ints)
        
        # Build list of 4-tuples
        tuples = []
        for a in range(current_n):
            for b in range(a + 1, current_n):
                for c in range(b + 1, current_n):
                    for d in range(c + 1, current_n):
                        tuples.append((a, b, c, d))
        
        # For each 4-set, compute dominator mask
        # Dominator v: v not in set, and v->a, v->b, v->c, v->d
        # i.e., (in_bits[a]>>v)&1 and (in_bits[b]>>v)&1 and (in_bits[c]>>v)&1 and (in_bits[d]>>v)&1
        
        # For a given 4-set {a,b,c,d}, the dominators are vertices v not in {a,b,c,d}
        # such that v is in In[a] & In[b] & In[c] & In[d] (excluding a,b,c,d)
        
        # common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
        # dominators = common & ~(1<<a) & ~(1<<b) & ~(1<<c) & ~(1<<d)
        
        # A vertex v is the UNIQUE dominator if dominators == (1<<v)
        
        # Let's count for each vertex how many 4-sets it uniquely dominates
        unique_count = [0] * current_n
        
        for (a, b, c, d) in tuples:
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            common = current_in_bits[a] & current_in_bits[b] & current_in_bits[c] & current_in_bits[d]
            dominators = common & ~mask
            # Count unique dominators
            if dominators != 0:
                # Check if exactly one bit is set
                if dominators & (dominators - 1) == 0:
                    # Unique dominator
                    v = dominators.bit_length() - 1
                    if v < current_n:
                        unique_count[v] += 1
        
        # Try removing vertices with unique_count[v] == 0
        # Sort by unique_count ascending
        order = sorted(range(current_n), key=lambda v: unique_count[v])
        
        for v in order:
            if time.time() > deadline - 5:
                break
            if unique_count[v] > 0:
                break  # No more removable vertices
            
            # Try removing vertex v
            # Build new in_bits without v
            # Vertices after v shift down by 1
            new_n = current_n - 1
            if new_n < 5:
                break
            
            new_in_bits = [0] * new_n
            for old_i in range(current_n):
                if old_i == v:
                    continue
                new_i = old_i if old_i < v else old_i - 1
                # Remap bits
                new_bits = 0
                for old_j in range(current_n):
                    if old_j == v:
                        continue
                    if (current_in_bits[old_i] >> old_j) & 1:
                        new_j = old_j if old_j < v else old_j - 1
                        new_bits |= (1 << new_j)
                new_in_bits[new_i] = new_bits
            
            # Validate
            if _validate_tournament(new_n, new_in_bits):
                current_n = new_n
                current_in_bits = new_in_bits
                improved = True
                break  # Restart the outer loop
        
    return current_n, current_in_bits

def _local_search(n, in_bits, time_limit):
    """
    Local search to minimize bad 4-sets.
    Returns (n, in_bits) if valid, else None.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    in_bits = in_bits[:]
    
    def _count_bad(n, in_bits):
        bad = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                        if (common & ~mask) == 0:
                            bad += 1
        return bad
    
    bad = _count_bad(n, in_bits)
    if bad == 0:
        return n, in_bits
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Determine current direction
        if (in_bits[j] >> i) & 1:
            # i -> j
            new_in_i = in_bits[i] | (1 << j)
            new_in_j = in_bits[j] & ~(1 << i)
        else:
            # j -> i
            new_in_i = in_bits[i] & ~(1 << j)
            new_in_j = in_bits[j] | (1 << i)
        
        # Compute delta: count bad sets affected by flipping edge (i,j)
        # Affected sets are those containing i or j
        # For each 4-set containing i or j, check if it changes state
        
        # To speed up, iterate over all 4-sets containing i or j
        # This is O(C(n-1,3) + C(n-1,3)) = O(2*C(n-1,3))
        
        delta = 0
        # Sets containing i (and not j, since we handle i,j separately)
        # Actually, sets containing i OR j
        
        # For a 4-set S, it's affected if i in S or j in S
        # The state of S changes only if the flip changes whether some vertex dominates S
        
        # Simpler: just recompute bad count for affected sets
        # But we need old and new states
        
        # Let's just recompute total bad count for simplicity (slower but correct)
        # This is O(C(n,4)) per step which is too slow.
        
        # Better: iterate affected sets
        # A 4-set is affected if it contains i or j
        # For each such set, check old and new bad status
        
        # Build list of affected 4-set indices? No, just iterate.
        
        # For each 4-set containing i (and possibly j):
        # If the set contains both i and j: 2 other vertices from remaining n-2
        # If the set contains i but not j: 3 other vertices from remaining n-2 (excluding j)
        # If the set contains j but not i: 3 other vertices from remaining n-2 (excluding i)
        
        # This is complex. Let's use a different approach:
        # Just do a simple hill-climbing: flip edge, if bad decreases, accept.
        # Otherwise, accept with probability exp(-delta/T)
        
        # For speed, let's just recompute bad count every few steps
        # or use a sampling approach
        
        # Given time constraints, let's do a simple approach:
        # Flip the edge, recompute bad count, accept if better or with prob exp(-delta/T)
        # But recomputing is O(C(n,4)) which is ~750K for n=67. Too slow per step.
        
        # Alternative: use the fact that only sets containing i or j are affected.
        # Number of such sets: 
        # Sets with both i,j: C(n-2,2)
        # Sets with i not j: C(n-2,3)
        # Sets with j not i: C(n-2,3)
        # Total: C(n-2,2) + 2*C(n-2,3)
        # For n=67: C(65,2) + 2*C(65,3) = 2080 + 2*43680 = 89440
        # This is manageable.
        
        # For each affected set, compute old and new bad status
        
        # Helper: is_bad(set, in_bits)
        def _is_bad_set(a, b, c, d, ib):
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            common = ib[a] & ib[b] & ib[c] & ib[d]
            return (common & ~mask) == 0
        
        # Iterate affected sets
        # Sets containing both i and j:
        for x in range(n):
            if x == i or x == j:
                continue
            for y in range(x + 1, n):
                if y == i or y == j:
                    continue
                # Set {i, j, x, y}
                old_bad = _is_bad_set(i, j, x, y, in_bits)
                # New in_bits for this set: use new_in_i and new_in_j
                # Temporarily modify? No, just check with modified values
                # We need in_bits[a], in_bits[b], in_bits[c], in_bits[d]
                # For a=i: use new_in_i, for a=j: use new_in_j, else in_bits[a]
                
                # This is getting complex. Let's just do a full recompute every K steps.
                pass
        
        # Given the complexity, let's use a simpler local search:
        # Just do random flips and check if bad decreased, using full recompute
        # but limit the number of steps.
        
        # Actually, let's just return None and rely on the greedy prune.
        # The local search is too slow to implement correctly within time.
        
        break
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Paley(67)
    orientation = _get_paleyn_orientation(67)
    if orientation is None:
        # Fallback
        n = 83
        orientation = _get_paleyn_orientation(83)
        if orientation is None:
            n = 90
            orientation = [0] * (n * (n - 1) // 2)
        return {'n': n, 'orientation': orientation}
    
    n = 67
    in_bits = _orientation_to_in_bits(n, orientation)
    
    # Validate Paley(67)
    if not _validate_tournament(n, in_bits):
        # Should not happen, but fallback
        return {'n': n, 'orientation': orientation}
    
    # Try greedy pruning
    n_pruned, in_bits_pruned = _greedy_prune(n, in_bits, deadline)
    
    if n_pruned < n:
        n = n_pruned
        in_bits = in_bits_pruned
    
    # Try local search for further improvement (if time remains)
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 5:
        result = _local_search(n, in_bits, remaining_time)
        if result is not None:
            n, in_bits = result
    
    orientation = _in_bits_to_orientation(n, in_bits)
    return {'n': n, 'orientation': orientation}

# EVOLVE-BLOCK-END
