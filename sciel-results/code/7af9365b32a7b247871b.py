<<<<<<< SEARCH
# Precompute LCM table for all pairs in [1, 128]
_LCM_TABLE = {}
def _lcm(a, b):
    key = (min(a,b), max(a,b))
    if key not in _LCM_TABLE:
        _LCM_TABLE[key] = a * b // gcd(a, b)
    return _LCM_TABLE[key]

def _is_bad_triple(a, b, c):
    """Check if {a, b, c} forms a bad triple (equal pairwise LCMs)."""
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _conflicts_with_set(x, S):
    """Check if adding x to S creates any bad triple. Uses precomputed LCM."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        la = _lcm(x, a)
        for j in range(i + 1, n):
            b = S_list[j]
            if la == _lcm(x, b) == _lcm(a, b):
                return True
    return False

def _greedy_with_order(order):
    """Add elements from order to a new set if they don't conflict."""
    S = set()
    for x in order:
        if not _conflicts_with_set(x, S):
            S.add(x)
    return S

def _local_search(S, deadline):
    """
    Try to improve S by:
    1. Adding any element that doesn't conflict
    2. 1-swaps: remove one, add one
    3. 2-swaps: remove two, add two
    Returns the improved set.
    """
    S = set(S)
    full_set = set(range(1, 129))
    S_list = sorted(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflicts_with_set(x, S):
                S.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        found_swap = False
        for a in list(S):
            S.remove(a)
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflicts_with_set(x, S):
                    S.add(x)
                    found_swap = True
                    break
            if not found_swap:
                S.add(a)
        
        if found_swap:
            continue
        
        # Phase 3: Try 2-swaps (remove two, add two)
        S_list = sorted(S)
        n = len(S_list)
        found_2swap = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S_list[i], S_list[j]
                S.remove(a)
                S.remove(b)
                for x in range(1, 129):
                    if x in S:
                        continue
                    if _conflicts_with_set(x, S):
                        continue
                    S.add(x)
                    for y in range(1, 129):
                        if y in S or y == x:
                            continue
                        if not _conflicts_with_set(y, S):
                            S.add(y)
                            found_2swap = True
                            break
                    if not found_2swap:
                        S.remove(x)
                        S.add(a)
                        S.add(b)
                        break
                if found_2swap:
                    break
            if found_2swap:
                break
        
        if not found_2swap:
            break
    
    return S
=======
# Precompute LCM table for all pairs in [1, 128]
_LCM_TABLE = {}
def _lcm(a, b):
    key = (min(a,b), max(a,b))
    if key not in _LCM_TABLE:
        _LCM_TABLE[key] = a * b // gcd(a, b)
    return _LCM_TABLE[key]

# Precompute conflict bitsets: for each x, which pairs (a,b) with a<b form a bad triple with x
# conflict_bits[x] is a set of (a,b) pairs where lcm(x,a)==lcm(x,b)==lcm(a,b)
_CONFLICT_BITS = {}
def _init_conflict_bits():
    """Precompute for each x in [1,128], the set of pairs (a,b) that conflict with x."""
    for x in range(1, 129):
        conflicts = set()
        for a in range(1, 129):
            if a == x:
                continue
            for b in range(a+1, 129):
                if b == x:
                    continue
                if _lcm(x, a) == _lcm(x, b) == _lcm(a, b):
                    conflicts.add((a, b))
        _CONFLICT_BITS[x] = conflicts

_init_conflict_bits()

def _can_add(x, S_bits):
    """Check if x can be added to set represented by S_bits (bitmask of elements in S).
    Returns True if no bad triple is formed."""
    # For each pair (a,b) in S, check if (a,b) conflicts with x
    # S_bits is an integer bitmask where bit i represents element i+1
    conflicts = _CONFLICT_BITS[x]
    for (a, b) in conflicts:
        # Check if both a and b are in S
        if (S_bits >> a) & 1 and (S_bits >> b) & 1:
            return False
    return True

def _set_to_bits(S):
    """Convert a set of elements to a bitmask."""
    bits = 0
    for x in S:
        bits |= (1 << x)
    return bits

def _bits_to_set(bits):
    """Convert a bitmask to a set of elements."""
    return {x for x in range(1, 129) if (bits >> x) & 1}

def _greedy_with_order(order):
    """Add elements from order to a new set if they don't conflict."""
    S_bits = 0
    for x in order:
        if not (S_bits >> x) & 1 and _can_add(x, S_bits):
            S_bits |= (1 << x)
    return _bits_to_set(S_bits)

def _local_search(S, deadline):
    """
    Try to improve S by:
    1. Adding any element that doesn't conflict
    2. 1-swaps: remove one, add one
    3. 2-swaps: remove two, add two
    Returns the improved set.
    """
    S_bits = _set_to_bits(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element not in S
        for x in range(1, 129):
            if (S_bits >> x) & 1:
                continue
            if _can_add(x, S_bits):
                S_bits |= (1 << x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        found_swap = False
        S_list = sorted(_bits_to_set(S_bits))
        for a in S_list:
            S_bits_without_a = S_bits & ~(1 << a)
            for x in range(1, 129):
                if (S_bits_without_a >> x) & 1:
                    continue
                if _can_add(x, S_bits_without_a):
                    S_bits = S_bits_without_a | (1 << x)
                    found_swap = True
                    break
            if found_swap:
                break
        
        if found_swap:
            continue
        
        # Phase 3: Try 2-swaps (remove two, add two)
        S_list = sorted(_bits_to_set(S_bits))
        n = len(S_list)
        found_2swap = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S_list[i], S_list[j]
                S_bits_without_ab = S_bits & ~(1 << a) & ~(1 << b)
                for x in range(1, 129):
                    if (S_bits_without_ab >> x) & 1:
                        continue
                    if not _can_add(x, S_bits_without_ab):
                        continue
                    S_bits_with_x = S_bits_without_ab | (1 << x)
                    for y in range(1, 129):
                        if (S_bits_with_x >> y) & 1:
                            continue
                        if not _can_add(y, S_bits_with_x):
                            continue
                        S_bits = S_bits_with_x | (1 << y)
                        found_2swap = True
                        break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            if found_2swap:
                break
        
        if not found_2swap:
            break
    
    return _bits_to_set(S_bits)
>>>>>>> REPLACE

