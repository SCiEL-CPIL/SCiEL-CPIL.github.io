import time
import random

N = 512

# Precompute all 4-APs
_all_4aps = []
_ap_by_element = [[] for _ in range(N + 1)]

for a in range(1, N - 2):
    for d in range(1, (N - a) // 3 + 1):
        e = a + 3 * d
        if e > N:
            break
        b = a + d
        c = a + 2 * d
        ap = (a, b, c, e)
        idx = len(_all_4aps)
        _all_4aps.append(ap)
        for elem in ap:
            _ap_by_element[elem].append(idx)

# For each element, precompute the other three elements in each AP it belongs to
_ap_partners = [None] * (N + 1)
for x in range(1, N + 1):
    partners = []
    for idx in _ap_by_element[x]:
        a, b, c, e = _all_4aps[idx]
        others = tuple(t for t in (a, b, c, e) if t != x)
        partners.append(others)
    _ap_partners[x] = tuple(partners)

def _can_add(x, bits):
    """Check if x can be added to set represented by bits without creating a 4-AP."""
    for others in _ap_partners[x]:
        o1, o2, o3 = others
        if ((bits >> o1) & 1) and ((bits >> o2) & 1) and ((bits >> o3) & 1):
            return False
    return True

def _is_valid(bits):
    """Check if the set is 4-AP-free."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for others in _ap_partners[x]:
            o1, o2, o3 = others
            if ((bits >> o1) & 1) and ((bits >> o2) & 1) and ((bits >> o3) & 1):
                return False
    return True

def _bits_to_set(bits):
    return [i for i in range(1, N + 1) if (bits >> i) & 1]

def _greedy_forward(bits_start=0):
    """Forward greedy: add elements in order 1..N if possible."""
    bits = bits_start
    for x in range(1, N + 1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits

def _greedy_reverse(bits_start=0):
    """Reverse greedy: add elements in order N..1 if possible."""
    bits = bits_start
    for x in range(N, 0, -1):
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits

def _greedy_random(bits_start=0, rng=None):
    """Random-order greedy."""
    if rng is None:
        rng = random.Random()
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = bits_start
    for x in order:
        if not ((bits >> x) & 1) and _can_add(x, bits):
            bits |= (1 << x)
    return bits

def _local_search(bits, deadline, rng):
    """
    Local search: try to add elements, and if blocked, try swaps.
    """
    current = bits
    current_size = bin(current).count('1')
    
    # Precompute list of all elements
    all_elems = list(range(1, N + 1))
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try to add any element
        # Use a random order for additions to avoid bias
        rng.shuffle(all_elems)
        for x in all_elems:
            if time.time() > deadline - 2:
                return current
            if not ((current >> x) & 1) and _can_add(x, current):
                current |= (1 << x)
                current_size += 1
                improved = True
        
        if not improved:
            # 2. Try swaps: for each x not in set, find minimal removal set
            # We try a limited number of swaps per pass to avoid being too slow
            for _attempt in range(50):
                if time.time() > deadline - 2:
                    return current
                
                x = rng.choice(all_elems)
                if (current >> x) & 1:
                    continue
                
                # Find all APs that would be completed
                blocking_aps = []
                for others in _ap_partners[x]:
                    o1, o2, o3 = others
                    if ((current >> o1) & 1) and ((current >> o2) & 1) and ((current >> o3) & 1):
                        blocking_aps.append((o1, o2, o3))
                
                if not blocking_aps:
                    # If no blocking APs, we should have added it in step 1
                    # This might happen if we didn't check it in step 1 due to random order
                    # But step 1 checks all. So this shouldn't happen if step 1 ran fully.
                    # However, if step 1 ran fully and didn't add it, it must be blocked.
                    continue
                
                # Collect all candidate removals
                candidates = set()
                for ap in blocking_aps:
                    for elem in ap:
                        candidates.add(elem)
                
                if not candidates:
                    continue
                
                # Try each single removal
                found_swap = False
                for rem in candidates:
                    if time.time() > deadline - 2:
                        return current
                    trial = current & ~(1 << rem)
                    if _can_add(x, trial):
                        trial |= (1 << x)
                        # Try to re-add some elements (greedy polish)
                        # This is expensive, so we do it only if we have time
                        if time.time() < deadline - 5:
                            trial = _greedy_forward(trial)
                        trial_size = bin(trial).count('1')
                        if trial_size > current_size:
                            current = trial
                            current_size = trial_size
                            improved = True
                            found_swap = True
                            break
                
                if found_swap:
                    break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Multiple seed constructions
    fwd = _greedy_forward()
    rev = _greedy_reverse()
    
    seeds = [fwd, rev]
    seed_sizes = [bin(s).count('1') for s in seeds]
    
    best = seeds[seed_sizes.index(max(seed_sizes))]
    best_size = max(seed_sizes)
    
    rng = random.Random(12345)
    
    # 2. Local search on best seed
    if time.time() < deadline - 30:
        improved = _local_search(best, deadline - 10, rng)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 3. Randomized restarts
    restart_count = 0
    while time.time() < deadline - 20 and restart_count < 50:
        restart_count += 1
        
        # Different strategies
        r = rng.random()
        if r < 0.5:
            # Random greedy
            bits = _greedy_random(0, rng)
        elif r < 0.7:
            # Start from best, perturb: remove a random element, then re-greedy
            bits = best
            if best_size > 1:
                elems = _bits_to_set(bits)
                if elems:
                    rem = rng.choice(elems)
                    bits = bits & ~(1 << rem)
                    bits = _greedy_forward(bits)
        else:
            # Reverse greedy from scratch
            bits = _greedy_reverse()
            bits = _greedy_forward(bits)
        
        size = bin(bits).count('1')
        if size > best_size:
            best = bits
            best_size = size
        
        # Local search on this result
        if time.time() < deadline - 15:
            improved = _local_search(bits, deadline - 10, rng)
            imp_size = bin(improved).count('1')
            if imp_size > best_size:
                best = improved
                best_size = imp_size
    
    # 4. Final local search on best
    if time.time() < deadline - 5:
        improved = _local_search(best, deadline, rng)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best = improved
            best_size = imp_size
    
    # 5. Verify
    if not _is_valid(best):
        # Fallback to a known valid set (Greedy Forward)
        best = _greedy_forward()
    
    return sorted(_bits_to_set(best))

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
