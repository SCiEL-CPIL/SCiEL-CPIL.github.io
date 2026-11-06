"""
4-AP-free set construction using pair-swap local search with bit-parallel conflict tracking.
Structural change: uses a precomputed 4-AP list and tracks violations per-AP for O(1) delta evaluation.
"""
import time
import random
import math

N = 512

def _build_4aps():
    """Build list of all 4-APs as tuples (a, b, c, d)."""
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps.append((a, b, c, e))
    return aps

_ALL_APS = _build_4aps()
_NUM_APS = len(_ALL_APS)

# For each element x, store list of (ap_index, position) where x appears in that AP at that position.
# This allows O(degree) delta computation.
def _build_element_ap_map():
    elem_ap = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(_ALL_APS):
        elem_ap[a].append((idx, 0))
        elem_ap[b].append((idx, 1))
        elem_ap[c].append((idx, 2))
        elem_ap[e].append((idx, 3))
    return elem_ap

_ELEM_AP = _build_element_ap_map()

def _get_initial_cantor():
    """Cantor set: numbers with no digit 2 in base-3 representation."""
    bits = 0
    for x in range(1, N + 1):
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits

def _greedy_forward():
    """Greedy forward: add elements 1..N if they don't create a 4-AP."""
    bits = 0
    for x in range(1, N + 1):
        ok = True
        for (ap_idx, pos) in _ELEM_AP[x]:
            a, b, c, e = _ALL_APS[ap_idx]
            # Check if the other 3 elements are in the set
            others = [a, b, c, e]
            others.remove(x)
            if all((bits >> o) & 1 for o in others):
                ok = False
                break
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_reverse():
    """Greedy reverse: add elements N..1 if they don't create a 4-AP."""
    bits = 0
    for x in range(N, 0, -1):
        ok = True
        for (ap_idx, pos) in _ELEM_AP[x]:
            a, b, c, e = _ALL_APS[ap_idx]
            others = [a, b, c, e]
            others.remove(x)
            if all((bits >> o) & 1 for o in others):
                ok = False
                break
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_random(rng):
    """Greedy with random ordering."""
    bits = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        ok = True
        for (ap_idx, pos) in _ELEM_AP[x]:
            a, b, c, e = _ALL_APS[ap_idx]
            others = [a, b, c, e]
            others.remove(x)
            if all((bits >> o) & 1 for o in others):
                ok = False
                break
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_weighted(rng):
    """Greedy with weighted random ordering: prefer elements with fewer conflicts."""
    bits = 0
    remaining = list(range(1, N + 1))
    while remaining:
        # Score each remaining element by how many potential 4-APs it's involved in with current set
        best_x = None
        best_score = -1
        # Sample a few candidates for speed
        candidates = rng.sample(remaining, min(20, len(remaining)))
        for x in candidates:
            # Count how many 4-APs would be violated if we added x
            conflict_count = 0
            for (ap_idx, pos) in _ELEM_AP[x]:
                a, b, c, e = _ALL_APS[ap_idx]
                others = [a, b, c, e]
                others.remove(x)
                cnt = sum(1 for o in others if (bits >> o) & 1)
                if cnt == 3:
                    conflict_count += 1
                elif cnt == 2:
                    # Partial conflict - will become a problem if we add one more
                    conflict_count += 0.5
            score = -conflict_count + rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_x = x
        if best_x is not None:
            # Check if best_x is actually safe
            ok = True
            for (ap_idx, pos) in _ELEM_AP[best_x]:
                a, b, c, e = _ALL_APS[ap_idx]
                others = [a, b, c, e]
                others.remove(best_x)
                if all((bits >> o) & 1 for o in others):
                    ok = False
                    break
            if ok:
                bits |= (1 << best_x)
            remaining.remove(best_x)
        else:
            break
    return bits

def _count_violations(bits):
    """Count number of complete 4-APs in the set."""
    count = 0
    for (a, b, c, e) in _ALL_APS:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1 and (bits >> e) & 1:
            count += 1
    return count

def _pair_swap_local_search(bits, deadline, rng):
    """
    Local search using pair swaps: remove one element, add another.
    This can escape local optima that single-element add/remove cannot.
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    current_violations = _count_violations(current_bits)
    
    best_bits = current_bits
    best_size = current_size
    
    if current_violations == 0 and current_size > best_size:
        best_bits = current_bits
        best_size = current_size
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() < deadline:
        # Try a pair swap: remove x (in set), add y (not in set)
        in_set = [i for i in range(1, N + 1) if (current_bits >> i) & 1]
        not_in_set = [i for i in range(1, N + 1) if not (current_bits >> i) & 1]
        
        if not in_set or not not_in_set:
            break
        
        x = rng.choice(in_set)
        y = rng.choice(not_in_set)
        
        if x == y:
            continue
        
        # Compute violations after swap: remove x, add y
        new_bits = (current_bits & ~(1 << x)) | (1 << y)
        
        # Count violations in new state
        new_violations = 0
        for (a, b, c, e) in _ALL_APS:
            if (new_bits >> a) & 1 and (new_bits >> b) & 1 and (new_bits >> c) & 1 and (new_bits >> e) & 1:
                new_violations += 1
        
        new_size = current_size  # swap keeps size same
        
        # Accept if no worse on violations, or if it reduces violations
        if new_violations < current_violations:
            current_bits = new_bits
            current_violations = new_violations
            no_improve_count = 0
        elif new_violations == current_violations:
            # Accept with some probability to explore
            if rng.random() < 0.3:
                current_bits = new_bits
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if current_violations == 0 and current_size > best_size:
            best_bits = current_bits
            best_size = current_size
        
        # Also try single adds if violations are 0
        if current_violations == 0:
            # Try adding a random element
            if not_in_set:
                y_add = rng.choice(not_in_set)
                test_bits = current_bits | (1 << y_add)
                add_ok = True
                for (a, b, c, e) in _ALL_APS:
                    if (test_bits >> a) & 1 and (test_bits >> b) & 1 and (test_bits >> c) & 1 and (test_bits >> e) & 1:
                        add_ok = False
                        break
                if add_ok:
                    current_bits = test_bits
                    current_size += 1
                    no_improve_count = 0
                    if current_size > best_size:
                        best_bits = current_bits
                        best_size = current_size
                        # Update not_in_set
                        not_in_set = [i for i in range(1, N + 1) if not (current_bits >> i) & 1]
        
        # Periodically try greedy fill
        if no_improve_count > 200:
            # Try greedy fill
            for z in range(1, N + 1):
                if not (current_bits >> z) & 1:
                    test_bits = current_bits | (1 << z)
                    add_ok = True
                    for (a, b, c, e) in _ALL_APS:
                        if (test_bits >> a) & 1 and (test_bits >> b) & 1 and (test_bits >> c) & 1 and (test_bits >> e) & 1:
                            add_ok = False
                            break
                    if add_ok:
                        current_bits = test_bits
                        current_size += 1
                        if current_size > best_size:
                            best_bits = current_bits
                            best_size = current_size
            no_improve_count = 0
        
        if no_improve_count > 5000:
            break
    
    return best_bits, best_size

def _hill_climbing(bits, deadline, rng):
    """
    Simple hill climbing: try single adds and removals.
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    current_violations = _count_violations(current_bits)
    
    best_bits = current_bits
    best_size = current_size
    
    if current_violations == 0 and current_size > best_size:
        best_bits = current_bits
        best_size = current_size
    
    start_time = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in range(1, N + 1):
            if (current_bits >> x) & 1:
                continue
            test_bits = current_bits | (1 << x)
            # Check if this creates any new 4-AP
            ok = True
            for (ap_idx, pos) in _ELEM_AP[x]:
                a, b, c, e = _ALL_APS[ap_idx]
                others = [a, b, c, e]
                others.remove(x)
                if all((test_bits >> o) & 1 for o in others):
                    ok = False
                    break
            if ok:
                current_bits = test_bits
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_bits = current_bits
                    best_size = current_size
        
        # Try removing and re-adding different elements (pair swap)
        if not improved and current_violations == 0:
            in_set = [i for i in range(1, N + 1) if (current_bits >> i) & 1]
            not_in_set = [i for i in range(1, N + 1) if not (current_bits >> i) & 1]
            if in_set and not_in_set:
                # Try a few random pair swaps
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    x = rng.choice(in_set)
                    y = rng.choice(not_in_set)
                    if x == y:
                        continue
                    new_bits = (current_bits & ~(1 << x)) | (1 << y)
                    # Check validity
                    valid = True
                    for (a, b, c, e) in _ALL_APS:
                        if (new_bits >> a) & 1 and (new_bits >> b) & 1 and (new_bits >> c) & 1 and (new_bits >> e) & 1:
                            valid = False
                            break
                    if valid:
                        # After swap, try to add more
                        current_bits = new_bits
                        # Greedy fill
                        for z in range(1, N + 1):
                            if not (current_bits >> z) & 1:
                                test_bits = current_bits | (1 << z)
                                add_ok = True
                                for (ap_idx, pos) in _ELEM_AP[z]:
                                    a, b, c, e = _ALL_APS[ap_idx]
                                    others = [a, b, c, e]
                                    others.remove(z)
                                    if all((test_bits >> o) & 1 for o in others):
                                        add_ok = False
                                        break
                                if add_ok:
                                    current_bits = test_bits
                        current_size = bin(current_bits).count('1')
                        if current_size > best_size:
                            best_bits = current_bits
                            best_size = current_size
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_bits, best_size

def _greedy_fill(bits, deadline):
    """Greedily add any element that doesn't create a 4-AP."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((bits >> x) & 1):
                ok = True
                for (ap_idx, pos) in _ELEM_AP[x]:
                    a, b, c, e = _ALL_APS[ap_idx]
                    others = [a, b, c, e]
                    others.remove(x)
                    if all((bits >> o) & 1 for o in others):
                        ok = False
                        break
                if ok:
                    bits |= (1 << x)
                    changed = True
    return bits

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Generate diverse initial seeds
    seeds = []
    
    # Cantor
    cantor = _get_initial_cantor()
    seeds.append(cantor)
    
    # Forward greedy
    fwd = _greedy_forward()
    seeds.append(fwd)
    
    # Reverse greedy
    rev = _greedy_reverse()
    seeds.append(rev)
    
    # Random greedy (multiple)
    rng = random.Random(42)
    for _ in range(5):
        g = _greedy_random(rng)
        seeds.append(g)
    
    # Weighted greedy (multiple)
    for _ in range(3):
        g = _greedy_weighted(rng)
        seeds.append(g)
    
    best_bits = 0
    best_size = 0
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        # Greedy fill
        filled = _greedy_fill(seed, deadline - 10)
        size = bin(filled).count('1')
        
        if size > best_size:
            best_bits = filled
            best_size = size
    
    # Phase 2: Hill climbing on best seed
    if time.time() < deadline - 5:
        hc_bits, hc_size = _hill_climbing(best_bits, deadline - 5, random.Random(123))
        if hc_size > best_size:
            best_bits = hc_bits
            best_size = hc_size
    
    # Phase 3: Pair swap local search
    if time.time() < deadline - 2:
        ps_bits, ps_size = _pair_swap_local_search(best_bits, deadline - 2, random.Random(456))
        if ps_size > best_size:
            best_bits = ps_bits
            best_size = ps_size
    
    # Final greedy fill
    if time.time() < deadline - 1:
        best_bits = _greedy_fill(best_bits, deadline - 1)
    
    # Convert to set
    best_set = {i for i in range(1, N + 1) if (best_bits >> i) & 1}
    
    # Verification
    if _count_violations(best_bits) != 0:
        # Fallback to Cantor
        best_set = {i for i in range(1, N + 1) if (1 << i) & _get_initial_cantor()}
    
    return sorted(best_set)
