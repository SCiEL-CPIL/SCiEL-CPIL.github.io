import time
import random

N = 256

def _can_add(sel_bits, sums_bits, x):
    """
    Check if x can be added to the set represented by sel_bits and sums_bits.
    sel_bits: integer where bit i is 1 if (i+1) is in set.
    sums_bits: integer where bit s is 1 if sum s is in sums.
    x: 1-based index to test.
    """
    # Check if 2x is already a sum
    if (sums_bits >> (2 * x)) & 1:
        return False
    
    # Check if x + a is a sum for any a in set
    # We need to check if any bit in sums_bits corresponds to (x + a) where a is in sel_bits
    # Let's iterate through set bits in sel_bits
    temp = sel_bits
    while temp:
        # Get lowest set bit
        lb = temp & (-temp)
        a_idx = lb.bit_length() - 1
        a_val = a_idx + 1
        s = x + a_val
        if (sums_bits >> s) & 1:
            return False
        temp ^= lb
    return True

def _add(sel_bits, sums_bits, x):
    """Add x to the set and update sums."""
    new_sel = sel_bits | (1 << (x - 1))
    new_sums = sums_bits
    # Add 2x
    new_sums |= (1 << (2 * x))
    # Add x + a for all a in old set
    temp = sel_bits
    while temp:
        lb = temp & (-temp)
        a_idx = lb.bit_length() - 1
        a_val = a_idx + 1
        s = x + a_val
        new_sums |= (1 << s)
        temp ^= lb
    return new_sel, new_sums

def _is_maximal(sel_bits, sums_bits, deadline):
    """
    Check if the set is maximal.
    Returns True if no missing element can be added.
    """
    if time.time() > deadline:
        return False
    for x in range(1, N + 1):
        if (sel_bits >> (x - 1)) & 1:
            continue
        if _can_add(sel_bits, sums_bits, x):
            return False
    return True

def _count_missing_addable(sel_bits, sums_bits, x):
    """
    Count how many missing elements are "blocked" by adding x.
    Specifically, how many missing elements y (y != x) become un-addable if we add x?
    Actually, for maximality, we just need to ensure that for every missing y, 
    it is not addable. 
    A better heuristic for minimization: 
    When choosing the next element to add, prefer elements that "cover" (block) the most 
    other potential elements, reducing the number of elements we *might* need later?
    No, we want to minimize the set size. 
    If we add x, it might prevent us from adding y, z, etc. 
    But we *must* block all missing elements eventually.
    If we add x, and x blocks y, we don't need to add y (unless y is needed to block something else).
    So, elements that block many other "candidates" are good.
    """
    if time.time() > deadline:
        return 0
    count = 0
    # Temporarily add x
    new_sel, new_sums = _add(sel_bits, sums_bits, x)
    # Check which other missing elements are now un-addable
    for y in range(1, N + 1):
        if y == x:
            continue
        if (sel_bits >> (y - 1)) & 1:
            continue
        # If y was addable before, but not after? 
        # Or just count how many are not addable now?
        # We want to maximize the "coverage" of constraints.
        # If y is not addable in new state, it's covered.
        # But we don't know if it was addable before.
        # Let's just count how many are NOT addable in the new state.
        if not _can_add(new_sel, new_sums, y):
            count += 1
    return count

def _backtrack(depth, sel_bits, sums_bits, best_set, best_size, deadline):
    """
    Recursive backtracking to find a small maximal Sidon set.
    """
    if time.time() > deadline:
        return
    
    # Pruning: if current size is already >= best_size, stop
    current_size = bin(sel_bits).count('1')
    if current_size >= best_size[0]:
        return
    
    # Check if maximal
    if _is_maximal(sel_bits, sums_bits, deadline):
        if current_size < best_size[0]:
            best_size[0] = current_size
            best_set[0] = sel_bits
            # Optional: stop early if we find a very small set? 
            # Theoretical lower bound for Sidon set of size k in [1, N] is roughly sqrt(N).
            # For N=256, sqrt(256)=16. Maximal sets might be larger.
            # If we find size < 20, it's very good.
            if current_size <= 18:
                raise TimeoutError("Found very small set")
        return

    # Generate candidates
    candidates = []
    for x in range(1, N + 1):
        if (sel_bits >> (x - 1)) & 1:
            continue
        if _can_add(sel_bits, sums_bits, x):
            # Heuristic score: how many other elements does adding x block?
            # Higher score = more blocking = potentially smaller final set
            score = _count_missing_addable(sel_bits, sums_bits, x)
            candidates.append((score, x))
    
    if not candidates:
        # Not maximal, but no candidates? This shouldn't happen if _is_maximal is correct.
        # If no candidates and not maximal, it means we are stuck? 
        # Actually, if no candidates are addable, it IS maximal. 
        # So this branch is redundant with _is_maximal check, but _is_maximal checks ALL missing.
        # If _is_maximal returned False, there is at least one addable element.
        # So candidates should not be empty.
        return

    # Sort by score descending (greedy heuristic for backtracking)
    candidates.sort(key=lambda t: -t[0])
    
    # Limit branching factor to keep time manageable
    # If we have too many candidates, pick top K
    K = 10
    for score, x in candidates[:K]:
        new_sel, new_sums = _add(sel_bits, sums_bits, x)
        _backtrack(depth + 1, new_sel, new_sums, best_set, best_size, deadline)
        if time.time() > deadline:
            return

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin
    
    # Best result storage
    best_set = [0]  # Bitmask
    best_size = [float('inf')]
    
    # Initial greedy to get a good upper bound
    # Use a simple greedy with randomization or heuristic
    sel_bits = 0
    sums_bits = 0
    # Greedy: pick element that blocks most others
    remaining = list(range(1, N + 1))
    random.shuffle(remaining)
    
    # Phase 1: Greedy construction to get an initial maximal set
    # We will build a set until it is maximal.
    # Instead of picking one by one, let's just run a fast greedy to get a valid maximal set.
    # This gives us an upper bound for backtracking.
    
    def greedy_construct(deadline):
        s_b = 0
        su_b = 0
        # Pick random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if time.time() > deadline:
                break
            if _can_add(s_b, su_b, x):
                s_b, su_b = _add(s_b, su_b, x)
        # Saturate
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            for x in range(1, N + 1):
                if (s_b >> (x - 1)) & 1:
                    continue
                if _can_add(s_b, su_b, x):
                    s_b, su_b = _add(s_b, su_b, x)
                    changed = True
        return s_b, su_b

    # Run a few greedy trials to find a decent upper bound
    for _ in range(10):
        if time.time() > deadline:
            break
        s_b, su_b = greedy_construct(deadline)
        sz = bin(s_b).count('1')
        if sz < best_size[0]:
            best_size[0] = sz
            best_set[0] = s_b

    # Phase 2: Backtracking to improve
    # Start from empty set
    try:
        _backtrack(0, 0, 0, best_set, best_size, deadline)
    except TimeoutError:
        pass
    
    # Convert bitmask to list
    result = []
    for i in range(N):
        if (best_set[0] >> i) & 1:
            result.append(i + 1)
            
    # Fallback if backtracking failed or didn't find anything
    if not result:
        s_b, su_b = greedy_construct(deadline)
        for i in range(N):
            if (s_b >> i) & 1:
                result.append(i + 1)
                
    return result

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Result size: {len(res)}, Time: {t1-t0:.2f}s")
    # Verify
    sums = set()
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            s = res[i] + res[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not valid:
        print("Invalid Sidon set!")
    else:
        # Check maximality
        in_set = set(res)
        for x in range(1, 257):
            if x in in_set:
                continue
            # Check if x can be added
            can = True
            for a in res:
                if (x + a) in sums:
                    can = False
                    break
            if (2*x) in sums:
                can = False
            if can:
                print(f"Not maximal: {x} can be added")
                break
        else:
            print("Maximal Sidon set verified.")
