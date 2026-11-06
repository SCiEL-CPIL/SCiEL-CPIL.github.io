# EVOLVE-BLOCK-START
import time
import random

def _try_add_fast(S, x, used_sums, forbidden_diffs):
    """Try to add x to Sidon set S.
    used_sums: set of all pair sums in S (including a+a)
    forbidden_diffs: set of diffs from the other set that must be avoided
    Returns (new_S, new_sums, new_diffs) or (None, None, None).
    """
    n = len(S)
    # Check Sidon: x + S[i] must not collide with existing sums
    for i in range(n):
        s = x + S[i]
        if s in used_sums:
            return None, None, None
    # Check x + x
    sx = x + x
    if sx in used_sums:
        return None, None, None
    # Check differences against forbidden (other set's diffs)
    for i in range(n):
        d = abs(x - S[i])
        if d > 0 and d in forbidden_diffs:
            return None, None, None
    
    new_S = S + [x]
    new_sums = set(used_sums)
    for i in range(n):
        new_sums.add(x + S[i])
    new_sums.add(sx)
    
    # Compute new diffs (all diffs involving x)
    new_diffs = set()
    for i in range(n):
        d = abs(x - S[i])
        if d > 0:
            new_diffs.add(d)
    
    return new_S, new_sums, new_diffs

def _compute_diffs(S):
    """Compute all nonzero absolute differences in S."""
    diffs = set()
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            d = abs(S[i] - S[j])
            if d > 0:
                diffs.add(d)
    return diffs

def _greedy_interleave(order, deadline, max_per_set=64):
    """Interleave adding elements to A and B."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Try to add to A
        if len(A) < max_per_set:
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A.update(new_diffs_A)
                continue
        
        # Try to add to B
        if len(B) < max_per_set:
            new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_A)
            if new_B is not None:
                B = new_B
                sums_B = new_sums_B
                diffs_B.update(new_diffs_B)
    
    return A, B

def _greedy_sequential(order, deadline, max_per_set=64):
    """Build A first, then B."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    # Build A
    for x in order:
        if time.time() > deadline:
            break
        if len(A) >= max_per_set:
            break
        new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, set())
        if new_A is not None:
            A = new_A
            sums_A = new_sums_A
            diffs_A.update(new_diffs_A)
    
    # Build B
    for x in order:
        if time.time() > deadline:
            break
        if len(B) >= max_per_set:
            break
        new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_A)
        if new_B is not None:
            B = new_B
            sums_B = new_sums_B
            diffs_B.update(new_diffs_B)
    
    return A, B

def _greedy_balanced(order, deadline, max_per_set=64):
    """Try to keep |A| and |B| balanced."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Prefer the smaller set
        if len(A) <= len(B) and len(A) < max_per_set:
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A.update(new_diffs_A)
                continue
        
        if len(B) < max_per_set:
            new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_A)
            if new_B is not None:
                B = new_B
                sums_B = new_sums_B
                diffs_B.update(new_diffs_B)
                continue
        
        # If B failed, try A
        if len(A) < max_per_set:
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A.update(new_diffs_A)
    
    return A, B

def _hill_climb(A, B, deadline):
    """Try to improve by adding elements."""
    A = list(A)
    B = list(B)
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    # Recompute
    for i in range(len(A)):
        for j in range(i, len(A)):
            sums_A.add(A[i] + A[j])
    for i in range(len(B)):
        for j in range(i, len(B)):
            sums_B.add(B[i] + B[j])
    diffs_A = _compute_diffs(A)
    diffs_B = _compute_diffs(B)
    
    # Try adding random unused elements
    used = set(A) | set(B)
    candidates = [x for x in range(1, 257) if x not in used]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates[:100]:
            if time.time() > deadline:
                break
            
            # Try adding to A
            if len(A) < 64:
                new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_B)
                if new_A is not None:
                    A = new_A
                    sums_A = new_sums_A
                    diffs_A.update(new_diffs_A)
                    candidates.remove(x)
                    improved = True
                    continue
            
            # Try adding to B
            if len(B) < 64:
                new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_A)
                if new_B is not None:
                    B = new_B
                    sums_B = new_sums_B
                    diffs_B.update(new_diffs_B)
                    candidates.remove(x)
                    improved = True
    
    return A, B

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    best_total = 0
    best_A = []
    best_B = []
    
    universe = list(range(1, 257))
    
    # Strategy 1: Sequential greedy with different random orders
    for trial in range(80):
        if time.time() > deadline:
            break
        order = list(universe)
        random.shuffle(order)
        A, B = _greedy_sequential(order, deadline)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
    
    # Strategy 2: Interleaved greedy
    for trial in range(80):
        if time.time() > deadline:
            break
        order = list(universe)
        random.shuffle(order)
        A, B = _greedy_interleave(order, deadline)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
    
    # Strategy 3: Balanced greedy
    for trial in range(80):
        if time.time() > deadline:
            break
        order = list(universe)
        random.shuffle(order)
        A, B = _greedy_balanced(order, deadline)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = list(A)
            best_B = list(B)
    
    # Hill climb on best result
    if time.time() < deadline - 3:
        A_hc, B_hc = _hill_climb(best_A, best_B, deadline - 3)
        total_hc = len(A_hc) + len(B_hc)
        if total_hc > best_total:
            best_total = total_hc
            best_A = A_hc
            best_B = B_hc
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
