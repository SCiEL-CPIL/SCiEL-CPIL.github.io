# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_sidon(S):
    """Check Sidon property: all pair sums (with replacement) are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    """Return set of absolute nonzero differences."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    """Validate solution. Returns (is_valid, score)."""
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def _backtrack_construct(max_n=256, max_size=64, time_budget=8.0, rng=None,
                         order=None, target_size=None):
    """
    Backtracking with forward checking.
    Assigns each x in order to A, B, or skip.
    State: A, B (sorted lists), sums_A, sums_B (sets of pair sums),
           diff_used (set of all differences used by either A or B).
    """
    if rng is None:
        rng = random.Random(0)
    if order is None:
        order = list(range(1, max_n + 1))
    if target_size is None:
        target_size = max_size

    start = time.time()
    best = [0, 0]  # sizes

    # We'll do a depth-first search with pruning.
    # To keep it tractable, we limit the search and use a heuristic order.
    
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diff_used = set()

    def _try_add_to_set(S, sums, x):
        """Check if x can be added to S (Sidon) without sum conflicts.
        Returns (ok, new_sums, new_diffs) where new_diffs are diffs from x to S."""
        new_diffs = set()
        for a in S:
            s = x + a
            if s in sums:
                return False, None, None
            d = abs(x - a)
            if d > 0:
                if d in diff_used:
                    return False, None, None
                new_diffs.add(d)
        sx = x + x
        if sx in sums:
            return False, None, None
        new_sums = sums | {x + a for a in S} | {sx}
        return True, new_sums, new_diffs

    def _search(idx):
        nonlocal A, B, sums_A, sums_B, diff_used, best

        if time.time() - start > time_budget:
            return False

        if idx == len(order):
            total = len(A) + len(B)
            if total > sum(best):
                best[0] = len(A)
                best[1] = len(B)
            return False

        x = order[idx]

        # Try adding to A
        if len(A) < max_size:
            ok, ns, nd = _try_add_to_set(A, sums_A, x)
            if ok:
                A.append(x)
                old_sums_A = sums_A
                old_diff_used = diff_used
                sums_A = ns
                diff_used = diff_used | nd
                if _search(idx + 1):
                    return True
                A.pop()
                sums_A = old_sums_A
                diff_used = old_diff_used

        # Try adding to B
        if len(B) < max_size:
            ok, ns, nd = _try_add_to_set(B, sums_B, x)
            if ok:
                B.append(x)
                old_sums_B = sums_B
                old_diff_used = diff_used
                sums_B = ns
                diff_used = diff_used | nd
                if _search(idx + 1):
                    return True
                B.pop()
                sums_B = old_sums_B
                diff_used = old_diff_used

        # Skip x
        if _search(idx + 1):
            return True

        return False

    # Use a heuristic: try to reach target_size in each set
    # Start with a randomized order to explore different structures
    _search(0)
    
    return A, B, best

def _greedy_init(max_n=256, max_size=64):
    """Quick greedy to get a valid starting point."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diff_used = set()
    
    for x in range(1, max_n + 1):
        # Try A
        if len(A) < max_size:
            ok = True
            new_diffs = set()
            for a in A:
                s = x + a
                if s in sums_A:
                    ok = False
                    break
                d = abs(x - a)
                if d > 0 and d in diff_used:
                    ok = False
                    break
                if d > 0:
                    new_diffs.add(d)
            if ok:
                sx = x + x
                if sx in sums_A:
                    ok = False
            if ok:
                A.append(x)
                for a in A:
                    if a != x:
                        sums_A.add(x + a)
                sums_A.add(x + x)
                diff_used |= new_diffs
                continue
        
        # Try B
        if len(B) < max_size:
            ok = True
            new_diffs = set()
            for b in B:
                s = x + b
                if s in sums_B:
                    ok = False
                    break
                d = abs(x - b)
                if d > 0 and d in diff_used:
                    ok = False
                    break
                if d > 0:
                    new_diffs.add(d)
            if ok:
                sx = x + x
                if sx in sums_B:
                    ok = False
            if ok:
                B.append(x)
                for b in B:
                    if b != x:
                        sums_B.add(x + b)
                sums_B.add(x + x)
                diff_used |= new_diffs
                continue
    
    return A, B

def _improve_with_backtrack(A, B, max_n=256, max_size=64, time_budget=5.0, rng=None):
    """
    Use backtracking starting from a greedy solution, trying to add more elements.
    We re-run backtracking but with a bias: prefer assignments that keep current elements.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    best_A, best_B = list(A), list(B)
    best_score = len(A) + len(B)
    
    # Create order: current elements first (to keep them), then others
    used = set(A) | set(B)
    remaining = [x for x in range(1, max_n + 1) if x not in used]
    rng.shuffle(remaining)
    
    # Order: interleave A and B elements, then remaining
    A_sorted = sorted(A)
    B_sorted = sorted(B)
    
    # Build order: try to keep existing, add new
    order = []
    ai, bi = 0, 0
    while ai < len(A_sorted) or bi < len(B_sorted):
        if ai < len(A_sorted) and (bi >= len(B_sorted) or A_sorted[ai] <= B_sorted[bi]):
            order.append(A_sorted[ai])
            ai += 1
        else:
            order.append(B_sorted[bi])
            bi += 1
    order.extend(remaining)
    
    A_new = []
    B_new = []
    sums_A = set()
    sums_B = set()
    diff_used = set()
    
    def _try_add_to_set(S, sums, x):
        new_diffs = set()
        for a in S:
            s = x + a
            if s in sums:
                return False, None, None
            d = abs(x - a)
            if d > 0:
                if d in diff_used:
                    return False, None, None
                new_diffs.add(d)
        sx = x + x
        if sx in sums:
            return False, None, None
        new_sums = sums | {x + a for a in S} | {sx}
        return True, new_sums, new_diffs
    
    def _search(idx):
        nonlocal A_new, B_new, sums_A, sums_B, diff_used, best_A, best_B, best_score
        
        if time.time() - start > time_budget:
            return False
        
        if idx == len(order):
            total = len(A_new) + len(B_new)
            if total > best_score:
                best_score = total
                best_A = list(A_new)
                best_B = list(B_new)
            return False
        
        x = order[idx]
        
        # Determine if x was in original A or B
        in_A = x in A
        in_B = x in B
        
        # Try A (prefer if was in A)
        if len(A_new) < max_size and (in_A or not in_B):
            ok, ns, nd = _try_add_to_set(A_new, sums_A, x)
            if ok:
                A_new.append(x)
                old_sums_A = sums_A
                old_diff_used = diff_used
                sums_A = ns
                diff_used = diff_used | nd
                if _search(idx + 1):
                    return True
                A_new.pop()
                sums_A = old_sums_A
                diff_used = old_diff_used
        
        # Try B
        if len(B_new) < max_size and (in_B or not in_A):
            ok, ns, nd = _try_add_to_set(B_new, sums_B, x)
            if ok:
                B_new.append(x)
                old_sums_B = sums_B
                old_diff_used = diff_used
                sums_B = ns
                diff_used = diff_used | nd
                if _search(idx + 1):
                    return True
                B_new.pop()
                sums_B = old_sums_B
                diff_used = old_diff_used
        
        # Skip
        if _search(idx + 1):
            return True
        
        return False
    
    _search(0)
    
    return best_A, best_B, best_score

def construct():
    """
    Main: greedy init + backtracking improvement with multiple random orders.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s kill
    
    # Phase 1: Greedy init
    A, B = _greedy_init(256, 64)
    valid, score = _validate(A, B)
    if not valid:
        A, B = [1], [3]
        valid, score = _validate(A, B)
    
    best_A, best_B = list(A), list(B)
    best_score = score if valid else 0
    
    # Phase 2: Backtracking improvement with multiple random seeds
    for seed in range(20):
        if time.time() > deadline - 3.0:
            break
        
        remaining = deadline - 3.0 - time.time()
        if remaining <= 1.0:
            break
        
        rng = random.Random(seed)
        A_imp, B_imp, score_imp = _improve_with_backtrack(
            best_A, best_B, max_n=256, max_size=64,
            time_budget=min(remaining, 4.0), rng=rng
        )
        
        valid, score = _validate(A_imp, B_imp)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A_imp), list(B_imp)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
