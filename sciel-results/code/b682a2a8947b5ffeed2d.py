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

def _backtrack_build(order, max_n=256, max_size=64, time_limit=1.5, rng=None):
    """
    Backtracking: assign each x in `order` to A, B, or skip.
    Uses forward checking on Sidon sums and diff disjointness.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diff_A = set()
    diff_B = set()
    
    def can_add_sidon(S, x, sums):
        """Check if x can be added to S maintaining Sidon."""
        for a in S:
            s = x + a
            if s in sums:
                return False
        sx = x + x
        if sx in sums:
            return False
        return True
    
    def can_add_diff(S, x, diff_self, diff_other):
        """Check if differences from x to S don't conflict."""
        for a in S:
            d = abs(x - a)
            if d > 0:
                if d in diff_self or d in diff_other:
                    return False
        return True
    
    def add_to_set(S, x, sums, diff_self, diff_other):
        """Add x to S, update sums and diffs. Returns list of (diff) added for undo."""
        new_diffs = []
        for a in S:
            d = abs(x - a)
            if d > 0:
                new_diffs.append(d)
                diff_self.add(d)
        for a in S:
            s = x + a
            sums.add(s)
        sums.add(x + x)
        S.append(x)
        return new_diffs
    
    def remove_from_set(S, x, sums, diff_self, new_diffs):
        """Remove x from S, undo sums and diffs."""
        S.remove(x)
        for a in S:
            s = x + a
            sums.discard(s)
            d = abs(x - a)
            if d > 0 and d in new_diffs:
                diff_self.discard(d)
        sums.discard(x + x)
    
    best_A = []
    best_B = []
    best_score = 0
    
    def backtrack(idx):
        nonlocal best_A, best_B, best_score
        if time.time() - start > time_limit:
            return
        if idx == len(order):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
            return
        
        x = order[idx]
        
        # Try adding to A
        if len(A) < max_size and can_add_sidon(A, x, sums_A) and can_add_diff(A, x, diff_A, diff_B):
            new_diffs = add_to_set(A, x, sums_A, diff_A, diff_B)
            backtrack(idx + 1)
            remove_from_set(A, x, sums_A, diff_A, new_diffs)
        
        # Try adding to B
        if len(B) < max_size and can_add_sidon(B, x, sums_B) and can_add_diff(B, x, diff_B, diff_A):
            new_diffs = add_to_set(B, x, sums_B, diff_B, diff_A)
            backtrack(idx + 1)
            remove_from_set(B, x, sums_B, diff_B, new_diffs)
        
        # Try skipping
        backtrack(idx + 1)
    
    backtrack(0)
    return best_A, best_B

def _greedy_init(max_n=256, max_size=64):
    """Fast greedy to get a starting point."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diff_A = set()
    diff_B = set()
    
    for x in range(1, max_n + 1):
        # Try the set with fewer elements first
        if len(A) <= len(B):
            if len(A) < max_size:
                ok = True
                for a in A:
                    if (x + a) in sums_A or (x + x) in sums_A:
                        ok = False
                        break
                if ok:
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (d in diff_A or d in diff_B):
                            ok = False
                            break
                if ok:
                    for a in A:
                        sums_A.add(x + a)
                        d = abs(x - a)
                        if d > 0:
                            diff_A.add(d)
                    sums_A.add(x + x)
                    A.append(x)
                    continue
        if len(B) < max_size:
            ok = True
            for b in B:
                if (x + b) in sums_B or (x + x) in sums_B:
                    ok = False
                    break
            if ok:
                for b in B:
                    d = abs(x - b)
                    if d > 0 and (d in diff_B or d in diff_A):
                        ok = False
                        break
            if ok:
                for b in B:
                    sums_B.add(x + b)
                    d = abs(x - b)
                    if d > 0:
                        diff_B.add(d)
                sums_B.add(x + x)
                B.append(x)
                continue
    
    return A, B

def _improve_with_backtrack(A, B, max_n=256, max_size=64, time_budget=3.0, rng=None):
    """
    Use backtracking on a subset of numbers to improve the solution.
    We fix the current A and B, then try to reassign some numbers.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    best_A = list(A)
    best_B = list(B)
    best_score = len(A) + len(B)
    
    # Get all numbers 1..256, shuffle for diversity
    all_nums = list(range(1, max_n + 1))
    
    for attempt in range(20):
        if time.time() - start > time_budget:
            break
        
        rng.shuffle(all_nums)
        
        # Run backtracking with limited time
        t_budget = min(1.0, time_budget - (time.time() - start))
        if t_budget < 0.3:
            break
        
        A_new, B_new = _backtrack_build(
            all_nums, max_n=max_n, max_size=max_size,
            time_limit=t_budget, rng=rng
        )
        
        if A_new and B_new:
            valid, score = _validate(A_new, B_new)
            if valid and score > best_score:
                best_score = score
                best_A = list(A_new)
                best_B = list(B_new)
    
    return best_A, best_B, best_score

def construct():
    """
    Main function: greedy init + backtracking improvement with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Phase 1: Greedy initialization
    A, B = _greedy_init(256, 64)
    valid, score = _validate(A, B)
    if not valid:
        A, B = [1], [3]
        valid, score = _validate(A, B)
    
    best_A, best_B = list(A), list(B)
    best_score = score if valid else 0
    
    # Phase 2: Backtracking improvement with multiple seeds
    if time.time() < deadline - 3.0:
        for seed in range(15):
            if time.time() > deadline - 2.0:
                break
            remaining = deadline - 2.0 - time.time()
            if remaining <= 1.0:
                break
            
            rng = random.Random(seed * 7919 + 13)
            A_ls, B_ls, score_ls = _improve_with_backtrack(
                best_A, best_B, max_n=256, max_size=64,
                time_budget=min(remaining * 0.6, 4.0), rng=rng
            )
            
            valid, score = _validate(A_ls, B_ls)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A_ls), list(B_ls)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
