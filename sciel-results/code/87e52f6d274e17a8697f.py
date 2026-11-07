"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START

N = 128
DEADLINE_MARGIN = 2.0  # seconds before the 90s limit

def _validity_check(S):
    """Check if set S (as a set of ints) satisfies the constraint.
    Forbid three distinct members a,b,c for which a+b, a+c and b+c are also in S.
    Note: a+b, a+c, b+c must be in S, and they need not be distinct from each other
    or from a,b,c. But a,b,c must be distinct.
    """
    S_set = set(S)
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _check_valid(S):
    """Faster validity check using sorted list and two-pointer or direct lookup."""
    S_set = set(S)
    # For each pair (a,b), check if there exists c in S, c != a, c != b,
    # such that a+c in S and b+c in S.
    # Actually the constraint is: no three DISTINCT a,b,c in S with a+b, a+c, b+c all in S.
    # Let's just do the triple loop but with early exit and set lookup.
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = lst[k]
                if (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _check_valid_fast(S_set):
    """Check validity given a set. Uses the fact that if a+b is in S, we need to check."""
    lst = sorted(S_set)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            ab = a + b
            if ab not in S_set:
                continue
            # Now check if there's a c > b (or c != a,b) such that a+c and b+c in S_set
            for k in range(j+1, n):
                c = lst[k]
                if (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _greedy_construct():
    """Greedy: add numbers in order, skip if it creates a violation."""
    S = set()
    for x in range(1, N+1):
        # Check if adding x would create a violation
        S_trial = S | {x}
        # We need to check: are there distinct a,b,c in S_trial with a+b,a+c,b+c in S_trial?
        # Only triples involving x can be new violations.
        # So check triples where x is one of a,b,c, or x is one of the sums.
        # Case 1: x is one of the three distinct members.
        #   Then we need two other distinct members p,q in S such that:
        #   the three sums are x+p, x+q, p+q, all in S_trial.
        #   Since x is in S_trial, and p,q in S subset S_trial:
        #   x+p in S_trial, x+q in S_trial, p+q in S_trial.
        # Case 2: x is one of the sums.
        #   Sub-case 2a: x = a+b for distinct a,b in S. Then we need a+c and b+c in S_trial for some c distinct from a,b.
        #     But a+c and b+c must be in S_trial. Since c is in S (not x, as x is a sum), a+c, b+c in S or equal to x.
        #     Actually this is getting complex. Let's just do a full check for small sets.
        if len(S_trial) < 20:
            if not _check_valid_fast(S_trial):
                continue
        else:
            # For larger sets, do a targeted check
            valid = True
            # Check triples involving x as a member
            for p in S:
                for q in S:
                    if p == q:
                        continue
                    if x + p in S_trial and x + q in S_trial and p + q in S_trial:
                        # Need x, p, q to be distinct
                        if x != p and x != q:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                # Check triples where x is a sum
                # x = a + b, a != b, both in S
                for a in S:
                    b = x - a
                    if b in S and a != b and a < b:
                        # Now need c distinct from a,b such that a+c and b+c in S_trial
                        for c in S:
                            if c == a or c == b:
                                continue
                            if (a+c) in S_trial and (b+c) in S_trial:
                                valid = False
                                break
                    if not valid:
                        break
            if not valid:
                continue
        S.add(x)
    return S

def _local_search(S_set, deadline):
    """Try to improve the set via single removals and additions."""
    best = S_set
    best_size = len(S_set)
    start_time = time.time()
    
    # Try removing one element and adding one or more
    # Simple approach: try all single swaps
    changed = True
    while changed and time.time() < deadline:
        changed = False
        S_list = sorted(S_set)
        for i, remove_elem in enumerate(S_list):
            if time.time() >= deadline:
                break
            S_without = S_set - {remove_elem}
            # Try adding elements not in S_without
            for add_elem in range(1, N+1):
                if add_elem in S_without:
                    continue
                S_trial = S_without | {add_elem}
                if len(S_trial) <= best_size:
                    continue
                # Check validity
                if _check_valid_fast(S_trial):
                    S_set = S_trial
                    best = S_trial
                    best_size = len(S_set)
                    changed = True
                    break
            if changed:
                break
    return best

def _randomized_greedy(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(1, N+1))
    rng.shuffle(order)
    S = set()
    for x in order:
        S_trial = S | {x}
        if len(S_trial) < 25:
            if not _check_valid_fast(S_trial):
                continue
        else:
            valid = True
            for p in S:
                for q in S:
                    if p == q:
                        continue
                    if x != p and x != q:
                        if (x+p) in S_trial and (x+q) in S_trial and (p+q) in S_trial:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                for a in S:
                    b = x - a
                    if b in S and a != b and a < b:
                        for c in S:
                            if c != a and c != b:
                                if (a+c) in S_trial and (b+c) in S_trial:
                                    valid = False
                                    break
                    if not valid:
                        break
            if not valid:
                continue
        S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN + 80  # 80s budget, 10s margin
    
    # Start with a good initial solution
    # The seed gives 64 elements (65..128)
    S0 = set(range(N//2+1, N+1))
    best = S0
    best_size = len(S0)
    
    # Try greedy constructions
    greedy1 = _greedy_construct()
    if len(greedy1) > best_size:
        best = greedy1
        best_size = len(greedy1)
    
    # Try several randomized greedies
    for seed in range(20):
        if time.time() >= deadline:
            break
        S_r = _randomized_greedy(seed)
        if len(S_r) > best_size:
            best = S_r
            best_size = len(S_r)
    
    # Local search on the best found
    if time.time() < deadline:
        best = _local_search(best, deadline)
        best_size = len(best)
    
    # Try local search on other candidates
    if time.time() < deadline:
        greedy1 = _greedy_construct()
        if len(greedy1) > best_size:
            best = _local_search(greedy1, deadline)
            best_size = len(best)
    
    # Final local search
    if time.time() < deadline - 5:
        best = _local_search(best, deadline)
        best_size = len(best)
    
    result = sorted(best)
    # Ensure all are Python int
    result = [int(x) for x in result]
    return result

# EVOLVE-BLOCK-END
