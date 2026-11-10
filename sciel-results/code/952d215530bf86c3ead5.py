"""Rewritten program: randomized local search with NumPy vectorized sum counting."""
import numpy as np
import random
import time

N = 1000
MAX_REP = 2
TIME_BUDGET = 110  # seconds, leave margin under 120s


def _count_reps(arr):
    """
    Given a boolean array arr of length N+1 (index 0..N),
    compute for every integer s the number of pairs (a,b) with a<=b,
    a,b in the set (where set = indices where arr is True), and a+b = s.
    Returns an integer array reps of length 2N+1 (index 0..2N).
    Uses convolution: c = arr * arr gives number of ordered pairs (a,b) with a+b=s.
    For s, ordered pairs = c[s]. Pairs with a<=b:
        if a != b: each unordered pair counted twice in c, so count = c[s]//2 ... 
        but we need to be careful. Let's think:
        c[s] = number of ordered pairs (a,b) with a+b=s, a,b in set.
        We want number of (a,b) with a<=b, a+b=s.
        For a != b, each pair {a,b} appears as (a,b) and (b,a) in c, so contributes 2 to c[s], and 1 to our count.
        For a == b (i.e., s even and s/2 in set), appears once in c[s] as (a,a), and 1 to our count.
        So: our_count = (c[s] + diag[s]) // 2, where diag[s] = 1 if s even and s/2 in set else 0.
        Actually: c[s] = 2*(# pairs with a<b) + (# pairs with a==b)
        our_count = (# pairs with a<b) + (# pairs with a==b) = (c[s] - diag) / 2 + diag = (c[s] + diag) / 2.
    """
    # Use numpy convolution via FFT or direct? For N=1000, direct convolution is fine but let's use np.convolve for simplicity.
    # arr is boolean, convert to int
    a = arr.astype(np.float64)
    # c[s] = sum_{i} a[i] * a[s-i] for i in range, which is convolution
    c = np.convolve(a, a, mode='full')  # length 2*(N+1)-1 = 2N+1
    # c[s] for s in 0..2N
    # diag: for each s, 1 if s even and s//2 in set (and s//2 between 0 and N)
    s_vals = np.arange(2 * N + 1)
    even_mask = (s_vals % 2 == 0)
    half = s_vals // 2
    # half is valid index if 0 <= half <= N
    valid_half = (half >= 0) & (half <= N)
    diag = np.zeros(2 * N + 1, dtype=np.float64)
    diag[even_mask & valid_half] = arr[half[even_mask & valid_half]].astype(np.float64)
    reps = (c + diag) / 2.0
    return reps


def construct():
    start_time = time.time()
    
    # Dense random initialization: select ~700 integers from 1..1000
    # We want to start dense and then prune/swap.
    # Let's start with a random set of size ~650 (since B2[2] allows roughly 3/2 * sqrt(N) * something? 
    # Actually for N=1000, the greedy gives some size. Let's estimate: greedy likely gives ~300-400.
    # The asymptotic constant is at least 3/2 for the extremal function... 
    # Wait, the task says "the asymptotic constant is at least 3/2" - this likely refers to the density constant.
    # For B2[2] sets (Sidon-like with multiplicity 2), the maximum size is ~ c * sqrt(N).
    # For N=1000, sqrt(1000) ~ 31.6, so size might be in the hundreds.
    # Let's start with ~500 random elements and see.
    
    random.seed(42)
    
    # Start with a moderately dense random set
    init_size = 500
    initial_set = set(random.sample(range(1, N + 1), init_size))
    
    # Build boolean array
    arr = np.zeros(N + 1, dtype=np.int8)
    for x in initial_set:
        arr[x] = 1
    
    # Compute initial reps
    reps = _count_reps(arr)
    
    # Track best valid solution
    best_set = None
    best_size = 0
    
    # First, try to prune invalid elements to get a valid starting point
    # Then do local search: try swaps and additions
    
    # Helper to check if current set is valid
    def is_valid():
        # reps[s] for s in 2..2N should be <= MAX_REP
        # But we only care about s where reps[s] > MAX_REP
        # Note: reps[0] and reps[1] are always 0 (no elements 0 or 1 can form them with a<=b from positive set)
        return np.all(reps[:2*N+1] <= MAX_REP + 1e-9)
    
    # Prune phase: remove elements that cause violations
    # We'll iteratively remove the element that contributes most to violations
    changed = True
    while changed and (time.time() - start_time) < TIME_BUDGET * 0.3:
        changed = False
        # Find violations
        viol = reps > MAX_REP + 1e-9
        if not np.any(viol):
            break
        
        # For each element x in the set, compute how many violations it "causes"
        # A violation at sum s is caused by pairs (a,b) with a+b=s.
        # Removing x affects all sums s where x+y=s for some y in set, and s=2x.
        # We'll compute for each x the number of violating sums that would be fixed by removing x.
        
        current_elements = np.where(arr > 0)[0]
        current_elements = current_elements[current_elements >= 1]  # exclude index 0
        
        if len(current_elements) == 0:
            break
        
        # For each element x, compute the impact of removing it
        # We'll do this approximately: for each x, count how many violating sums s
        # have a representation involving x.
        
        # To be efficient, let's compute for each x the number of pairs (x, y) with y in set
        # such that x+y is a violating sum, plus (x,x) if 2x is violating.
        
        best_remove = None
        best_gain = -1
        
        for x in current_elements:
            # Sums involving x: x + y for y in set, and 2*x
            # We need to count how many of these sums are currently violating
            # and how many would still be violating after removing x (i.e., have other representations)
            
            # For sum s = x + y (y in set, y != x), the number of representations decreases by 1 when x is removed.
            # For sum s = 2*x, the number of representations decreases by 1 when x is removed (the pair (x,x)).
            
            # We want to maximize the number of violations that are resolved (reps[s] goes from >2 to <=2)
            # This is complex. Let's use a simpler heuristic: remove the element that is involved in the most violating pairs.
            
            # Count violating sums involving x
            # For each y in set with y != x: s = x+y, check if reps[s] > MAX_REP
            # For s = 2*x: check if reps[s] > MAX_REP
            
            # This is O(n^2) per iteration which might be slow. Let's vectorize.
            
            # For a given x, the sums are x + set_elements
            # We can compute: for each y in set, s = x+y, and check reps[s]
            
            y_vals = current_elements
            s_vals = x + y_vals
            valid_s = (s_vals >= 0) & (s_vals <= 2*N)
            s_vals = s_vals[valid_s]
            
            # Also include 2*x (which is x+x, already included if x is in set)
            # The above includes x+x when y=x
            
            if len(s_vals) == 0:
                continue
            
            # Count how many of these sums are violating
            viol_count = np.sum(reps[s_vals] > MAX_REP + 1e-9)
            
            # Also, we should weight by how "critical" the violation is.
            # Let's just use viol_count as a proxy.
            
            if viol_count > best_gain:
                best_gain = viol_count
                best_remove = x
        
        if best_remove is not None:
            arr[best_remove] = 0
            initial_set.discard(best_remove)
            reps = _count_reps(arr)
            changed = True
    
    # Now we have a valid set (hopefully). Record it.
    if is_valid():
        best_set = set(np.where(arr > 0)[0])
        best_set.discard(0)
        best_size = len(best_set)
    
    # Local search phase: try to add elements and swap to improve size
    # We'll iterate: try adding a random element, if it violates, try to fix by removing/swapping.
    
    current_set = set(best_set) if best_set else set()
    arr = np.zeros(N + 1, dtype=np.int8)
    for x in current_set:
        if 1 <= x <= N:
            arr[x] = 1
    reps = _count_reps(arr)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while (time.time() - start_time) < TIME_BUDGET:
        # Try to add a random element not in the set
        candidates = [x for x in range(1, N + 1) if arr[x] == 0]
        if not candidates:
            break
        
        # Try a few random candidates
        trial_count = 20
        improved = False
        
        for _ in range(trial_count):
            x = random.choice(candidates)
            
            # Check if adding x keeps the set valid
            # New sums: x + y for y in current_set, and 2*x
            # We need to check that no sum exceeds MAX_REP after adding x
            
            # Compute new reps incrementally
            # For each y in current_set: s = x+y, reps[s] increases by 1
            # For s = 2*x: reps[s] increases by 1
            
            current_elements = np.where(arr > 0)[0]
            current_elements = current_elements[current_elements >= 1]
            
            if len(current_elements) > 0:
                s_vals = x + current_elements
                valid_s = (s_vals >= 0) & (s_vals <= 2*N)
                s_vals = s_vals[valid_s]
            else:
                s_vals = np.array([], dtype=int)
            
            # Include 2*x
            s_vals = np.concatenate([s_vals, [2*x]])
            
            # Check if any of these sums would exceed MAX_REP
            new_reps = reps.copy()
            new_reps[s_vals] += 1
            
            if np.all(new_reps[:2*N+1] <= MAX_REP + 1e-9):
                # Valid addition!
                arr[x] = 1
                reps = new_reps
                current_set.add(x)
                improved = True
                no_improve_count = 0
                break
            else:
                # Invalid. Try to fix by removing one element that causes the most conflicts.
                # Find the sums that would violate
                viol_s = s_vals[new_reps[s_vals] > MAX_REP + 1e-9]
                
                if len(viol_s) > 0:
                    # For each candidate removal y in current_set, check if removing y and adding x fixes the violations
                    # This is expensive. Let's try a simpler approach: just skip this x.
                    pass
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a swap: remove one element and add another
                # Pick a random element to remove and try to replace it
                if len(current_set) > 0:
                    to_remove = random.choice(list(current_set))
                    arr[to_remove] = 0
                    current_set.discard(to_remove)
                    reps = _count_reps(arr)
                    no_improve_count = 0
                else:
                    break
    
    # Final check and return
    if is_valid():
        final_set = set(np.where(arr > 0)[0])
        final_set.discard(0)
        if len(final_set) > best_size:
            best_set = final_set
            best_size = len(final_set)
    
    if best_set is None or len(best_set) == 0:
        # Fallback: return empty or a small valid set
        return []
    
    return sorted(best_set)
