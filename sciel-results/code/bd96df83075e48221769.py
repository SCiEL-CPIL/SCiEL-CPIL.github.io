"""Optimized additive base construction via local search."""
import numpy as np
import random
import time

def _compute_rep_matrix(A, n=128):
    """Compute representation counts for sums 128..256 using A (as numpy array)."""
    a = np.asarray(A, dtype=np.int64)
    # All unordered pairs (i,j) with i<=j
    # Sums range from 2*min(a) to 2*max(a)
    # We only care about sums in [128, 256]
    # Use outer sum
    sums = a[:, None] + a[None, :]
    # We want unordered pairs: for i<j, count once; for i=j, count once
    # Representation count for sum s = number of pairs (i,j) with i<=j and a[i]+a[j]=s
    # Using full matrix, diagonal has i=j, off-diagonal has both (i,j) and (j,i)
    # So for off-diagonal, each unordered pair appears twice.
    # rep_count[s] = (number of (i,j) with i<=j and sum=s)
    # = (off_diag_count[s] + diag_count[s]) where we count each unordered pair once
    # Actually: total ordered pairs with sum s = T[s]. 
    #   diag contributes d[s] (where a[i]+a[i]=s, i.e., 2*a[i]=s)
    #   off-diag contributes T[s] - d[s] ordered pairs, which is 2 * (number of unordered off-diag pairs)
    #   So unordered off-diag pairs = (T[s] - d[s]) / 2
    #   Total unordered pairs (including equal) = d[s] + (T[s] - d[s])/2 = (T[s] + d[s])/2
    # But this must be integer. Let's just compute directly for small A.
    
    # For efficiency with small A (say up to ~50), just do it directly
    k = len(a)
    if k <= 200:
        reps = np.zeros(257, dtype=np.int64)
        for i in range(k):
            for j in range(i, k):
                s = a[i] + a[j]
                if 128 <= s <= 256:
                    reps[s] += 1
        return reps[128:257]  # length 129, index 0 -> sum 128
    else:
        # Use matrix approach
        T = np.zeros(513, dtype=np.int64)
        # Compute all sums
        for i in range(k):
            for j in range(k):
                s = a[i] + a[j]
                if 0 <= s <= 512:
                    T[s] += 1
        # Diagonal: 2*a[i] = s
        d = np.zeros(513, dtype=np.int64)
        for i in range(k):
            s = 2 * a[i]
            if 0 <= s <= 512:
                d[s] += 1
        reps = (T + d) // 2
        return reps[128:257]

def _is_covered(reps):
    """Check if all target sums have at least one representation."""
    return np.all(reps > 0)

def _peak(reps):
    """Maximum representation multiplicity."""
    return int(np.max(reps)) if len(reps) > 0 else 0

def _objective(A, n=128):
    """Return (is_valid, peak)."""
    reps = _compute_rep_matrix(A, n)
    valid = _is_covered(reps)
    pk = _peak(reps)
    return valid, pk

def _initial_solution():
    """Create an initial covering set."""
    # Start with a structured set that's known to cover
    # range(64, 129) works but is large. Let's try smaller.
    # A = {0, 1, 2, ..., k-1} union {64, 65, ..., 128} might work
    # Actually, let's start with the seed and then remove elements.
    A = set(range(64, 129))
    # Try to remove elements greedily
    removed = 0
    for x in sorted(A, reverse=True):
        A_test = A - {x}
        if len(A_test) < 2:
            continue
        reps = _compute_rep_matrix(list(A_test))
        if _is_covered(reps):
            A = A_test
            removed += 1
    return list(A)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Initial solution
    A = _initial_solution()
    if not A:
        A = list(range(64, 129))
    
    valid, peak = _objective(A)
    best_A = list(A)
    best_peak = peak if valid else 10**9
    
    # If initial isn't valid, try the seed
    if not valid:
        A = list(range(64, 129))
        valid, peak = _objective(A)
        best_A = list(A)
        best_peak = peak
    
    random.seed(42)
    
    # Simulated annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove an element
            if len(A) <= 2:
                T *= cooling
                continue
            x = random.choice(A)
            A_new = A - {x}
        elif move_type < 0.6:
            # Add an element
            candidates = [x for x in range(129) if x not in A]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            A_new = A | {x}
        else:
            # Swap: remove one, add one
            if len(A) <= 1:
                T *= cooling
                continue
            x_remove = random.choice(A)
            candidates = [x for x in range(129) if x not in A]
            if not candidates:
                T *= cooling
                continue
            x_add = random.choice(candidates)
            A_new = (A - {x_remove}) | {x_add}
        
        if not A_new:
            T *= cooling
            continue
        
        valid_new, peak_new = _objective(A_new)
        
        if valid_new:
            # Valid solution
            if peak_new < peak:
                # Accept (improvement)
                A = A_new
                peak = peak_new
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A)
            else:
                # Accept with probability exp(-(peak_new - peak)/T)
                delta = peak_new - peak
                if delta < 0 or random.random() < np.exp(-delta / T):
                    A = A_new
                    peak = peak_new
        else:
            # Invalid: accept with very low probability to escape
            if random.random() < np.exp(-5.0 / T):
                A = A_new
                peak = peak_new
        
        T *= cooling
    
    # Final polish: try to remove more elements
    for x in sorted(best_A, reverse=True):
        if time.time() > deadline - 2:
            break
        A_test = set(best_A) - {x}
        if len(A_test) < 2:
            continue
        reps = _compute_rep_matrix(list(A_test))
        if _is_covered(reps):
            pk = _peak(reps)
            if pk <= best_peak:
                best_A = list(A_test)
                if pk < best_peak:
                    best_peak = pk
    
    # Return the best valid solution found
    if best_peak < 10**9:
        return best_A
    else:
        # Fallback to seed
        return list(range(64, 129))
