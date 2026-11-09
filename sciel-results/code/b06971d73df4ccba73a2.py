"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
LCM = 30030
MAX_VAL = 1000000
N_DIV = len(DIVISORS)

def _solve_assignment(window_start, window_end):
    """
    Check if we can assign each divisor to a distinct integer in [window_start, window_end]
    such that the integer is divisible by the divisor.
    Returns a list of assigned integers corresponding to DIVISORS, or None if impossible.
    """
    # Precompute available numbers for each divisor
    # available[i] is a set of numbers in the window divisible by DIVISORS[i]
    available = [set() for _ in range(N_DIV)]
    for num in range(window_start, window_end + 1):
        for i, d in enumerate(DIVISORS):
            if num % d == 0:
                available[i].add(num)
                
    # If any divisor has no available numbers, fail immediately
    for i in range(N_DIV):
        if not available[i]:
            return None

    # Order divisors by number of options (MRV heuristic)
    # We need to track original index to return correct order
    order = sorted(range(N_DIV), key=lambda i: len(available[i]))
    
    assignment = [None] * N_DIV
    used_numbers = set()
    
    def backtrack(idx):
        if idx == N_DIV:
            return True
        
        div_idx = order[idx]
        d = DIVISORS[div_idx]
        
        # Try each available number for this divisor
        # To help pruning, we can iterate in any order
        for num in available[div_idx]:
            if num not in used_numbers:
                # Check if assigning this number doesn't make it impossible for others
                # Simple forward checking: ensure other unassigned divisors still have options
                # For small N, simple backtracking is usually fast enough.
                # But let's add a light check: if a future divisor has only 1 option left and it's this num, fail?
                # Actually, just checking used_numbers is often sufficient for small W.
                
                used_numbers.add(num)
                assignment[div_idx] = num
                
                if backtrack(idx + 1):
                    return True
                
                used_numbers.remove(num)
                assignment[div_idx] = None
        
        return False

    if backtrack(0):
        return assignment
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit for safety, leaving 10s for overhead
    
    # We search for the minimum window size W (span)
    # The span is max - min. So if window is [s, s+W], span is W.
    # We iterate W from 1 upwards.
    
    # Upper bound for W: The divisors are small, so a small window should suffice.
    # LCM is 30030. In any interval of length 30030, each divisor has ~10000 multiples.
    # With 10 divisors, we have plenty of room.
    # Let's try W up to 100 first. If not found, expand.
    
    max_w_to_try = 200
    
    for W in range(1, max_w_to_try + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # The window can start at any offset s in [0, LCM-1]
        # However, due to periodicity, we only need to check s in [0, LCM-1]
        # But actually, if a solution exists with span W, it exists in every period.
        # So checking s in [0, LCM-1] is sufficient.
        
        for s in range(LCM):
            if time.time() > deadline:
                break
                
            window_start = s
            window_end = s + W
            
            # We don't need to check windows that go beyond MAX_VAL if we care about absolute values,
            # but since the problem asks for integers <= 1000000, and LCM is small,
            # any solution in [0, LCM-1 + W] is valid as long as values <= 1000000.
            # Given LCM=30030 and W is small, s+W < 30030+200 << 1000000.
            
            res = _solve_assignment(window_start, window_end)
            if res is not None:
                # Verify distinctness and bounds (should be guaranteed by logic)
                # res contains the numbers assigned to DIVISORS[i]
                # Ensure all are distinct (backtracking ensures this)
                # Ensure all <= 1000000 (guaranteed by small W and s < LCM)
                
                # Convert to list of ints
                return [int(x) for x in res]
                
    # Fallback: If no small window found, use a heuristic or return a known valid solution
    # A valid solution is just the divisors themselves, but they might not be distinct or minimal span.
    # Actually, divisors are distinct. Span = 39 - 6 = 33.
    # Let's return the divisors as a safe fallback, although we expect to find better.
    return DIVISORS[:]

# EVOLVE-BLOCK-END
