"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses for 1 to N
    # inv[x] is the unique y in 1..N such that x*y % MOD == 1
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # For an interval [s, e] to be valid: for all x in [s, e], inv[x] < s or inv[x] > e.
    # This means inv[x] must NOT be in [s, e].
    #
    # Key insight: For a fixed start s, the maximum valid end is:
    #   max_e(s) = min over all x in [s, N] with inv[x] >= s of (inv[x] - 1)
    #   but we also need e >= s, and we must handle self-inverses (inv[x]==x).
    #
    # More precisely: [s, e] is valid iff for all x in [s, e], inv[x] not in [s, e].
    # This is equivalent to: e < min{ inv[x] : x in [s, N], inv[x] >= s }
    # (if such x exists; otherwise e can go up to N)
    #
    # We use a sweep from right to left to compute, for each s, the minimum inv[x]
    # among x >= s with inv[x] >= s.
    
    # Self-inverse check: if inv[s] == s, then s cannot be in any valid interval.
    # Mark these as forbidden.
    forbidden = [False] * (N + 1)
    for x in range(1, N + 1):
        if inv[x] == x:
            forbidden[x] = True
    
    # For each position s, compute the maximum valid end.
    # We need: for all x in [s, e], inv[x] < s or inv[x] > e.
    # Rearranging: e must be < inv[x] for all x in [s, e] with inv[x] >= s.
    # So max_e(s) = min{ inv[x] - 1 : x in [s, N], inv[x] >= s } (if set non-empty), else N.
    # But we also need to ensure no forbidden elements in [s, e].
    
    # Efficient computation:
    # For each x, it constrains intervals [s, e] where s <= x <= e and inv[x] >= s and inv[x] <= e.
    # The constraint is: if s <= x and inv[x] >= s, then e < inv[x].
    # So for a given s, the constraint from x (where x >= s and inv[x] >= s) is e <= inv[x] - 1.
    # The tightest constraint is the minimum inv[x] - 1 over all such x.
    
    # We can compute this with a sweep from right to left.
    # For each s, we need min{ inv[x] : x in [s, N], inv[x] >= s }.
    
    # Let's precompute: for each s, what is the minimum inv[x] for x >= s with inv[x] >= s?
    # This is a range minimum query with a condition. We can do it with a segment tree
    # or simply by observation that N=1008 is small.
    
    # Precompute max_valid_end[s] = maximum e such that [s,e] is valid.
    # Using the insight: [s,e] valid iff for all x in [s,e], inv[x] < s or inv[x] > e.
    # 
    # For fixed s, define f(s) = min{ inv[x] : x in [s, N], inv[x] >= s }.
    # Then max_valid_end(s) = f(s) - 1 (if f(s) exists), else N.
    # But we also need to exclude forbidden positions.
    #
    # Actually, let's just use the two-pointer correctly. The key fix:
    # The two-pointer with monotone right boundary IS correct.
    # The condition to extend is: inv[e+1] not in [s, e+1].
    # This is: inv[e+1] < s or inv[e+1] > e+1.
    # Since inv is a permutation and e+1 >= s, inv[e+1] > e+1 means inv[e+1] > e+1.
    # inv[e+1] < s means inv[e+1] is to the left of the window.
    
    best_start = 1
    best_length = 0
    e = 0  # right boundary, non-decreasing
    
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Ensure e >= s
        if e < s:
            e = s
        
        # If s is self-inverse, no valid interval starts at s
        if forbidden[s]:
            continue
        
        # Expand e as far as possible while [s, e] remains valid
        while e < N:
            next_e = e + 1
            # [s, next_e] is valid iff [s, e] was valid AND inv[next_e] not in [s, next_e]
            # Since [s, e] is valid by invariant, we just check the new element.
            if inv[next_e] < s or inv[next_e] > next_e:
                e = next_e
            else:
                break
        
        # Verify [s, e] is actually valid (safety check for small N)
        # For N=1008, this O(N) check per s gives O(N^2) = ~1M ops, fine.
        valid = True
        for x in range(s, e + 1):
            if s <= inv[x] <= e:
                valid = False
                break
        
        if valid and (e - s + 1) > best_length:
            best_length = e - s + 1
            best_start = s
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
