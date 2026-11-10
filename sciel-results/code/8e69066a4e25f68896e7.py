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
    
    # Mark self-inverse residues (x such that x*x % MOD == 1)
    # These are 1 and MOD-1 (1008)
    forbidden = [False] * (N + 1)
    for x in range(1, N + 1):
        if inv[x] == x:
            forbidden[x] = True
    
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
        # (Since s is in the interval, s*s % MOD == 1 would be a violation)
        if forbidden[s]:
            continue
        
        # Expand e as far as possible while [s, e] remains valid
        while e < N:
            next_e = e + 1
            # [s, next_e] is valid iff [s, e] was valid AND inv[next_e] not in [s, next_e]
            # Since [s, e] is valid by invariant, we just check the new element.
            # inv[next_e] not in [s, next_e] is equivalent to:
            # inv[next_e] < s OR inv[next_e] > next_e
            if inv[next_e] < s or inv[next_e] > next_e:
                e = next_e
            else:
                break
        
        # The interval [s, e] is now the maximum valid interval starting at s
        if e >= s:
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
