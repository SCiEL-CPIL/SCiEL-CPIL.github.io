"""Size-efficient sorting networks on 17 wires via bit-vector search."""
import time
import random

N = 17
ALL_BITS = (1 << N) - 1
DEADLINE = 85.0  # safety margin before 90s kill
START_TIME = time.time()

# Precompute masks for comparators
# comp_mask[i][j] for i < j: mask for wire i and wire j
# We'll use a different representation: for each comparator (i,j),
# we precompute the operation as a function on bit-vectors

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def apply_comp(v, i, j):
    """Apply comparator (i,j) to bit-vector v. Returns new bit-vector."""
    bi = (v >> i) & 1
    bj = (v >> j) & 1
    if bi > bj:
        # swap: set i to 0, j to 1
        v &= ~(1 << i)
        v |= (1 << j)
    return v

def is_sorted(v):
    """Check if bit-vector is sorted (all 1s before all 0s, i.e., non-increasing).
    Actually for sorting network: output should have all 1s on top (higher wires) 
    or all 0s on top? Let's think: compare-exchange puts min on i, max on j.
    So after full sort, wire 0 has min, wire N-1 has max.
    For 0-1 inputs, sorted means: some number of 0s followed by 1s.
    So v should be of form: 0...0 1...1
    Check: for all i < j, if bit i is 1 then bit j must be 1.
    Equivalent: the bit-vector, when read from wire 0 to N-1, is non-decreasing.
    """
    # Find the position where 1s start
    # A sorted 0-1 vector has form: 0^k 1^(N-k)
    # So v & ((1 << N) - 1) should equal ((1 << N) - 1) ^ ((1 << k) - 1) for some k
    # Simpler: check that no 1 appears before a 0
    # i.e., once we see a 1, all subsequent bits must be 1
    seen_one = False
    for i in range(N):
        if (v >> i) & 1:
            seen_one = True
        elif seen_one:
            return False
    return True

def verify_network(comps):
    """Verify that the network sorts all 2^N inputs using 0-1 principle.
    Returns True if valid."""
    for v in range(1 << N):
        cur = v
        for (i, j) in comps:
            cur = apply_comp(cur, i, j)
        if not is_sorted(cur):
            return False
    return True

def verify_network_fast(comps):
    """Faster verification: for each input, simulate and check sorted."""
    n_comps = len(comps)
    for v in range(1 << N):
        cur = v
        for k in range(n_comps):
            i, j = comps[k]
            bi = (cur >> i) & 1
            bj = (cur >> j) & 1
            if bi > bj:
                cur &= ~(1 << i)
                cur |= (1 << j)
        # Check sorted
        # Quick check: count trailing zeros and see if rest are all 1s
        if cur == 0 or cur == ALL_BITS:
            continue
        # Find lowest set bit
        lowbit = cur & (-cur)
        k = lowbit.bit_length() - 1  # index of lowest set bit
        # All bits from k to N-1 should be 1
        expected = ALL_BITS ^ ((1 << k) - 1)
        if cur != expected:
            return False
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps):
    """Try to remove each comparator if the network remains valid.
    Returns a possibly smaller valid network."""
    n = len(comps)
    # Try removing each comparator, starting from the end
    for idx in range(n - 1, -1, -1):
        if _time_left() < 2.0:
            break
        candidate = comps[:idx] + comps[idx+1:]
        if verify_network_fast(candidate):
            comps = candidate
            # After removal, we might be able to remove more, restart scan
            return greedy_remove(comps)
    return comps

def random_mutation(comps, rng):
    """Randomly modify the network: add, remove, or swap comparators."""
    n = len(comps)
    op = rng.choice(['remove', 'add', 'swap', 'move'])
    new_comps = list(comps)
    
    if op == 'remove' and n > 0:
        idx = rng.randrange(n)
        new_comps.pop(idx)
    elif op == 'add':
        i = rng.randrange(N)
        j = rng.randrange(N)
        if i > j:
            i, j = j, i
        if i != j:
            pos = rng.randrange(n + 1)
            new_comps.insert(pos, (i, j))
    elif op == 'swap' and n >= 2:
        i1 = rng.randrange(n)
        i2 = rng.randrange(n)
        new_comps[i1], new_comps[i2] = new_comps[i2], new_comps[i1]
    elif op == 'move' and n >= 1:
        i1 = rng.randrange(n)
        i2 = rng.randrange(n)
        elem = new_comps.pop(i1)
        new_comps.insert(i2, elem)
    
    return new_comps

def hill_climbing(comps, rng, max_iters):
    """Hill climbing: try mutations, keep if valid and smaller or same size."""
    best = list(comps)
    best_len = len(best)
    iters = 0
    
    while iters < max_iters and _time_left() > 3.0:
        iters += 1
        mutated = random_mutation(best, rng)
        
        # Don't accept if larger
        if len(mutated) > best_len:
            continue
        
        if verify_network_fast(mutated):
            if len(mutated) < best_len:
                best = mutated
                best_len = len(best)
                # After improvement, do greedy removal
                best = greedy_remove(best)
                best_len = len(best)
    
    return best

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    # Initial size: N phases * ~8 comparators per phase = ~136
    print(f"Seed network: {len(comps)} comparators", flush=True)
    
    # First pass: greedy removal
    comps = greedy_remove(comps)
    print(f"After greedy removal: {len(comps)} comparators", flush=True)
    
    # Hill climbing with multiple restarts
    best_overall = list(comps)
    best_len = len(best_overall)
    
    restarts = 0
    while _time_left() > 10.0:
        restarts += 1
        # Mutate from best
        current = list(best_overall)
        # Add some random perturbations to escape local minima
        for _ in range(rng.randint(1, 5)):
            current = random_mutation(current, rng)
        
        # Do hill climbing
        result = hill_climbing(current, rng, max_iters=2000)
        
        if len(result) < best_len:
            best_overall = result
            best_len = len(result)
            print(f"Restart {restarts}: improved to {best_len} comparators", flush=True)
        elif _time_left() < 15.0:
            break
    
    # Final greedy pass
    best_overall = greedy_remove(best_overall)
    print(f"Final: {len(best_overall)} comparators", flush=True)
    
    return best_overall
