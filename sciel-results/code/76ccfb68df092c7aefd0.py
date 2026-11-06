"""
Program for large integer sets avoiding equal-gcd triples in [1,128].
Strategy: Deterministic Multi-Pass Greedy Construction with Conflict Minimization.
Instead of random SA, we use a structured approach:
1. Precompute conflict data (which pairs of existing elements block a new element).
2. Iteratively add the element that creates the fewest new conflicts (or zero).
3. Use local search to swap out high-conflict elements for low-conflict ones if it increases size or maintains size with better structure.
4. The core insight is that the problem is a 3-uniform hypergraph independent set problem.
   We approximate the greedy choice by minimizing the "degree" of the new node in the conflict hypergraph restricted to the current set.
"""

import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin from 90s limit

    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = math.gcd(elements[i], elements[j])

    # Precompute bad_k[i][j] as a bitmask of k's such that {i, j, k} is a bad triple.
    # bad_k[i][j] is symmetric in i, j? No, the set of k is the same, so yes, bad_k[i][j] == bad_k[j][i].
    # We store it for all i != j.
    bad_k = [[0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            g = gcd_mat[i][j]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                if gcd_mat[i][k] == g and gcd_mat[j][k] == g:
                    mask |= (1 << k)
            bad_k[i][j] = mask

    # Helper: Count the number of bad triples formed by adding x to a set S (represented as mask).
    # A triple {x, a, b} is bad if a, b in S and b is in bad_k[x][a] (and a in bad_k[x][b]).
    # As derived before, count = sum_{a in S} popcount(bad_k[x][a] & S) // 2
    def count_conflicts_add(x, S_mask):
        total = 0
        temp = S_mask
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            total += (bad_k[x][a] & S_mask).bit_count()
            temp ^= lsb
        return total // 2

    # Helper: Count the number of bad triples removed by removing x from a set S (x in S).
    # This is the number of bad triples {x, a, b} with a, b in S\{x}.
    # Same formula: sum_{a in S\{x}} popcount(bad_k[x][a] & (S\{x})) // 2
    def count_conflicts_remove(x, S_mask):
        # S_mask contains x
        S_without_x = S_mask & ~(1 << x)
        total = 0
        temp = S_without_x
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            total += (bad_k[x][a] & S_without_x).bit_count()
            temp ^= lsb
        return total // 2

    # Initial State: Empty set
    current_mask = 0
    current_energy = 0
    current_size = 0
    
    best_mask = 0
    best_size = 0

    # Phase 1: Greedy Construction
    # We try to add elements one by one.
    # To be effective, we should add elements that "fit" well.
    # Heuristic: Add elements with low "potential conflict" with the current set.
    # Since we start empty, we can add anything.
    # Let's use a random restart strategy for the greedy order to find a good local optimum,
    # but keep it deterministic-ish or use a fixed seed for reproducibility in this specific run?
    # The task allows random. Let's use a few random orders and pick the best.
    
    max_restarts = 5
    best_global_mask = 0
    best_global_size = 0

    for restart in range(max_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        rng = random.Random(1234 + restart)
        order = list(range(N))
        rng.shuffle(order)
        
        curr_mask = 0
        curr_size = 0
        
        # Greedy Add
        for x in order:
            # Check if adding x is valid (creates 0 conflicts)
            # If not, we skip it for now? Or do we try to swap?
            # Standard greedy for independent set: add if valid.
            conflicts = count_conflicts_add(x, curr_mask)
            if conflicts == 0:
                curr_mask |= (1 << x)
                curr_size += 1
        
        # Phase 2: Local Search / Improvement
        # Try to increase size by swapping or adding.
        # Since we only added valid elements, curr_mask is valid (energy 0).
        # We try to add any remaining element that fits.
        changed = True
        while changed and time.time() < DEADLINE - 5:
            changed = False
            # Find an element x NOT in curr_mask such that count_conflicts_add(x, curr_mask) == 0
            for x in range(N):
                if not (curr_mask & (1 << x)):
                    if count_conflicts_add(x, curr_mask) == 0:
                        curr_mask |= (1 << x)
                        curr_size += 1
                        changed = True
                        break # Restart loop to check new opportunities
            
            if not changed:
                # No simple additions possible. Try swaps.
                # Swap i (in set) with j (not in set) if it keeps validity.
                # Actually, if we swap, we remove i and add j.
                # We need to check if the new set is valid.
                # Since we are in a local optimum for additions, we might need to allow
                # temporary invalid states? No, let's stick to valid moves for now.
                # A valid swap: Remove i, Add j.
                # The new set must have 0 conflicts.
                # This is hard to check O(1).
                # However, if we just try random swaps and verify, it might be slow.
                # Alternative: Try to remove i that is "critical" and see if it allows adding multiple j's?
                # For now, let's just do a few random swap attempts to break ties or find better structures.
                
                # Try a few random swaps
                for _ in range(20):
                    if time.time() > DEADLINE - 5:
                        break
                    # Pick i in set
                    set_bits = [i for i in range(N) if curr_mask & (1 << i)]
                    if not set_bits:
                        break
                    i = rng.choice(set_bits)
                    # Pick j not in set
                    not_set_bits = [j for j in range(N) if not (curr_mask & (1 << j))]
                    if not not_set_bits:
                        break
                    j = rng.choice(not_set_bits)
                    
                    # Test swap: new_mask = curr_mask ^ (1<<i) ^ (1<<j)
                    new_mask = curr_mask ^ (1 << i) ^ (1 << j)
                    
                    # Check validity of new_mask
                    # We can check energy quickly?
                    # Energy of new_mask = Energy(curr_mask) - Conflicts(i, curr_mask\{i}) + Conflicts(j, curr_mask\{i})
                    # Wait, Conflicts(j, curr_mask\{i}) is the number of bad triples involving j and pairs in curr_mask\{i}.
                    # If this is 0, and we removed i, the rest of the set was valid, so the new set is valid.
                    # Is that true?
                    # Old set S was valid.
                    # New set S' = S \ {i} U {j}.
                    # Triples in S' are either:
                    # 1. In S \ {i} (valid, since S was valid)
                    # 2. Involve j and two elements from S \ {i}.
                    # So S' is valid iff count_conflicts_add(j, S \ {i}) == 0.
                    
                    S_minus_i = curr_mask & ~(1 << i)
                    if count_conflicts_add(j, S_minus_i) == 0:
                        # Valid swap
                        curr_mask = new_mask
                        # Size stays same, but structure changed.
                        # This might allow future additions.
                        changed = True
                        break

        if curr_size > best_global_size:
            best_global_size = curr_size
            best_global_mask = curr_mask

    # Final Check: If we found nothing, fallback
    if best_global_size == 0:
        # Fallback: Powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128]

    # Convert best_global_mask to list
    result = []
    for i in range(N):
        if best_global_mask & (1 << i):
            result.append(elements[i])
            
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Final size: {len(res)}")
    print(res)
