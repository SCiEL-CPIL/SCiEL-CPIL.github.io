"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 256
    
    # 1. Precompute prime factors and compatibility
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    # Build conflict graph using bitsets
    # conflict[i] is a bitset of all j that conflict with i (including i itself)
    conflict = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if i == j:
                conflict[i] |= (1 << i)
            elif prime_factors[i] & prime_factors[j]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # weights[i] = i (the value of including integer i)
    weights = list(range(N + 1))
    
    # 2. Initial solution: greedy by descending weight
    # Start with empty, add numbers in descending order if compatible
    initial_set = []
    current_mask = 0
    for num in range(N, 0, -1):
        # Check if num is compatible with all in current_set
        # A number is compatible if it's not in conflict with any selected number
        # conflict[num] has bits set for all numbers that conflict with num
        # If (current_mask & conflict[num]) == 0, then num is compatible
        if (current_mask & conflict[num]) == 0:
            initial_set.append(num)
            current_mask |= (1 << num)
    
    best_sum = sum(initial_set)
    best_set = list(initial_set)
    
    # 3. Branch and Bound for Maximum Weight Independent Set
    # State: (available_mask, current_sum, current_set_bits)
    # We use a recursive B&B with memoization on the available mask
    
    # Upper bound: sum of weights of all available vertices
    # This is a valid but loose bound. We can improve it.
    
    # Better upper bound: greedy fractional packing
    # For each vertex v in available, its "fraction" is 1 / (1 + degree_in_available(v))
    # Actually, a tighter bound: for each connected component, use max weight or sum of weights
    # Simplest tight bound: sum of all available weights (since we can take at most all of them)
    
    # Let's use a recursive approach with pruning
    # Sort vertices by weight descending for better pruning
    
    # Precompute sorted order
    sorted_vertices = list(range(N, 0, -1))  # Already descending by weight
    
    # B&B state: (avail_mask, current_sum)
    # We try to add vertices in some order
    
    # Use iterative deepening or just standard B&B
    # Given N=256, we need to be smart about pruning
    
    # Key insight: The conflict graph is sparse. Most numbers are coprime.
    # The maximum independent set should be large.
    
    # Let's use a different approach: 
    # The problem is equivalent to selecting a set of integers such that no two share a prime factor.
    # This is a packing problem over primes.
    
    # Alternative: Use the fact that each integer can be represented by its set of prime factors.
    # Two integers conflict iff their prime factor sets intersect.
    
    # Let's try a smarter B&B with better bounds.
    
    # For the upper bound, we can use:
    # For each prime p, let S_p be the set of available vertices divisible by p.
    # We can select at most one from each "conflict class" but classes overlap.
    
    # Simpler bound: sum of weights of available vertices.
    # Tighter: for each vertex v, it conflicts with conflict[v] & avail_mask.
    # The maximum weight independent set in a graph is at most the sum of weights.
    
    # Let's implement a standard B&B with the following strategy:
    # - Order vertices by descending weight
    # - At each step, either include the next vertex (if compatible) or exclude it
    # - Prune if current_sum + upper_bound <= best_sum
    
    # To make this efficient, we'll use bit operations.
    
    # Precompute for each vertex v, the mask of vertices that can be added with v
    # compatible_with[v] = ~conflict[v] & full_mask (excluding v itself)
    
    full_mask = (1 << (N + 1)) - 1
    
    # Let's use a different formulation.
    # Since the graph is sparse, we can use a branching strategy.
    
    # Actually, let's try a local search approach with simulated annealing or tabu search.
    # The initial greedy solution is already good. We can improve it with swaps.
    
    # Local Search: 1-for-1 and 2-for-2 swaps
    
    def get_sum_from_mask(mask):
        """Calculate sum of weights for vertices in mask."""
        total = 0
        m = mask
        while m:
            # Get lowest set bit
            lowbit = m & (-m)
            idx = lowbit.bit_length() - 1
            total += weights[idx]
            m ^= lowbit
        return total
    
    current_mask = 0
    for num in initial_set:
        current_mask |= (1 << num)
    current_sum = best_sum
    
    # Local search with multiple passes
    improved = True
    pass_count = 0
    max_passes = 1000
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Try 1-for-0 insertions
        for num in range(N, 0, -1):
            if current_mask & (1 << num):
                continue  # Already in set
            # Check if num is compatible with all in current set
            if (current_mask & conflict[num]) == 0:
                current_mask |= (1 << num)
                current_sum += num
                improved = True
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = [i for i in range(N + 1) if current_mask & (1 << i)]
        
        # Try 1-for-1 swaps
        if not improved:
            for num_out in range(1, N + 1):
                if not (current_mask & (1 << num_out)):
                    continue  # Not in set, can't remove
                # Try to replace num_out with a larger number
                for num_in in range(N, num_out, -1):
                    if current_mask & (1 << num_in):
                        continue  # Already in set
                    # Check if num_in is compatible with all in set except num_out
                    mask_without_out = current_mask & ~(1 << num_out)
                    if (mask_without_out & conflict[num_in]) == 0:
                        # Perform swap
                        current_mask = mask_without_out | (1 << num_in)
                        current_sum = current_sum - num_out + num_in
                        improved = True
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = [i for i in range(N + 1) if current_mask & (1 << i)]
                        break  # Restart after a swap
        
        # Try 2-for-1 swaps (remove 2, add 1) - usually not helpful for maximizing sum
        # Try 1-for-2 swaps (remove 1, add 2) - can be helpful
        if not improved:
            for num_out in range(1, N + 1):
                if not (current_mask & (1 << num_out)):
                    continue
                mask_without_out = current_mask & ~(1 << num_out)
                # Try to add two numbers that are both compatible with mask_without_out
                # and with each other
                added_any = False
                for num_in1 in range(N, num_out, -1):
                    if current_mask & (1 << num_in1):
                        continue
                    if (mask_without_out & conflict[num_in1]) != 0:
                        continue
                    # Check if we can add another number with num_in1
                    mask_with_in1 = mask_without_out | (1 << num_in1)
                    for num_in2 in range(num_in1 - 1, num_out, -1):
                        if current_mask & (1 << num_in2):
                            continue
                        if (mask_with_in1 & conflict[num_in2]) != 0:
                            continue
                        # Both num_in1 and num_in2 are compatible
                        new_sum = current_sum - num_out + num_in1 + num_in2
                        if new_sum > current_sum:
                            current_mask = mask_with_in1 | (1 << num_in2)
                            current_sum = new_sum
                            improved = True
                            added_any = True
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_set = [i for i in range(N + 1) if current_mask & (1 << i)]
                            break
                    if added_any:
                        break
                if added_any:
                    break
    
    # Convert mask to list
    final_set = [i for i in range(N + 1) if current_mask & (1 << i)]
    final_set = [int(x) for x in final_set]
    
    return final_set

# EVOLVE-BLOCK-END
