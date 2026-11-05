"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # 1. Identify "Free" numbers: Primes in (128, 256]
    # These do not conflict with any other number in [2, 256]
    # Proof: For prime p > 128, lcm(p, b) > 256 for all b != p.
    free_primes = []
    for i in range(129, 257):
        if all(i % j != 0 for j in range(2, int(math.isqrt(i)) + 1)):
            free_primes.append(i)
    
    # 2. Remaining candidates R
    R = [i for i in range(2, 257) if i not in free_primes]
    
    # 3. Build conflict graph on R
    # Conflict if lcm(a, b) <= 256
    n = len(R)
    index_map = {val: idx for idx, val in enumerate(R)}
    # Precompute gcd or just use math.gcd
    conflicts = [set() for _ in range(n)]
    
    for i in range(n):
        a = R[i]
        for j in range(i + 1, n):
            b = R[j]
            # Optimization: if a*b > 256 * gcd(a,b) is hard to check directly without gcd
            # lcm = a*b//gcd(a,b)
            g = math.gcd(a, b)
            if (a * b) // g <= 256:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # 4. Solve Max Weight Independent Set on R
    # Weights are 1/x
    weights = [1.0 / R[i] for i in range(n)]
    
    best_set_indices = set()
    best_weight = 0.0
    
    # Heuristic 1: Greedy by Weight (Small numbers first)
    # Sort by weight descending (i.e., value ascending)
    order1 = sorted(range(n), key=lambda i: R[i])
    current_set = set()
    for i in order1:
        # Check if i conflicts with any in current_set
        if not current_set.intersection(conflicts[i]):
            current_set.add(i)
    
    w1 = sum(weights[i] for i in current_set)
    if w1 > best_weight:
        best_weight = w1
        best_set_indices = set(current_set)
    
    # Heuristic 2: Greedy by Weight/Degree (Efficiency)
    degrees = [len(conflicts[i]) for i in range(n)]
    # Avoid division by zero, though unlikely for isolated nodes
    eff_scores = [weights[i] / (degrees[i] + 1) for i in range(n)]
    order2 = sorted(range(n), key=lambda i: -eff_scores[i])
    current_set = set()
    for i in order2:
        if not current_set.intersection(conflicts[i]):
            current_set.add(i)
    
    w2 = sum(weights[i] for i in current_set)
    if w2 > best_weight:
        best_weight = w2
        best_set_indices = set(current_set)
        
    # Heuristic 3: Local Search / Hill Climbing
    # Start from best found so far
    current = set(best_set_indices)
    current_w = best_weight
    improved = True
    # Limit iterations to stay within time
    max_iter = 1000
    
    while improved and time.time() < deadline:
        improved = False
        # Try to add a node
        for i in range(n):
            if i not in current:
                if not current.intersection(conflicts[i]):
                    # Can add i
                    new_w = current_w + weights[i]
                    if new_w > current_w:
                        current.add(i)
                        current_w = new_w
                        improved = True
                        break # Restart loop as set changed
        # Try to swap one node out and two in? Or just swap 1 for 1?
        # Simple swap: remove i, add j
        # This is expensive. Let's stick to addition first.
        # If no addition possible, try swaps.
        if not improved:
            # Try swaps: remove i from current, add j not in current
            # Only if j doesn't conflict with current \ {i}
            for i in list(current):
                if time.time() > deadline: break
                for j in range(n):
                    if j in current: continue
                    # Check if j conflicts with current - {i}
                    # j conflicts with i? Doesn't matter if we remove i.
                    # j must not conflict with any k in current, k != i
                    conflict_found = False
                    for k in current:
                        if k == i: continue
                        if j in conflicts[k]:
                            conflict_found = True
                            break
                    if not conflict_found:
                        # Calculate new weight
                        new_w = current_w - weights[i] + weights[j]
                        if new_w > current_w:
                            current.remove(i)
                            current.add(j)
                            current_w = new_w
                            improved = True
                            break
                    if improved: break
                if improved: break

    # Final check for time
    if time.time() > deadline:
        # Return best found so far
        pass

    # Construct final list
    final_set = list(free_primes)
    for i in best_set_indices:
        final_set.append(R[i])
        
    return final_set

# EVOLVE-BLOCK-END
