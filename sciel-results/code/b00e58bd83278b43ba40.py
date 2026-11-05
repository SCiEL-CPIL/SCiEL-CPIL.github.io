import time
import random
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121" # 12, 21. Length 3. Greedy: 1,2 -> 12. 2,1 -> 21. Overlap 1. 121.

    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # O[i][j] = max k in 0..n-1 such that perms[i][-k:] == perms[j][:k]
    # Using a list of lists for speed
    O = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: We can compute this faster.
    # For each pair (i, j), check k from n-1 down to 1.
    # But 5040^2 * 7 is ~175M ops. Might be tight in Python.
    # Let's try to optimize the inner loop.
    
    # Alternative: Use string conversion? No, tuple comparison is fast.
    # Let's just do it. If it's too slow, we'll see.
    # Actually, we can reduce work.
    # perms[i] is a tuple.
    
    # Let's try a vectorized approach or just efficient loops.
    # For n=7, 5040 is small enough that a tight loop might pass.
    # 5040 * 5040 = 25,401,600 pairs.
    # Inner loop up to 6 comparisons.
    # ~150M comparisons. In Python, this could take 10-30 seconds.
    # We have 115s. This should be fine.
    
    for i in range(num_perms):
        p1 = perms[i]
        # We only need to compute O[i][j] for all j.
        # Can we do better?
        # Not easily without complex data structures.
        
    # Let's do the precomputation in a way that allows early termination or better caching?
    # No, let's just run it.
    
    for i in range(num_perms):
        p1 = perms[i]
        for j in range(num_perms):
            p2 = perms[j]
            max_k = n - 1
            # Check from largest overlap to smallest
            for k in range(max_k, 0, -1):
                if p1[n - k:] == p2[:k]:
                    O[i][j] = k
                    break
    
    # Greedy Initialization
    # Start with a random permutation
    order = [random.randint(0, num_perms - 1)]
    remaining = set(range(num_perms))
    remaining.remove(order[0])
    
    current_last = order[0]
    
    while remaining:
        # Find the next perm with max overlap
        best_next = -1
        best_ov = -1
        # To avoid O(N^2) every step, we can just pick randomly from top few?
        # Or just scan all. N=5040. 5040 * 5040 / 2 operations total for init.
        # This is fine.
        
        # Actually, scanning all remaining every step is O(N^2).
        # 5040 * 5040 / 2 ~ 12M ops. Fine.
        
        for j in remaining:
            ov = O[current_last][j]
            if ov > best_ov:
                best_ov = ov
                best_next = j
            elif ov == best_ov:
                # Tie break randomly to avoid bias
                if random.random() < 0.5:
                    best_next = j
        
        order.append(best_next)
        remaining.remove(best_next)
        current_last = best_next
        
    # Calculate initial total overlap
    total_overlap = 0
    for i in range(len(order) - 1):
        total_overlap += O[order[i]][order[i + 1]]
        
    best_order = order[:]
    best_overlap = total_overlap
    
    # Simulated Annealing
    # We maximize total_overlap.
    # Energy E = -total_overlap.
    # Delta E = -delta_overlap.
    # Accept if delta_overlap > 0 or exp(delta_overlap / T) > random()
    
    T = 10.0  # Initial temperature
    cooling = 0.999
    
    # Deadline check inside loop
    # Estimate steps per second.
    # Each step: 3 lookups, 3 additions, 1 random, 1 exp (sometimes).
    # Very fast. We can do many steps.
    
    while time.time() < deadline:
        if T < 0.01:
            # Restart with lower T? Or just stop?
            # Let's just keep going with min T or restart.
            # Restarting might help escape deep local minima.
            # But for now, just continue.
            pass
            
        # Pick random index for adjacent swap
        i = random.randint(0, len(order) - 2)
        
        a = order[i]
        b = order[i + 1]
        
        # Calculate delta overlap
        # Old: O[order[i-1]][a] + O[a][b] + O[b][order[i+2]]
        # New: O[order[i-1]][b] + O[b][a] + O[a][order[i+2]]
        
        delta = 0
        
        if i > 0:
            left = order[i - 1]
            delta += O[left][b] - O[left][a]
            
        delta += O[b][a] - O[a][b]
        
        if i + 2 < len(order):
            right = order[i + 2]
            delta += O[a][right] - O[b][right]
            
        if delta > 0:
            # Always accept improvement
            order[i] = b
            order[i + 1] = a
            total_overlap += delta
            
            if total_overlap > best_overlap:
                best_overlap = total_overlap
                best_order = order[:]
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            if random.random() < math.exp(delta / T):
                order[i] = b
                order[i + 1] = a
                total_overlap += delta
                
        T *= cooling
        
    # Construct final word from best_order
    word = list(perms[best_order[0]])
    
    for i in range(len(best_order) - 1):
        curr = perms[best_order[i]]
        next_p = perms[best_order[i + 1]]
        
        ov = O[best_order[i]][best_order[i + 1]]
        
        # Append non-overlapping part
        word += list(next_p[ov:])
        
    return "".join(map(str, word))
