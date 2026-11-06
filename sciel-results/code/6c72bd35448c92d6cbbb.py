"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter to only divisors <= target
    divisors = [d for d in divisors if d <= target]
    
    # Sort divisors in descending order for deterministic tie-breaking if needed, 
    # but for BFS we just need them all.
    # Let's sort ascending to make "distinct" constraint easier? No, distinct is handled by bitmask or set.
    # Since 5040 has 60 divisors, bitmask is 2^60 which is too large for memory if we store full states.
    # But we can store (remaining_sum, frozenset_of_used_indices) or similar?
    # No, 2^60 is huge.
    
    # Wait, the state space is (remaining_sum, set_of_used_divisors).
    # The number of divisors is 60.
    # However, we can use a dictionary: state -> (steps, path)
    # Key: (remaining_sum, frozenset_of_indices)
    # This might still be large.
    
    # Alternative: BFS where state is just (remaining_sum, used_mask) is too big.
    # But note: we only care about the *count* of divisors.
    # Can we do BFS on (remaining_sum, used_mask) but prune aggressively?
    
    # Let's look at the size. 60 divisors.
    # Actually, a standard BFS on (remaining, used_set) might blow up memory.
    
    # Let's try a different approach: Meet in the Middle?
    # Or, since the target is 4979 and divisors sum to something larger, maybe we can use
    # a dictionary mapping (remaining_sum, used_mask) -> path.
    
    # Let's check the number of divisors of 5040.
    # 5040 = 2^4 * 3^2 * 5 * 7
    # Num divisors = (4+1)*(2+1)*(1+1)*(1+1) = 5*3*2*2 = 60.
    
    # 2^60 states is too many for a full BFS.
    # But we are looking for the *shortest* path.
    # Most paths will be short (4-6 terms).
    # The branching factor is 60.
    # Depth 4: 60^4 = 12,960,000. This is manageable.
    # Depth 5: 60^5 = 777,600,000. This is getting large but might be okay in Python if optimized?
    # Probably too slow in pure Python.
    
    # Let's use a BFS that stores (remaining_sum, used_mask) in a dictionary.
    # We can prune if we've seen a state with same remaining_sum and subset of used_mask?
    # No, used_mask must be exact for "distinct" constraint.
    
    # However, we can observe that if we reach the same remaining_sum with the same set of used divisors,
    # we don't need to revisit.
    
    # Let's try BFS with state = (remaining, used_mask).
    # used_mask is an integer (bitmask).
    # We store dist[state] = path.
    
    # To save memory, we can store the path as a tuple of indices.
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Map divisor value to index for bitmask
    # Sort divisors to have consistent indices
    divisors.sort()
    d_to_idx = {d: i for i, d in enumerate(divisors)}
    num_divs = len(divisors)
    
    # BFS
    # State: (remaining_sum, used_mask)
    # Queue: deque of (remaining_sum, used_mask, path_tuple)
    
    # Initial state: (target, 0, ())
    # Target state: remaining_sum == 0
    
    # Optimization: We can use a dictionary `seen` to store the shortest path length for a state?
    # Or just mark visited states. Since BFS guarantees shortest path in terms of edges (divisors added),
    # the first time we reach a state is the optimal path length.
    
    visited = set()
    queue = deque()
    queue.append((target, 0, ()))
    visited.add((target, 0))
    
    # To avoid huge memory usage, we might need to be careful.
    # But depth is expected to be small.
    
    while queue:
        if time.time() > deadline:
            break
            
        remaining, mask, path = queue.popleft()
        
        if remaining == 0:
            # Found a solution. Convert indices to values
            result = [divisors[i] for i in path]
            return result
            
        # Try adding each divisor
        for i in range(num_divs):
            if mask & (1 << i):
                continue  # Already used
            
            d = divisors[i]
            if d > remaining:
                continue  # Cannot use if divisor is larger than remaining
            
            new_remaining = remaining - d
            new_mask = mask | (1 << i)
            new_path = path + (i,)
            
            state = (new_remaining, new_mask)
            if state not in visited:
                visited.add(state)
                queue.append((new_remaining, new_mask, new_path))
                
    # Fallback if BFS didn't find it in time (unlikely for this size)
    # Greedy fallback
    remainder = target
    chosen = []
    # Use divisors in descending order for greedy
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
            
    if remainder == 0:
        return chosen
    else:
        # Return empty or best effort? Task requires exact sum.
        # If we can't find it, we might fail. But BFS should find it.
        return []

# EVOLVE-BLOCK-END
