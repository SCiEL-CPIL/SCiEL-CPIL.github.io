"""
Maximize the number of distinct d>0 such that a, a+d, a+2d all lie in a 32-element subset of [1,512].

Strategy:
- Fast scoring via bitset convolution (numpy FFT) to count 3-APs quickly.
- Actually, we want the number of DISTINCT d values, not the total number of 3-APs.
  So we need to track which d values appear.
- Use a bitset representation: for a given set S (as a bitmask over 512 positions),
  for each d in 1..255, check if there exists a such that a, a+d, a+2d are all in S.
  This can be done with bit operations: (mask & (mask >> d) & (mask >> 2*d)) != 0.
  But we need to be careful about bounds.
  
  Actually, let's use a different approach: represent the set as a boolean array of length 512.
  For each d, compute the number of a such that a, a+d, a+2d are all in the set.
  This is a convolution-like operation.
  
  Let's use numpy for speed:
  - Create a boolean array b of length 512 where b[i] = 1 if i+1 is in the set (using 0-indexed).
  - For a given d, the condition is: b[a-1] & b[a+d-1] & b[a+2d-1] for some a.
  - We can compute this as: for each a, check if all three are set.
  - To do this fast, we can use the fact that this is equivalent to checking if the convolution
    of b with itself has a specific pattern, but it's not exactly a simple convolution.
    
  Alternative: for each d, shift the bitset by d and by 2d, and AND them with the original.
  If the result is non-zero, then d is a valid progression difference.
  
  Let's use Python integers as bitsets for efficiency, since 512 bits fits in a Python int.
  For a set S, let mask be an integer where bit i is 1 if (i+1) is in S (0-indexed, so bit 0 = value 1).
  
  For a given d:
  - mask_shifted_d = mask >> d
  - mask_shifted_2d = mask >> (2*d)
  - If (mask & mask_shifted_d & mask_shifted_2d) != 0, then d is valid.
  
  But we need to ensure that a, a+d, a+2d are all in [1, 512].
  Since mask only has bits 0..511 set (for values 1..512), and shifting right by d will clear
  the lower d bits, the AND operation naturally handles the bounds:
  - a must be such that a-1 >= 0 (always true for a>=1)
  - a+d-1 >= 0 (always true)
  - a+2d-1 >= 0 (always true)
  - a-1 <= 511 => a <= 512
  - a+d-1 <= 511 => a <= 512-d
  - a+2d-1 <= 511 => a <= 512-2d
  
  The bit operations handle this correctly because:
  - mask has bits 0..511
  - mask >> d has bits 0..511-d (bits 511-d+1..511 are cleared)
  - mask >> 2d has bits 0..511-2d
  - The AND will only have bits set where all three have bits set, which automatically
    ensures a+2d <= 512 (i.e., a <= 512-2d).
  
  So the check is: (mask & (mask >> d) & (mask >> 2*d)) != 0
  
  We need to count the number of d in 1..255 for which this is true.

- Search: use simulated annealing or local search with this fast scoring.
  The scoring is O(255) per evaluation, which is very fast.
  We can do many evaluations per second.

Let's implement:
1. Fast scoring function using bitset.
2. Local search with random swaps, using the fast scorer.
3. Multiple random restarts.
"""
import time
import random

N = 32
MAX_VAL = 512

def _score_mask(mask):
    """
    Count distinct d>0 such that exists a with a, a+d, a+2d in the set.
    mask: integer where bit i (0-indexed) is 1 if value (i+1) is in the set.
    """
    count = 0
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are all in the set
        # Using bit operations:
        # mask has bits 0..511 for values 1..512
        # mask >> d shifts right by d, so bit i in (mask >> d) corresponds to value i+d+1 in original
        # Actually, let's be precise:
        # If value v is in the set, bit (v-1) is set in mask.
        # For a 3-AP (a, a+d, a+2d):
        #   a is in set => bit (a-1) set
        #   a+d is in set => bit (a+d-1) set
        #   a+2d is in set => bit (a+2d-1) set
        # We want to check if there exists a such that all three bits are set.
        # Let's define:
        #   mask_a: bits where a-1 is set, i.e., mask itself
        #   mask_ad: bits where a+d-1 is set. This is mask shifted left by d? No.
        #   If bit (a+d-1) is set in mask, then in (mask >> d), bit (a-1) is set.
        #   Because (mask >> d) has bit i set iff mask has bit (i+d) set.
        #   So (mask >> d) has bit (a-1) set iff mask has bit (a-1+d) = (a+d-1) set.
        #   Similarly, (mask >> 2d) has bit (a-1) set iff mask has bit (a-1+2d) = (a+2d-1) set.
        # Therefore, (mask & (mask >> d) & (mask >> 2d)) has bit (a-1) set iff a, a+d, a+2d are all in the set.
        # So we just need to check if this AND is non-zero.
        if (mask & (mask >> d) & (mask >> 2*d)) != 0:
            count += 1
    return count

def _set_to_mask(s):
    """Convert a set of values (1..512) to a bitmask."""
    mask = 0
    for v in s:
        mask |= (1 << (v-1))
    return mask

def _mask_to_set(mask):
    """Convert a bitmask to a set of values (1..512)."""
    s = set()
    for i in range(512):
        if (mask >> i) & 1:
            s.add(i+1)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    best_mask = 0
    best_score = 0
    
    # Try multiple random restarts with local search
    for trial in range(500):  # up to 500 restarts, but time-limited
        if time.time() > deadline:
            break
        
        # Random initial set
        current_set = set(random.sample(range(1, MAX_VAL+1), N))
        current_mask = _set_to_mask(current_set)
        current_score = _score_mask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
        
        # Local search: random swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try a batch of random swaps
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                # Pick a random element to remove
                x = random.choice(list(current_set))
                # Pick a random element to add (not in current_set)
                y = random.randint(1, MAX_VAL)
                if y in current_set:
                    continue
                
                # Try the swap
                current_set.remove(x)
                current_set.add(y)
                new_mask = _set_to_mask(current_set)
                new_score = _score_mask(new_mask)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = new_mask
                else:
                    # Revert
                    current_set.remove(y)
                    current_set.add(x)
        
        # If we found a perfect score (max possible is 255), we can stop early
        if best_score >= 255:
            break
    
    return sorted(_mask_to_set(best_mask))

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Score: {_score_mask(_set_to_mask(set(result)))}")
    print(f"Length: {len(result)}")
    print(f"Distinct: {len(set(result)) == 32}")
    print(f"All in range: {all(1 <= x <= 512 for x in result)}")
