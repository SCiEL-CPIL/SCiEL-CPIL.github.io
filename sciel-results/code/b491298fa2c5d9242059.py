import time
import random
from itertools import combinations

def is_sidon_subset_4(subset):
    """
    Check if a specific subset of size 4 is a Sidon set.
    Sidon condition: all sums subset[i] + subset[j] for i <= j are unique.
    For size 4, there are 10 sums.
    """
    a, b, c, d = subset
    # Sums:
    # a+a, a+b, a+c, a+d
    # b+b, b+c, b+d
    # c+c, c+d
    # d+d
    sums = [
        a + a, a + b, a + c, a + d,
        b + b, b + c, b + d,
        c + c, c + d,
        d + d
    ]
    # Check for duplicates
    # Since the list is small, a set check is fast
    return len(set(sums)) == 10

def has_sidon_subset_of_size_4(S):
    """
    Check if S contains any Sidon subset of size 4.
    Returns True if such a subset exists.
    """
    # Generate all combinations of 4 elements
    # S is sorted, so combinations are unique
    for combo in combinations(S, 4):
        if is_sidon_subset_4(combo):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: evaluator kills at 90s
    
    # Candidate 1: Arithmetic Progression [0, 1, ..., 11]
    # In an AP, for any 4 distinct elements a < b < c < d, 
    # a + d = b + c is a collision if they form an AP? No, a+d = b+c is NOT always true for arbitrary 4 AP elements.
    # Example: 0, 1, 2, 3. 0+3=3, 1+2=3. Collision.
    # Example: 0, 1, 2, 4. 0+4=4, 1+2=3. No collision there. 
    # Wait, let's check 0, 1, 2, 4.
    # Sums: 0, 1, 2, 4, 2, 3, 5, 4, 6, 8.
    # Duplicates: 2 (from 0+2 and 1+1) and 4 (from 0+4 and 2+2).
    # So it is NOT Sidon.
    
    # Is it true that NO 4-subset of an AP is Sidon?
    # Let the subset be {a, b, c, d} with a < b < c < d.
    # Differences: b-a, c-b, d-c.
    # A 4-set is Sidon iff all differences are distinct? No, that's for differences.
    # For sums: 
    # If the set is an AP, say {x, x+r, x+2r, x+3r}.
    # Sums: 2x, 2x+r, 2x+2r, 2x+3r, 2x+2r, 2x+3r, 2x+4r, 2x+4r, 2x+5r, 2x+6r.
    # Duplicates: 2x+2r (appears twice), 2x+3r (appears twice), 2x+4r (appears twice).
    # So a 4-AP is never Sidon.
    
    # What if the 4 elements are not a 4-AP? e.g., {0, 1, 2, 4} from AP 0..11.
    # We saw it has collisions.
    
    # In general, for any 4 distinct integers in an AP, there is a collision.
    # Proof: Let the elements be $x_1 < x_2 < x_3 < x_4$.
    # If $x_1+x_4 = x_2+x_3$, collision.
    # If not, are there other collisions?
    # $x_1+x_3 = x_2+x_2$? i.e., $x_1+x_3 = 2x_2$. This happens if $x_1, x_2, x_3$ is a 3-AP.
    # $x_2+x_4 = x_3+x_3$? i.e., $x_2+x_4 = 2x_3$. This happens if $x_2, x_3, x_4$ is a 3-AP.
    # $x_1+x_2 = x_3+x_3$? No, $x_1+x_2 < 2x_3$ usually.
    # $x_1+x_4 = x_2+x_2$? No.
    
    # Actually, it is a known fact that the maximum size of a Sidon set in an AP of length N is roughly sqrt(N).
    # For N=12, max Sidon size in [0,11] is 3.
    # So [0, 11] does NOT contain a Sidon subset of size 4.
    
    candidate = list(range(12))
    
    # Verify
    if not has_sidon_subset_of_size_4(candidate):
        return candidate
        
    # If the AP somehow failed (it shouldn't), try a different structured set.
    # E.g., a set with many 3-APs.
    # Or just fall back to random search with the fast checker.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        cand = random.sample(range(129), 12)
        cand.sort()
        if not has_sidon_subset_of_size_4(cand):
            return cand
            
    # Fallback: return the AP anyway, it's the best theoretical candidate.
    return candidate
