import time
import random

START_TIME = time.time()
DEADLINE = 60.0  # seconds

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def can_add(A_sums, sum_to_card, c):
    """
    Check if c can be added to the set represented by A_sums and sum_to_card.
    A_sums: list of sets, A_sums[k] contains sums of subsets of size k.
    sum_to_card: dict mapping sum -> cardinality.
    c: candidate integer.
    Returns True if valid, False otherwise.
    """
    n = len(A_sums) - 1  # current size
    # We need to check if any new sum s+c (where s is in A_sums[k])
    # collides with an existing sum in sum_to_card with a different cardinality.
    
    for k in range(n + 1):
        for s in A_sums[k]:
            ns = s + c
            if ns in sum_to_card:
                # If the existing cardinality for this sum is not k+1, we have a conflict.
                # Because the new sum will have cardinality k+1.
                if sum_to_card[ns] != k + 1:
                    return False
    return True

def add_element(A_sums, sum_to_card, c):
    """
    Add c to the set, updating A_sums and sum_to_card.
    Assumes can_add returned True.
    """
    n = len(A_sums) - 1
    # Update in reverse to avoid using c multiple times
    # We need to create a new list of sets or update carefully.
    # Since we are adding to k+1 from k, and iterating k down, we can update in place?
    # No, A_sums[k+1] is being modified while we might be reading A_sums[k]?
    # We read A_sums[k] to generate sums for k+1.
    # If we iterate k from n down to 0:
    #   We read A_sums[k].
    #   We add to A_sums[k+1].
    #   A_sums[k+1] was already processed in previous iteration (k+1 > k).
    #   So reading A_sums[k] is safe.
    #   But we are modifying A_sums[k+1].
    #   Is A_sums[k+1] used as a source? Yes, in the next iteration k+1.
    #   But we are going DOWN. So when we are at k, we have already processed k+1.
    #   So A_sums[k+1] is not used as a source again.
    #   Wait. The source for new sums of size k+1 is the OLD A_sums[k].
    #   The destination is A_sums[k+1].
    #   If we modify A_sums[k+1], does it affect the source for any other k?
    #   Source for size j+1 is A_sums[j].
    #   We modify A_sums[j+1].
    #   So we modify the destination of j, which is the source for j-1? No.
    #   Source for j is A_sums[j-1].
    #   So modifying A_sums[k+1] does not affect A_sums[k] (source for k+1).
    #   It does not affect A_sums[k-1] (source for k).
    #   So in-place update iterating k from n down to 0 is safe.
    
    # However, we must be careful: A_sums[k] might be empty.
    for k in range(n, -1, -1):
        if not A_sums[k]:
            continue
        # Get a list of current sums to iterate, because A_sums[k] is not modified,
        # but A_sums[k+1] is. We are reading A_sums[k].
        # Wait, we are adding to A_sums[k+1]. We are reading A_sums[k].
        # These are different indices.
        for s in A_sums[k]:
            ns = s + c
            A_sums[k+1].add(ns)
            # Update sum_to_card
            # We know it's valid, so if it exists, it must be k+1.
            # If it doesn't exist, set to k+1.
            sum_to_card[ns] = k + 1
            
    # Note: A_sums[n+1] was created?
    # A_sums is a list. We need to ensure it has length n+2.
    # The caller should handle extending the list.

def greedy_search(order, max_time=5.0):
    """
    Greedily build a set using the given order of candidates.
    """
    # Initialize
    # A_sums[0] = {0}
    A_sums = [set() for _ in range(61)] # Max size 60
    A_sums[0].add(0)
    sum_to_card = {0: 0}
    current_size = 0
    used = set()
    
    for c in order:
        if is_done():
            break
        if c in used:
            continue
        
        # Check if we can add c
        if can_add(A_sums, sum_to_card, c):
            # Add it
            used.add(c)
            # Extend A_sums if needed
            if current_size + 1 >= len(A_sums):
                A_sums.append(set())
            
            add_element(A_sums, sum_to_card, c)
            current_size += 1
            
            # Optional: If we hit a high size, we might break early if time is tight?
            # No, try to maximize.
            
    return list(used)

def construct():
    # Baseline: Powers of 2
    # {1, 2, 4, 8, 16, 32, 64, 128} -> Size 8
    # Can we do better?
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Ascending
    if elapsed() < DEADLINE:
        order1 = list(range(1, 257))
        s1 = greedy_search(order1)
        if len(s1) > best_size:
            best_size = len(s1)
            best_set = s1
    
    # Strategy 2: Descending
    if elapsed() < DEADLINE - 5:
        order2 = list(range(256, 0, -1))
        s2 = greedy_search(order2)
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = s2
    
    # Strategy 3: Randomized restarts
    # Try to find a larger set by randomizing the order or skipping some small numbers?
    # Small numbers tend to create dense low sums.
    # Maybe a mix?
    
    for _ in range(100):
        if elapsed() > DEADLINE - 2:
            break
        
        # Random order
        order3 = list(range(1, 257))
        random.shuffle(order3)
        s3 = greedy_search(order3)
        if len(s3) > best_size:
            best_size = len(s3)
            best_set = s3
    
    # Ensure return type is list of ints
    result = [int(x) for x in best_set]
    return result

if __name__ == "__main__":
    # Quick self-test
    pass
