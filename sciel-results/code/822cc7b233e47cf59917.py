import time
import sys
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 256
    
    # Precompute all 3-APs: for each (a, b, c) where a < b < c and b-a = c-b
    # Store as a list of tuples, and also build conflict maps
    ap_triples = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = b + d
            if c <= N:
                ap_triples.append((a, b, c))
    
    # For each element x, precompute which pairs (y, z) with y < z would form a 3-AP with x
    # i.e., x is the middle: y = x-d, z = x+d
    # x is the first: x, x+d, x+2d
    # x is the last: x-2d, x-d, x
    
    # Better: for each x, store the set of pairs (y,z) such that {x,y,z} forms a 3-AP
    # But that's expensive. Instead, use the conflict_count approach but optimize.
    
    # Let's use a different approach: maintain in_set as a list, and for checking if x can be added,
    # iterate over all existing elements y and check if adding x would create a 3-AP.
    # A 3-AP exists if there's y in S such that:
    #   - 2x - y is in S (x is middle or end)
    #   - (x + y) / 2 is in S and integer (x is end, y is start)
    #   - 2y - x is in S (x is start)
    
    # Actually, the cleanest check: x can be added iff for all y in S, 
    # the third element to form a 3-AP with {x, y} is not in S.
    # The three possible thirds for pair (x, y):
    #   If x < y: thirds are 2y-x (if <=N), 2x-y (if >=1), (x+y)/2 (if integer)
    #   If x > y: same as above with roles swapped
    
    def can_add(x, in_set):
        """Check if x can be added to the set defined by in_set without creating a 3-AP."""
        for y in range(1, N + 1):
            if not in_set[y]:
                continue
            # Check three possible thirds
            # Case 1: x, y are endpoints, middle is (x+y)/2
            if (x + y) % 2 == 0:
                mid = (x + y) // 2
                if 1 <= mid <= N and in_set[mid]:
                    return False
            # Case 2: x is first, y is second, third is 2y-x
            third = 2 * y - x
            if 1 <= third <= N and in_set[third]:
                return False
            # Case 3: y is first, x is second, third is 2x-y
            third = 2 * x - y
            if 1 <= third <= N and in_set[third]:
                return False
        return True
    
    def greedy_fill(in_set, start_order=None):
        """Greedily add elements in the given order (or 1..N) if possible."""
        if start_order is None:
            start_order = list(range(1, N + 1))
        for x in start_order:
            if not in_set[x] and can_add(x, in_set):
                in_set[x] = True
        return in_set
    
    def score(in_set):
        return sum(1.0 / x for x in range(1, N + 1) if in_set[x])
    
    best_set = None
    best_score = 0.0
    
    # Strategy 1: Greedy from 1 to N
    in_set = [False] * (N + 1)
    greedy_fill(in_set)
    s = score(in_set)
    if s > best_score:
        best_score = s
        best_set = in_set[:]
    
    # Strategy 2: Randomized greedy with different orderings
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Create a random ordering biased towards small numbers
        order = list(range(1, N + 1))
        # Shuffle but keep small numbers somewhat preferred
        # Use exponential distribution to weight small numbers more
        weights = [1.0 / i for i in range(1, N + 1)]
        random.shuffle(order)
        
        in_set = [False] * (N + 1)
        greedy_fill(in_set, order)
        s = score(in_set)
        if s > best_score:
            best_score = s
            best_set = in_set[:]
    
    # Strategy 3: Local search from best_set
    # Try swap moves: remove one element, try to add another
    in_set = best_set[:]
    cur_score = best_score
    
    improved = True
    round_count = 0
    while improved and time.time() < deadline - 3:
        improved = False
        round_count += 1
        
        # Try all swap moves: remove x, add y
        # To speed up, only try y < some threshold or y that are "attractive"
        for x in range(1, N + 1):
            if time.time() > deadline - 3:
                break
            if not in_set[x]:
                continue
            
            # Remove x
            in_set[x] = False
            
            # Try to add each y not in set
            for y in range(1, N + 1):
                if in_set[y]:
                    continue
                if can_add(y, in_set):
                    in_set[y] = True
                    new_score = cur_score - 1.0/x + 1.0/y
                    if new_score > cur_score + 1e-12:
                        cur_score = new_score
                        best_score = cur_score
                        best_set = in_set[:]
                        improved = True
                        break
                    else:
                        in_set[y] = False
            
            # If we didn't add anything, restore x
            if not any(in_set[y] for y in range(1, N+1) if y != x and in_set[y]):
                # This check is wrong, let's just restore x if we didn't improve
                pass
            
            # Actually, we need to be more careful. Let's restructure.
        
        # The above is getting complex. Let's use a cleaner approach.
        break  # Skip complex local search for now, rely on random restarts
    
    # Strategy 4: More aggressive random restarts with simulated annealing-like acceptance
    for trial in range(500):
        if time.time() > deadline - 2:
            break
        
        # Start from a random subset of best_set, then greedily fill
        in_set = [False] * (N + 1)
        
        # With some probability, start from best_set and randomly remove some elements
        if trial % 3 == 0:
            for x in range(1, N + 1):
                if best_set[x] and random.random() < 0.1:
                    in_set[x] = True
            greedy_fill(in_set)
        else:
            # Random start
            order = list(range(1, N + 1))
            random.shuffle(order)
            greedy_fill(in_set, order)
        
        s = score(in_set)
        if s > best_score:
            best_score = s
            best_set = in_set[:]
    
    return [x for x in range(1, N + 1) if best_set[x]]
