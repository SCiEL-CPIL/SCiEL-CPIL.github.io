import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Find a large subset S of {1..128} such that there do not exist three distinct
    members a,b,c with a+b, a+c, b+c all in S.
    
    Strategy:
    1. Start with a good initial set (upper half is sum-free, so no two small elements sum to a large one in the set... actually let's think differently).
    
    Actually, the constraint is: no three distinct a,b,c in S such that a+b, a+c, b+c are ALL in S.
    
    A simple approach: take the upper half {65..128}. This is sum-free (sum of any two is >128, so not in the set). Thus no pair sums are in S at all, so the condition is vacuously satisfied. Size = 64.
    
    Can we do better? We can try to add some smaller numbers.
    
    Let's use a greedy/local search approach:
    1. Start with a candidate set.
    2. Check all triples for violations.
    3. Try to add elements from the complement, or remove conflicting elements.
    
    Better approach: Use a randomized local search.
    - State: a set S (represented as a bitset or boolean array of size 128).
    - Objective: maximize |S| subject to no violating triple.
    - Move: add a random element from complement, or remove a random element.
    - Accept if the move doesn't create a violation (for adds), or always accept removes (to escape local optima).
    
    Actually, let's think about the structure more carefully.
    
    If S contains a,b,c distinct, and a+b, a+c, b+c are all in S, that's a violation.
    
    Note that if we take S to be a sum-free set, then no two elements sum to another element in S, so the condition is automatically satisfied. The largest sum-free subset of {1..n} has size ceil(n/2). For n=128, that's 64.
    
    But we can do better than sum-free. We just need to avoid the specific configuration of 3 elements whose 3 pair sums are also in the set.
    
    Let's try a different approach: use a SAT-like or constraint-based search, or a greedy algorithm with backtracking.
    
    Given the time limit (90s), let's use a randomized local search with simulated annealing or simple hill-climbing.
    
    Representation: S as a frozenset or a boolean list of length 129 (index 0 unused).
    
    Violation check: for each triple (a,b,c) in S with a<b<c, check if a+b, a+c, b+c are all in S.
    
    To make this efficient, we can precompute for each pair (a,b) in S whether a+b is in S.
    
    Let's implement a local search:
    1. Initialize S as a random subset or the upper half.
    2. Compute all violations.
    3. While not at deadline:
       a. If no violations, try to add a random element from complement. If it creates a violation, skip.
       b. If violations exist, try to remove one of the elements involved in a violation.
    
    Actually, let's use a more systematic approach.
    
    Key insight: The condition involves 6 numbers: a, b, c, a+b, a+c, b+c. Note that a+b > c (assuming a<b<c), so the sums are larger than the elements.
    
    Let's try to build S from large to small.
    
    Alternative: Use the fact that if we partition {1..128} into two sets, one "small" and one "large", and ensure that the small set is sum-free with respect to the large set...
    
    Let me just implement a solid local search.
    """
    deadline = time.time() + 80  # Leave 10s margin
    
    n = 128
    all_elems = set(range(1, n+1))
    
    def is_valid(S):
        """Check if S has no violating triple."""
        S_list = sorted(S)
        S_set = set(S_list)
        # For each triple a < b < c in S
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k in range(j+1, len(S_list)):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac in S_set and bc in S_set:
                        return False
        return True
    
    def count_violations(S):
        """Return number of violating triples."""
        S_set = set(S)
        count = 0
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k in range(j+1, len(S_list)):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac in S_set and bc in S_set:
                        count += 1
        return count
    
    def get_violation_elements(S):
        """Return a set of elements that are part of at least one violation."""
        S_set = set(S)
        S_list = sorted(S)
        violators = set()
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k in range(j+1, len(S_list)):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac in S_set and bc in S_set:
                        violators.update([a, b, c, ab, ac, bc])
        return violators
    
    # Start with a good initial set: upper half
    S = set(range(n//2 + 1, n + 1))  # {65..128}, size 64
    
    best_S = set(S)
    best_size = len(S)
    
    # Try to improve via local search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Check current validity
        if not is_valid(S):
            # Remove a random violator
            violators = get_violation_elements(S)
            if violators:
                v = rng.choice(list(violators))
                S.discard(v)
            else:
                # Should not happen if is_valid returned False
                break
            continue
        
        # S is valid, try to add elements
        complement = all_elems - S
        if not complement:
            break
        
        # Try adding a random element
        added = False
        for _ in range(min(len(complement), 20)):  # Try a few random additions
            elem = rng.choice(list(complement))
            new_S = S | {elem}
            if is_valid(new_S):
                S = new_S
                added = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                complement = all_elems - S
                break
        
        if not added:
            # Try a different strategy: remove a random element and see if we can add two
            if len(S) > 10:
                elem_to_remove = rng.choice(list(S))
                S.discard(elem_to_remove)
                # Now try to add two elements
                complement = all_elems - S
                added_any = False
                for _ in range(min(len(complement), 30)):
                    elem = rng.choice(list(complement))
                    new_S = S | {elem}
                    if is_valid(new_S):
                        S = new_S
                        added_any = True
                        complement = all_elems - S
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        break
                if not added_any:
                    # Put the element back
                    S.add(elem_to_remove)
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
