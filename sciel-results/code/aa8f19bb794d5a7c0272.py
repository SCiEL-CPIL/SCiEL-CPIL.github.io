import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    def is_valid_addition(S, x):
        """
        Check if adding x to S keeps the set valid.
        S: list of currently selected integers.
        x: candidate integer.
        Returns True if S U {x} is valid.
        """
        # 1. Check if x divides the sum of any 3 distinct elements in S
        #    The triple consists of 3 elements from S.
        #    The divisor is x.
        #    The triple elements must be distinct from x (they are, since x not in S).
        if len(S) >= 3:
            for t in combinations(S, 3):
                if sum(t) % x == 0:
                    return False
        
        # 2. Check if any a in S divides (x + b + c)
        #    The triple consists of x, b, c.
        #    b, c are distinct elements from S, and distinct from a.
        #    The divisor is a.
        #    x is distinct from a (x not in S).
        #    b, c are distinct from a (we exclude a from S for the pairs).
        #    b, c are distinct from each other.
        for a in S:
            # Create list of elements in S excluding a
            # We need pairs (b, c) from this list
            # Optimization: if S is small, list comprehension is fine.
            S_without_a = [v for v in S if v != a]
            if len(S_without_a) >= 2:
                for b, c in combinations(S_without_a, 2):
                    if (x + b + c) % a == 0:
                        return False
        return True

    selected = []
    
    # Greedy selection from largest to smallest
    # Large numbers are less likely to divide sums of other numbers
    # because the sum of 3 positive integers is usually smaller than the largest number
    # unless the numbers are very large.
    # Actually, if a > sum(b,c,d), then a cannot divide sum(b,c,d) (unless sum=0).
    # So large 'a' are safe as divisors.
    # But large 'a' as summands?
    # If we pick large numbers, their sums are large.
    # A small divisor 'a' might divide a large sum.
    # So small numbers are dangerous as divisors.
    # We want to avoid small numbers in S.
    # Picking large numbers first ensures we fill the set with "safe" divisors.
    
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(selected, x):
            selected.append(x)
            if len(selected) == 48:
                return selected

    # If we didn't reach 48, we could try a random restart or local search,
    # but given the time limit and the likelihood that the greedy works,
    # we return the best found so far.
    return selected
