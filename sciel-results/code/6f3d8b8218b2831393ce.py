import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    N = 128
    
    # Precompute: for each pair (i,j), what is i+j
    # We'll use bitsets (Python ints as bitsets)
    
    # Start with the known valid set: {65, ..., 128}
    base_set = set(range(65, 129))
    
    # Try to add elements from {1, ..., 64}
    # Check if adding x to current set S creates a violation
    # A violation exists if there exist distinct a,b,c in S such that a+b, a+c, b+c all in S
    
    def is_valid(S):
        """Check if set S is valid (no forbidden triple)."""
        S_list = list(S)
        S_set = set(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if a + b in S_set and a + c in S_set and b + c in S_set:
                        return False
        return True
    
    # More efficient check using bitsets
    def check_addition(S_bits, x):
        """Check if adding x to S (represented as bitset) creates a violation.
        Returns True if adding x is safe."""
        # After adding x, we need to check all triples involving x
        # Triple (x, a, b) for distinct a, b in S: need NOT (x+a in S∪{x} and x+b in S∪{x} and a+b in S∪{x})
        # Since x is new, x+a in S∪{x} means x+a in S or x+a == x (impossible for a>0)
        # So x+a in S∪{x} iff x+a in S (since x+a > x for a >= 1)
        # Similarly for x+b
        # a+b in S∪{x} iff a+b in S or a+b == x
        
        # Case 1: a+b != x, then we need NOT (x+a in S and x+b in S and a+b in S)
        # Case 2: a+b == x, then a+b is in S∪{x} (it's x), so we need NOT (x+a in S and x+b in S)
        
        # Actually let's just check: for all pairs (a,b) in S with a < b:
        #   After adding x, the triple (x,a,b) is forbidden if x+a, x+b, a+b are all in S∪{x}
        
        S_set = set()
        # Extract set from bitset for checking
        for i in range(1, N+1):
            if S_bits & (1 << (i-1)):
                S_set.add(i)
        
        # Also check triples where x is one of the three, and the other two are in S
        for a in S_set:
            for b in S_set:
                if a == b:
                    continue
                # Triple (x, a, b) - all distinct
                sa = x + a
                sb = x + b
                ab = a + b
                # Check if all three sums are in S ∪ {x}
                in_sa = (sa in S_set) or (sa == x)
                in_sb = (sb in S_set) or (sb == x)
                in_ab = (ab in S_set) or (ab == x)
                if in_sa and in_sb and in_ab:
                    return False
        
        # Also need to check: does adding x create a violation among existing elements?
        # No, because existing elements were already valid among themselves.
        # The only new triples are those involving x.
        
        return True
    
    # Greedy: try to add elements from {1,...,64} in some order
    best_set = set(base_set)
    best_size = len(best_set)
    
    # Try different orderings of adding small elements
    small_elements = list(range(1, 65))
    
    # Greedy addition
    current_set = set(base_set)
    current_bits = 0
    for i in range(65, 129):
        current_bits |= (1 << (i-1))
    
    added = set()
    for x in small_elements:
        if check_addition(current_bits, x):
            current_set.add(x)
            current_bits |= (1 << (x-1))
            added.add(x)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Try random restarts with different orders
    best_result = list(best_set)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(small_elements)
        current_set = set(base_set)
        current_bits = 0
        for i in range(65, 129):
            current_bits |= (1 << (i-1))
        
        for x in small_elements:
            if time.time() > deadline:
                break
            if check_addition(current_bits, x):
                current_set.add(x)
                current_bits |= (1 << (x-1))
        
        if len(current_set) > len(best_result):
            best_result = list(current_set)
    
    # Try local search: remove some elements, try to add others
    # ... (skip for now, keep it simple)
    
    # Verify the result
    if not is_valid(set(best_result)):
        # Fallback to base set
        best_result = list(range(65, 129))
    
    return best_result
