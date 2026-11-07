import time
from itertools import combinations

def construct():
    # Primes and their powers in 2..40
    # 2: 2, 4, 8, 16, 32
    # 3: 3, 9, 27
    # 5: 5, 25
    # 7: 7
    # 11, 13, 17, 19, 23, 29, 31, 37
    
    # Decoupled optimal sets for each prime
    # p=2: {2, 4, 16} (exps 1, 2, 4)
    # p=3: {3, 9} (exps 1, 2)
    # p=5: {5, 25} (exps 1, 2)
    # p=7: {7}
    # others: {p}
    
    base_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify base set
    def is_valid(S):
        prods = set()
        n = len(S)
        # Use bitmasks for subset enumeration
        for mask in range(1 << n):
            prod = 1
            for i in range(n):
                if mask & (1 << i):
                    prod *= S[i]
            if prod in prods:
                return False
            prods.add(prod)
        return True

    if not is_valid(base_set):
        # Fallback to primes if base set fails (should not happen)
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    current_best = base_set[:]
    
    # Try to add remaining numbers
    all_numbers = list(range(2, 41))
    candidates = [x for x in all_numbers if x not in current_best]
    
    # Greedy addition
    for x in candidates:
        new_set = current_best + [x]
        if is_valid(new_set):
            current_best = new_set
            
    return current_best
