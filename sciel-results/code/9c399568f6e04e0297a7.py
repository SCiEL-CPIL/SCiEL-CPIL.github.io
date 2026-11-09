import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n = 16
    vertices = list(range(n))
    all_triples = list(itertools.combinations(vertices, 3))
    
    # Precompute sets for faster membership? 
    # Using a set of tuples is standard and fast enough.
    
    def check_safe(T, current_set):
        a, b, c = T
        # We need to find a', b', c' such that 
        # {a', b, c}, {a, b', c}, {a, b, c'}, 
        # {a', b', c}, {a', b, c'}, {a, b', c'}, {a', b', c'}
        # are all in current_set.
        
        # a' cannot be a, b, c
        # b' cannot be a, b, c, a'
        # c' cannot be a, b, c, a', b'
        
        # Optimization: 
        # If {a', b, c} is not in current_set, we can skip this a'.
        # This is a huge pruning factor.
        
        for a_prime in vertices:
            if a_prime in (a, b, c):
                continue
            t1 = (a_prime, b, c)
            if t1 not in current_set:
                continue
            
            for b_prime in vertices:
                if b_prime in (a, b, c, a_prime):
                    continue
                t2 = (a, b_prime, c)
                if t2 not in current_set:
                    continue
                
                # Now we need c' such that:
                # {a, b, c'} in set
                # {a', b', c'} in set
                # {a', b, c'} in set
                # {a, b', c'} in set
                # {a', b', c} in set (this one does not depend on c')
                
                # Check {a', b', c} first, it's independent of c'
                t6 = (a_prime, b_prime, c)
                if t6 not in current_set:
                    continue
                
                # Now iterate c'
                for c_prime in vertices:
                    if c_prime in (a, b, c, a_prime, b_prime):
                        continue
                    
                    # Check remaining 4
                    if ((a, b, c_prime) in current_set and
                        (a_prime, b_prime, c_prime) in current_set and
                        (a_prime, b, c_prime) in current_set and
                        (a, b_prime, c_prime) in current_set):
                        return False
        return True

    best_system = []
    best_len = 0
    
    while time.time() < deadline:
        random.shuffle(all_triples)
        current_set = set()
        current_list = []
        
        for T in all_triples:
            if check_safe(T, current_set):
                current_set.add(T)
                current_list.append(T)
        
        if len(current_list) > best_len:
            best_len = len(current_list)
            best_system = current_list
            
        # If we reached a very high number, we might stop early?
        # Max possible is 560. But K(2,2,2) free systems are sparse.
        # Let's just run until deadline.

    # Convert tuples to lists for output
    return [list(t) for t in best_system]
