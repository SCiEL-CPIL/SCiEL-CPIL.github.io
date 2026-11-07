import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # Numbers 2 to 40
    candidates = list(range(2, 41))
    
    # Precompute prime factorization vectors for quick check?
    # Actually, just using integer products is fast enough for small k.
    
    def get_subset_products(S):
        """
        Returns a set of all subset products of S.
        S is a list of ints.
        """
        prods = {1}
        for x in S:
            # Add x * p for all p in current prods
            # Be careful not to modify prods while iterating
            new_prods = {x * p for p in prods}
            prods.update(new_prods)
        return prods

    def is_valid(S):
        """
        Checks if all subset products are distinct.
        """
        k = len(S)
        prods = get_subset_products(S)
        return len(prods) == (1 << k)

    # Initial set: All primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = primes[:]
    best_size = len(best_set)
    
    # Verify initial set
    if not is_valid(best_set):
        # Should not happen, but fallback
        best_set = []
        best_size = 0

    current_set = best_set[:]
    
    # Local Search
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # 1. Try to add a random element
        if len(current_set) < 40:
            others = [x for x in candidates if x not in current_set]
            if others:
                # Try a few random additions
                for _ in range(10):
                    if time.time() > deadline: break
                    cand = random.choice(others)
                    new_set = current_set + [cand]
                    if is_valid(new_set):
                        current_set = new_set
                        if len(current_set) > best_size:
                            best_set = current_set[:]
                            best_size = len(current_set)
                            no_improve_steps = 0
                        break
                    else:
                        # Try to swap
                        if current_set:
                            swap_idx = random.randint(0, len(current_set) - 1)
                            swap_elem = current_set[swap_idx]
                            # Try swapping cand with swap_elem
                            new_set = [x for i, x in enumerate(current_set) if i != swap_idx] + [cand]
                            if is_valid(new_set):
                                current_set = new_set
                                if len(current_set) > best_size:
                                    best_set = current_set[:]
                                    best_size = len(current_set)
                                    no_improve_steps = 0
                                break
        
        # 2. If no improvement, try to escape local minima
        # Randomly remove an element and see if we can add a better one later
        # Or just random restart of the local search phase if stuck
        
        no_improve_steps += 1
        if no_improve_steps > max_no_improve:
            # Reset current set to best found and try different randomization
            current_set = best_set[:]
            no_improve_steps = 0
            # Shuffle candidates order implicitly by random choice
            
    # Final check and return
    # Ensure types are correct
    return [int(x) for x in best_set]
