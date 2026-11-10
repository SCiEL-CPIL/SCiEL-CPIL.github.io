"""Optimized: Set-based local search with targeted 1-out/2-in moves."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = 85.0  # Safety margin

    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def greedy_init():
        """Standard greedy: iterate 2..128, add if no tail-product collision."""
        values = []
        products = set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def greedy_randomized(rng):
        """Randomized greedy: try values in random order."""
        values = []
        products = set()
        candidates = list(range(2, 129))
        rng.shuffle(candidates)
        for x in candidates:
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        return values

    # 1. Generate initial candidates
    best = greedy_init()
    best_len = len(best)
    
    # Try a few randomized greedy starts
    for seed in range(1, 11):
        if time.time() > deadline - 30:
            break
        rng_g = random.Random(seed)
        cand = greedy_randomized(rng_g)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)

    # 2. Local Search
    # We work with the set of elements and the sorted list.
    current_seq = best[:]
    current_set = set(current_seq)
    missing = [x for x in range(2, 129) if x not in current_set]
    
    # Helper to update state after a successful move
    def update_state(new_seq):
        nonlocal current_seq, current_set, missing
        current_seq = new_seq
        current_set = set(new_seq)
        missing = [x for x in range(2, 129) if x not in current_set]

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Greedy Insertion
        # Try to insert any missing number
        inserted_any = True
        while inserted_any:
            inserted_any = False
            if time.time() > deadline - 5:
                break
            for x in missing[:]: # Iterate over a copy
                new_seq = current_seq + [x]
                new_seq.sort()
                if is_valid(new_seq):
                    update_state(new_seq)
                    improved = True
                    inserted_any = True
                    break # Restart insertion loop with updated state
    
        # Phase 2: 1-out/2-in Search
        # If we are stuck (no insertions), try to swap 1 out for 2 in
        if not improved and time.time() < deadline - 5:
            attempts = 0
            max_attempts = 2000
            while attempts < max_attempts and time.time() < deadline - 5:
                attempts += 1
                if not current_seq or not missing:
                    break
                
                # Pick element to remove
                e = random.choice(current_seq)
                
                # Pick two elements to add
                if len(missing) < 2:
                    break
                x, y = random.sample(missing, 2)
                
                # Construct new set
                new_set = current_set - {e}
                new_set.add(x)
                new_set.add(y)
                
                # Sort and check
                new_seq = sorted(new_set)
                
                if is_valid(new_seq):
                    update_state(new_seq)
                    improved = True
                    break # Go back to Phase 1 to see if more insertions are possible

        # Phase 3: 1-out/1-in Shuffle (Exploration)
        # If 1-out/2-in failed, try to shuffle the set to escape local optimum
        if not improved and time.time() < deadline - 10:
            attempts = 0
            max_attempts = 1000
            while attempts < max_attempts and time.time() < deadline - 10:
                attempts += 1
                if not current_seq or not missing:
                    break
                
                e = random.choice(current_seq)
                x = random.choice(missing)
                
                if e == x:
                    continue
                    
                new_set = current_set - {e}
                new_set.add(x)
                
                new_seq = sorted(new_set)
                
                if is_valid(new_seq):
                    update_state(new_seq)
                    # We don't mark 'improved' as True here to allow further shuffling
                    # or we can break to re-evaluate. Let's break to re-evaluate insertions.
                    improved = True
                    break

    # Final check
    if not is_valid(current_seq):
        # Fallback to greedy if something went wrong
        current_seq = greedy_init()

    return current_seq

# EVOLVE-BLOCK-END
