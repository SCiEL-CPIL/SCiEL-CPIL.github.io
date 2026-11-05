import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    
    best_triples = None
    best_count = 0
    
    def check_violation_on_add(triple, present):
        """Check if adding `triple` to `present` creates a K(2,2,2)."""
        a, b, c = triple
        # For the new triple (a,b,c), we need to check if there exist
        # a' != a, b' != b, c' != c such that all 8 transversals are present.
        # The 8 transversals for pairs (a,a'),(b,b'),(c,c') are:
        # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'),
        # (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
        # Since (a,b,c) is the new one, we need the other 7 to be in `present`.
        for a_prime in range(n):
            if a_prime == a:
                continue
            for b_prime in range(n):
                if b_prime == b:
                    continue
                for c_prime in range(n):
                    if c_prime == c:
                        continue
                    # Check all 7 other transversals
                    if ((a, b, c_prime) in present and
                        (a, b_prime, c) in present and
                        (a, b_prime, c_prime) in present and
                        (a_prime, b, c) in present and
                        (a_prime, b, c_prime) in present and
                        (a_prime, b_prime, c) in present and
                        (a_prime, b_prime, c_prime) in present):
                        return True
        return False
    
    def check_violation_on_remove(triple, present):
        """Check if removing `triple` from `present` eliminates all K(2,2,2) violations involving it.
        Actually, we just need to check if `triple` is part of any K(2,2,2) in `present`."""
        a, b, c = triple
        for a_prime in range(n):
            if a_prime == a:
                continue
            for b_prime in range(n):
                if b_prime == b:
                    continue
                for c_prime in range(n):
                    if c_prime == c:
                        continue
                    # Check all 8 transversals are in present
                    if ((a, b, c) in present and
                        (a, b, c_prime) in present and
                        (a, b_prime, c) in present and
                        (a, b_prime, c_prime) in present and
                        (a_prime, b, c) in present and
                        (a_prime, b, c_prime) in present and
                        (a_prime, b_prime, c) in present and
                        (a_prime, b_prime, c_prime) in present):
                        return True
        return False
    
    def count_violations(present):
        """Count number of K(2,2,2) configurations in present."""
        count = 0
        # For each set of 3 disjoint pairs, check if all 8 transversals are present
        # This is expensive, so we only use it for verification
        for a, a_prime in combinations(range(n), 2):
            for b, b_prime in combinations(range(n), 2):
                if b in (a, a_prime) or b_prime in (a, a_prime):
                    continue
                for c, c_prime in combinations(range(n), 2):
                    if c in (a, a_prime, b, b_prime) or c_prime in (a, a_prime, b, b_prime):
                        continue
                    # Check all 8 transversals
                    if ((a, b, c) in present and
                        (a, b, c_prime) in present and
                        (a, b_prime, c) in present and
                        (a, b_prime, c_prime) in present and
                        (a_prime, b, c) in present and
                        (a_prime, b, c_prime) in present and
                        (a_prime, b_prime, c) in present and
                        (a_prime, b_prime, c_prime) in present):
                        count += 1
        return count
    
    def is_valid(present):
        """Check if present has no K(2,2,2)."""
        a, b, c = None, None, None
        # Quick check: for each triple in present, see if it's part of a K(2,2,2)
        for triple in present:
            if check_violation_on_remove(triple, present):
                return False
        return True
    
    # Phase 1: Greedy construction
    def greedy_construct(seed):
        random.seed(seed)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        present = set()
        for t in shuffled:
            if not check_violation_on_add(t, present):
                present.add(t)
        return present
    
    # Phase 2: Local search
    def local_search(present, seed, max_iter=50000):
        random.seed(seed)
        current = set(present)
        current_count = len(current)
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if no_improve_count > 10000:
                break
            
            # Try to add a random missing triple
            missing = [t for t in all_triples if t not in current]
            if missing:
                t = random.choice(missing)
                if not check_violation_on_add(t, current):
                    current.add(t)
                    current_count += 1
                    no_improve_count = 0
                    continue
            
            # Try to remove a triple that's part of a violation (shouldn't happen if valid)
            # Instead, try shake: remove 3 random triples, then try to re-add
            if random.random() < 0.1:
                # Shake move
                to_remove = random.sample(list(current), min(3, len(current)))
                for t in to_remove:
                    current.discard(t)
                current_count = len(current)
                no_improve_count += 1
            
            no_improve_count += 1
        
        return current, current_count
    
    # Main loop with random restarts
    restart = 0
    while time.time() < deadline:
        # Greedy construction
        present = greedy_construct(restart)
        count = len(present)
        
        if count > best_count:
            best_count = count
            best_triples = set(present)
        
        # Local search
        improved_present, improved_count = local_search(present, restart + 10000, max_iter=30000)
        
        if improved_count > best_count:
            best_count = improved_count
            best_triples = set(improved_present)
        
        restart += 1
        if restart > 200:
            break
    
    # Verify best is valid
    if best_triples is not None and is_valid(best_triples):
        return [list(t) for t in best_triples]
    
    # Fallback: seed solution
    return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
