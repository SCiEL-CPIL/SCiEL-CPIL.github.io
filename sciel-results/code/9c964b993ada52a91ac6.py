"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s kill
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def conflicts_count(S, x):
        """Count conflicts if we add x to set S."""
        # S is a set
        # Conflict if: exists a,b in S with a+b=x
        #             exists a in S with x+a in S
        #             2x in S
        cnt = 0
        for a in S:
            if (x - a) in S:  # a + (x-a) = x, but we need both in S
                cnt += 1
            if (x + a) in S:  # x + a = c in S
                cnt += 1
        if (2 * x) in S:
            cnt += 1
        # But we're double counting: for a+b=x, both (a,b) and (b,a) counted
        # Actually for a+b=x: we count pairs (a, x-a) where both in S
        # For x+a=c: we count a where x+a in S
        # These are different types of conflicts
        return cnt
    
    def can_add(S, x):
        """Check if x can be added to S without creating conflicts."""
        for a in S:
            if (x - a) in S:
                return False
            if (x + a) in S:
                return False
        if (2 * x) in S:
            return False
        return True
    
    def greedy_add(S, candidates):
        """Greedily add candidates that don't create conflicts."""
        added = True
        while added:
            added = False
            for x in candidates[:]:
                if can_add(S, x):
                    S.add(x)
                    candidates.remove(x)
                    added = True
        return S
    
    def local_search(S, all_elements, time_limit):
        """Local search with swaps."""
        S = set(S)
        candidates = [x for x in all_elements if x not in S]
        
        # First, greedily add all possible
        S = greedy_add(S, candidates)
        
        # Now try swaps
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_swap = None
            best_score = len(S)
            
            for x in list(S):
                S.remove(x)
                # Try adding each candidate
                for y in candidates:
                    if can_add(S, y):
                        new_size = len(S) + 1
                        if new_size > best_score:
                            best_score = new_size
                            best_swap = (x, y)
                S.add(x)
            
            if best_swap:
                x_out, y_in = best_swap
                S.remove(x_out)
                S.add(y_in)
                candidates.remove(y_in)
                candidates.append(x_out)
                improved = True
        
        return S
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Return best we have so far, fill rest with odd numbers
            inst = instance(size, seed)
            results.append([x for x in inst if x % 2])
            continue
        
        inst = instance(size, seed)
        # Start with odd numbers
        S = set(x for x in inst if x % 2)
        all_elements = inst
        
        # Local search
        S = local_search(S, all_elements, DEADLINE)
        
        results.append(sorted(S))
    
    return results
# EVOLVE-BLOCK-END
