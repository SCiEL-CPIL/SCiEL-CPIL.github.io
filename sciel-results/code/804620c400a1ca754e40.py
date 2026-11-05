"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    def has_3ap(S):
        """Check if set S has any 3-term arithmetic progression."""
        s = set(S)
        for a in s:
            for b in s:
                if a < b:
                    c = 2*b - a
                    if c in s:
                        return True
        return False
    
    def reciprocal_sum(S):
        """Compute exact reciprocal sum as Fraction."""
        total = Fraction(0, 1)
        for x in S:
            total += Fraction(1, x)
        return total
    
    def valid_addition(S, candidate):
        """Check if adding candidate to S creates a 3-term AP."""
        if candidate in S:
            return False
        s = set(S)
        # Check all pairs with candidate as middle: a, candidate, c where a+c=2*candidate
        for a in s:
            c = 2*candidate - a
            if c in s and c != candidate and c > 0 and c <= 256:
                return False
        # Check all pairs where candidate is first or third
        for a in s:
            for b in s:
                if a < b:
                    c = 2*b - a
                    if c == candidate:
                        return False
                    if c in s and c != a and c != b:
                        pass  # already in S, no new AP from this pair
        # More careful: candidate as third term: a, b, candidate where a+candidate=2b
        for b in s:
            a = 2*b - candidate
            if a in s and a != candidate and a > 0:
                return False
        # candidate as first term: candidate, b, c where candidate+c=2b
        for b in s:
            c = 2*b - candidate
            if c in s and c != candidate and c > 0 and c <= 256:
                return False
        return True
    
    def greedy_initial():
        """Greedy: add elements in order of decreasing 1/x."""
        selected = []
        for x in range(1, 257):
            if valid_addition(selected, x):
                selected.append(x)
        return selected
    
    def local_search(S, max_iter=10000):
        """Local search with swaps and additions/removals."""
        S = list(S)
        best_sum = reciprocal_sum(S)
        best_S = S[:]
        improved = True
        iters = 0
        while improved and iters < max_iter and (time.time() - start_time) < DEADLINE:
            improved = False
            iters += 1
            # Try single removal + addition (swap)
            for i, elem in enumerate(S):
                S_without = S[:i] + S[i+1:]
                # Try adding each candidate not in S
                for cand in range(1, 257):
                    if cand not in S_without and valid_addition(S_without, cand):
                        new_S = S_without + [cand]
                        new_sum = reciprocal_sum(new_S)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_S = new_S
                            S = new_S
                            improved = True
            # Try single addition (if we can add without removing)
            for cand in range(1, 257):
                if cand not in S and valid_addition(S, cand):
                    new_S = S + [cand]
                    new_sum = reciprocal_sum(new_S)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_S = new_S
                        S = new_S
                        improved = True
        return best_S
    
    def local_search_double(S, max_iter=5000):
        """Local search allowing removal of one and addition of two."""
        S = list(S)
        best_sum = reciprocal_sum(S)
        best_S = S[:]
        improved = True
        iters = 0
        while improved and iters < max_iter and (time.time() - start_time) < DEADLINE:
            improved = False
            iters += 1
            # Try removal of one element
            for i, elem in enumerate(S):
                S_without = S[:i] + S[i+1:]
                # Try adding two candidates
                for cand1 in range(1, 257):
                    if cand1 in S_without:
                        continue
                    S_temp = S_without + [cand1]
                    if not valid_addition(S_without, cand1):
                        continue
                    for cand2 in range(cand1+1, 257):
                        if cand2 in S_temp:
                            continue
                        if not valid_addition(S_temp, cand2):
                            continue
                        new_S = S_temp + [cand2]
                        new_sum = reciprocal_sum(new_S)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_S = new_S
                            S = new_S
                            improved = True
        return best_S
    
    # Phase 1: Greedy initial solution
    S = greedy_initial()
    
    # Phase 2: Local search with single swaps
    S = local_search(S, max_iter=500)
    
    # Phase 3: Local search with double additions
    S = local_search_double(S, max_iter=200)
    
    # Phase 4: More single swaps
    S = local_search(S, max_iter=500)
    
    # Phase 5: Random restarts with local search
    best_overall = S
    best_overall_sum = reciprocal_sum(S)
    
    restarts = 0
    while (time.time() - start_time) < DEADLINE - 5:
        restarts += 1
        # Create a perturbed version
        S_perturbed = S[:]
        random.shuffle(S_perturbed)
        # Remove a few random elements
        for _ in range(min(3, len(S_perturbed))):
            if S_perturbed:
                idx = random.randint(0, len(S_perturbed)-1)
                S_perturbed.pop(idx)
        # Re-add greedily
        S_new = S_perturbed[:]
        for x in range(1, 257):
            if x not in S_new and valid_addition(S_new, x):
                S_new.append(x)
        # Local search
        S_new = local_search(S_new, max_iter=200)
        S_new = local_search_double(S_new, max_iter=100)
        S_new = local_search(S_new, max_iter=200)
        
        new_sum = reciprocal_sum(S_new)
        if new_sum > best_overall_sum:
            best_overall_sum = new_sum
            best_overall = S_new
    
    # Final polish
    best_overall = local_search(best_overall, max_iter=500)
    best_overall = local_search_double(best_overall, max_iter=100)
    best_overall = local_search(best_overall, max_iter=500)
    
    return best_overall
# EVOLVE-BLOCK-END
