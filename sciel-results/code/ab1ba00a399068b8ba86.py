import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = 192
    max_size = 48
    
    # Precompute: for each a, the set of sums that are multiples of a
    # Sum of 3 distinct others is in [3+4+5, 190+191+192] = [12, 573]
    # For a given a, bad sums are {a, 2a, 3a, ..., floor(573/a)*a}
    
    def is_valid(selected, candidate):
        """Check if adding candidate to selected maintains validity."""
        s = set(selected)
        s.add(candidate)
        for a in s:
            # Check: does a divide sum of any 3 distinct others in s?
            others = [x for x in s if x != a]
            if len(others) < 3:
                continue
            # Check all triples
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    for k in range(j+1, len(others)):
                        total = others[i] + others[j] + others[k]
                        if total % a == 0:
                            return False
        return True
    
    def is_valid_fast(selected, candidate):
        """Faster check: only need to check new constraints involving candidate."""
        # When adding candidate, new violations can only involve:
        # 1. candidate divides sum of 3 others in selected
        # 2. some a in selected divides sum of 2 from selected + candidate
        # 3. some a in selected divides sum of 1 from selected + 2... no, candidate is new
        # Actually: 
        # - candidate as divisor: check if candidate | (b+c+d) for any 3 distinct b,c,d in selected
        # - existing a as divisor: check if a | (b+c+candidate) for any 2 distinct b,c in selected, b,c != a
        # - also a | (candidate + b + c) where b,c in selected, b,c != a, b != c
        
        sel = list(selected)
        
        # Check 1: candidate divides sum of 3 from selected
        for i in range(len(sel)):
            for j in range(i+1, len(sel)):
                for k in range(j+1, len(sel)):
                    if (sel[i] + sel[j] + sel[k]) % candidate == 0:
                        return False
        
        # Check 2: existing a divides (b + c + candidate) for b,c in selected, b,c != a
        for a in sel:
            others = [x for x in sel if x != a]
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if (others[i] + others[j] + candidate) % a == 0:
                        return False
        
        return True
    
    # Greedy approach: try to build a large set
    # Start with large numbers (less likely to divide sums)
    best = []
    
    # Try multiple orderings
    candidates = list(range(1, n+1))
    
    # Strategy: try adding numbers from largest to smallest
    for x in range(n, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if is_valid_fast(best, x):
            best.append(x)
        if len(best) >= max_size:
            break
    
    current_best = len(best)
    
    # Try to improve with local search
    if time.time() < deadline - 10.0:
        # Try random restarts with different orderings
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            random.shuffle(candidates)
            curr = []
            for x in candidates:
                if len(curr) >= max_size:
                    break
                if is_valid_fast(curr, x):
                    curr.append(x)
            if len(curr) > current_best:
                current_best = len(curr)
                best = curr
    
    return best[:max_size]
