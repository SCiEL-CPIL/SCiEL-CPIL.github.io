"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 180
    
    # Precompute all bad triples (frozensets of 3 distinct elements)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            den = b - a
            num = a * b
            if num % den == 0:
                c = num // den
                if 1 <= c <= N and c != a and c != b:
                    bad_triples.append(frozenset((a, b, c)))
    
    # Build index: for each element, list of bad triples containing it
    elem_to_triples = defaultdict(list)
    for t in bad_triples:
        for x in t:
            elem_to_triples[x].append(t)
    
    def is_valid(S):
        """Check if S contains no bad triple."""
        for t in bad_triples:
            if t.issubset(S):
                return False
        return True
    
    def count_violations(S):
        """Count number of bad triples fully contained in S."""
        count = 0
        for t in bad_triples:
            if t.issubset(S):
                count += 1
        return count
    
    def remove_violations(S, max_iter=1000):
        """Remove elements to eliminate all violations. Returns (clean_set, num_removed)."""
        S = set(S)
        removed = 0
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Find a violating triple
            violated = None
            for t in bad_triples:
                if t.issubset(S):
                    violated = t
                    break
            if violated is None:
                break
            # Remove the element from violated that appears in the most bad triples (greedy)
            best_elem = None
            best_count = -1
            for x in violated:
                cnt = len(elem_to_triples[x])
                if cnt > best_count:
                    best_count = cnt
                    best_elem = x
            S.discard(best_elem)
            removed += 1
        return S, removed
    
    def local_search(S):
        """Try to add back removed elements or swap to improve size."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each element not in S
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                new_S = S | {x}
                if count_violations(new_S) == 0:
                    S = new_S
                    improved = True
                    break
            # Try swaps: remove one, add one
            if not improved:
                s_list = list(S)
                random.shuffle(s_list)
                for rm in s_list:
                    if time.time() > deadline:
                        break
                    for x in candidates:
                        new_S = (S - {rm}) | {x}
                        if count_violations(new_S) == 0 and len(new_S) > len(S):
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
        return S
    
    # Strategy 1: Randomized greedy removal from full set
    best_S = set(range(1, N + 1))
    best_size = 0
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start with full set, randomly remove elements to break violations
        S = set(range(1, N + 1))
        random.shuffle(list(S))
        
        # Greedy removal: repeatedly remove an element that breaks the most violations
        for _ in range(200):
            if time.time() > deadline:
                break
            violated = None
            for t in bad_triples:
                if t.issubset(S):
                    violated = t
                    break
            if violated is None:
                break
            # Pick random element from violated to remove
            x = random.choice(list(violated))
            S.discard(x)
        
        # Verify it's clean
        if not is_valid(S):
            S, _ = remove_violations(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        # Local search to try to add back elements
        S2 = local_search(S)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = set(S2)
    
    # Strategy 2: Greedy addition from small sets
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Start with a random subset of the upper half
        S = set(random.sample(range(N//2, N + 1), random.randint(40, 80)))
        
        # Try to add elements from 1 to N//2 in random order
        candidates = list(range(1, N//2 + 1))
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            new_S = S | {x}
            if count_violations(new_S) == 0:
                S = new_S
        
        # Also try adding some upper-half elements
        upper_candidates = [x for x in range(N//2, N + 1) if x not in S]
        random.shuffle(upper_candidates)
        for x in upper_candidates:
            if time.time() > deadline:
                break
            new_S = S | {x}
            if count_violations(new_S) == 0:
                S = new_S
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        S2 = local_search(S)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = set(S2)
    
    # Final local search on best solution
    if time.time() < deadline - 5:
        best_S = local_search(best_S)
    
    return list(best_S)

# EVOLVE-BLOCK-END
