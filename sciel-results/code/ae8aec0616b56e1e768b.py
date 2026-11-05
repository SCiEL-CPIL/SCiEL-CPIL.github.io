import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # 85s safety margin within 90s limit
    N = 180
    
    # Precompute all invalid triples (a, b, c) with a < b < c, 1/a = 1/b + 1/c
    # c = a*b/(b-a), requires b > a, (b-a) | a*b, c > b, c <= N
    invalid_pairs = {}  # maps (a, b) -> c, meaning if a,b in set then c cannot be
    # Also need: if a, c in set then b cannot be; if b, c in set then a cannot be
    # Actually, let's precompute for each element x, which pairs (i, j) with i<j, i!=x, j!=x
    # would form a triple with x.
    
    # Better approach: precompute all triples
    triples = []  # list of (a, b, c) with a < b < c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            denom = b - a
            num = a * b
            if num % denom != 0:
                continue
            c = num // denom
            if c > b and c <= N:
                triples.append((a, b, c))
    
    # For each element, precompute the set of (other1, other2) pairs that form a triple with it
    # conflict[x] = set of frozenset({y, z}) such that {x, y, z} is a triple
    conflict = [set() for _ in range(N + 1)]
    for (a, b, c) in triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))
    
    # Precompute for each element x, the set of elements that "conflict" with x
    # in the sense that there exists y such that {x, y, z} is a triple
    # This is useful for quick rejection but not sufficient.
    
    def is_valid_set(S_set):
        """Check if S_set (a set of ints) has no invalid triple."""
        S_list = sorted(S_set)
        S_set_local = set(S_list)
        for (a, b, c) in triples:
            if a in S_set_local and b in S_set_local and c in S_set_local:
                return False
        return True
    
    def greedy_random(order):
        """Greedy: process elements in given order, add if doesn't create a triple."""
        S = set()
        for x in order:
            # Check if adding x creates a triple
            # x could be a, b, or c in some triple
            valid = True
            # x is a: need b, c in S with (b,c) in conflict[x]
            for (b, c) in conflict[x]:
                if b in S and c in S:
                    valid = False
                    break
            if valid:
                # x is b: need a, c in S with (a,c) in conflict[x]
                for (a, c) in conflict[x]:
                    if a in S and c in S:
                        valid = False
                        break
            if valid:
                # x is c: need a, b in S with (a,b) in conflict[x]
                for (a, b) in conflict[x]:
                    if a in S and b in S:
                        valid = False
                        break
            if valid:
                S.add(x)
        return S
    
    def local_search(S, time_limit):
        """Try to improve S by swapping elements out and in."""
        S_set = set(S)
        S_list = list(S)
        best = len(S_list)
        best_set = S_set.copy()
        
        # Precompute which elements are NOT in S
        not_in_S = [x for x in range(1, N + 1) if x not in S_set]
        
        # For each element in S, try removing it and adding a different element
        # Also try just adding elements not in S (if possible)
        
        # First, try to add any element not in S
        for x in not_in_S:
            if time.time() > time_limit:
                break
            valid = True
            for (b, c) in conflict[x]:
                if b in S_set and c in S_set:
                    valid = False
                    break
            if valid:
                S_set.add(x)
                best_set = S_set.copy()
                best = len(S_set)
        
        # Now try swaps: remove one element from S, add one not in S
        # Do multiple passes
        for _pass in range(10):
            if time.time() > time_limit:
                break
            improved = False
            S_list = list(S_set)
            not_in_S = [x for x in range(1, N + 1) if x not in S_set]
            
            for rem in S_list:
                if time.time() > time_limit:
                    break
                S_set.remove(rem)
                for add in not_in_S:
                    if add == rem:
                        continue
                    valid = True
                    for (b, c) in conflict[add]:
                        if b in S_set and c in S_set:
                            valid = False
                            break
                    if valid:
                        S_set.add(add)
                        improved = True
                        break
                if not improved:
                    S_set.add(rem)
            
            if not improved:
                break
        
        return best_set, best
    
    best_size = 0
    best_set = set()
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    # First, try a few deterministic orderings
    # 1. Descending
    S1 = greedy_random(list(range(N, 0, -1)))
    best_size = max(best_size, len(S1))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1
    
    # 2. Ascending
    S2 = greedy_random(list(range(1, N + 1)))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2
    
    # 3. Random orderings
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 10:  # Leave time for local search
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = greedy_random(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search on the best set found
    if time.time() < deadline - 5:
        improved_set, improved_size = local_search(best_set, deadline - 2)
        if improved_size > best_size:
            best_size = improved_size
            best_set = improved_set
    
    # Try local search on a few other good sets
    if time.time() < deadline - 5:
        # Try local search on descending greedy result
        S1 = greedy_random(list(range(N, 0, -1)))
        if len(S1) >= best_size - 2:
            improved_set, improved_size = local_search(S1, deadline - 2)
            if improved_size > best_size:
                best_size = improved_size
                best_set = improved_set
    
    return list(best_set)

