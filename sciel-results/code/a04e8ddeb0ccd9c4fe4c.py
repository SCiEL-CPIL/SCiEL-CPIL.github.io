"""Rewritten program: bit-parallel greedy maximal independent set on a 3-uniform hypergraph."""
import time
import random
from itertools import combinations

N = 128
DEADLINE = 85.0  # safety margin below 90s

def construct():
    start_time = time.time()
    
    # Precompute all "bad" triples: (a, b, c) with a < b < c, a,b,c in 1..N
    # such that a+b, a+c, b+c are all in 1..N.
    # We'll store for each vertex v, the list of hyperedges (as frozensets or tuples) containing v.
    # But for greedy, we need to check: if we add v, which existing pairs (x,y) in our set
    # would form a bad triple with v?
    
    # Alternative approach: For a set S, it's valid iff there do NOT exist distinct a,b,c in S
    # such that a+b in S, a+c in S, b+c in S.
    
    # Greedy: iterate vertices in some order, add v if adding it doesn't create a forbidden triple.
    # To check efficiently: when considering adding v, look at all pairs (x, y) already in S
    # such that v, x, y would form a forbidden triple. A triple (a,b,c) is forbidden if
    # a+b, a+c, b+c are all in the domain 1..N AND we require them to be in S.
    # Wait, re-reading: "Forbid THREE DISTINCT members a,b,c for which a+b, a+c and b+c are also selected."
    # So a,b,c must be in S, AND a+b, a+c, b+c must ALSO be in S.
    
    # So when adding v, we need to check: are there x, y in S such that {v,x,y} has all three pair sums in S?
    # i.e., check if v+x in S, v+y in S, and x+y in S.
    
    # For each candidate v, and each pair (x,y) in S, check if v+x, v+y, x+y are all in S.
    # This is O(|S|^2) per insertion, which is fine for |S| up to ~64.
    
    # Let's use a set for O(1) membership.
    
    def is_valid_addition(v, S_set):
        """Check if adding v to S_set keeps it valid."""
        # For each pair (x, y) in S_set, check if {v, x, y} is a forbidden triple.
        # A triple (a,b,c) is forbidden if a+b, a+c, b+c are all in S_set ∪ {v}.
        # Since x, y are already in S_set, we need:
        #   v+x in S_set ∪ {v}  -> v+x in S_set (since v+x != v unless x=0)
        #   v+y in S_set ∪ {v}  -> v+y in S_set
        #   x+y in S_set ∪ {v}  -> x+y in S_set or x+y == v
        # But wait, a+b, a+c, b+c must be "selected", i.e., in the final set.
        # When we add v, the set becomes S_set ∪ {v}.
        # So we need v+x ∈ S_set∪{v}, v+y ∈ S_set∪{v}, x+y ∈ S_set∪{v}.
        # Since v+x > v (for x>=1), v+x ≠ v, so v+x must be in S_set.
        # Similarly v+y must be in S_set.
        # x+y could be in S_set or equal to v.
        
        for x in S_set:
            if v + x > N:
                continue
            if (v + x) not in S_set:
                continue
            for y in S_set:
                if y == x:
                    continue
                if v + y > N:
                    continue
                if (v + y) not in S_set:
                    continue
                if x + y > N:
                    continue
                if (x + y) in S_set or (x + y) == v:
                    return False
        return True
    
    # Try multiple random orderings and keep the best result.
    best_set = set()
    best_size = 0
    
    num_trials = 50
    trial = 0
    while trial < num_trials and (time.time() - start_time) < DEADLINE:
        random.seed(trial * 12345 + 67890)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_set = set()
        for v in order:
            if (time.time() - start_time) > DEADLINE:
                break
            if is_valid_addition(v, S_set):
                S_set.add(v)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
        
        trial += 1
    
    # Also try the natural ordering (ascending, descending)
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        if (time.time() - start_time) > DEADLINE:
            break
        S_set = set()
        for v in order:
            if (time.time() - start_time) > DEADLINE:
                break
            if is_valid_addition(v, S_set):
                S_set.add(v)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
    
    result = sorted(best_set)
    return [int(x) for x in result]

# Quick sanity check at module level (not called by construct, just for testing)
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(result)
    # Verify
    S = set(result)
    valid = True
    for a, b, c in combinations(result, 3):
        if (a + b) in S and (a + c) in S and (b + c) in S:
            valid = False
            break
    print(f"Valid: {valid}")
