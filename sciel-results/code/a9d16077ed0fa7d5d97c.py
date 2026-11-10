"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples and build conflict graph
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We want a set S with no bad triples.
    # This is equivalent to: For every bad triple (a,b,c), not all three are in S.
    # This is an Independent Set problem in a 3-uniform hypergraph.
    # Or equivalently, we can view it as: if a and b are in S, then c cannot be in S (for any c such that (a,b,c) is a bad triple).
    
    # Let's build the conflict structure:
    # For each pair (u, v), let conflict[u][v] be the set of w such that (min(u,v,w), mid(u,v,w), max(u,v,w)) is a bad triple.
    # Actually, it's easier to store: for each element x, a list of pairs (u, v) such that (u, v, x) is a bad triple.
    # Then x is incompatible with a set S if there exists (u, v) in conflict[x] such that u in S and v in S.
    
    conflict = {i: set() for i in range(1, n + 1)}
    # conflict[x] contains sets of pairs? No, just pairs.
    # If u and v are both in S, and (u,v,x) is a bad triple, we can't add x.
    
    # Let's find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    # Add to conflict lists
                    conflict[c].add((a, b))
                    conflict[b].add((a, c))
                    conflict[a].add((b, c))
                    
    # 3. Greedy initialization
    # Sort by degree (number of conflicting pairs)
    # Actually, a better heuristic:
    # An element x is "safe" to add if no conflicting pair is fully present in S.
    # Let's try a greedy approach:
    # Sort elements by some priority.
    # One priority: number of bad triples involving x (degree in hypergraph).
    # Low degree first? Or high degree first?
    # Usually for Max Independent Set, we want to pick vertices that block the fewest others?
    # Or pick vertices that are involved in the most conflicts to satisfy them?
    # Let's try: Pick the element that participates in the fewest bad triples first?
    # No, if we pick an element, we forbid many others.
    # If we pick an element with high degree, we forbid many. So maybe low degree first?
    
    degrees = {i: len(conflict[i]) for i in range(1, n + 1)}
    
    # Greedy construction
    S = set()
    remaining = list(range(1, n + 1))
    # Sort by degree ascending?
    # Let's try multiple orders and keep the best
    # Order 1: Ascending degree
    # Order 2: Descending degree
    # Order 3: Random
    
    def greedy(order):
        S_local = set()
        for x in order:
            # Check if x can be added
            valid = True
            for u, v in conflict[x]:
                if u in S_local and v in S_local:
                    valid = False
                    break
            if valid:
                S_local.add(x)
        return S_local
    
    # Try ascending degree
    order1 = sorted(remaining, key=lambda x: degrees[x])
    S1 = greedy(order1)
    
    # Try descending degree
    order2 = sorted(remaining, key=lambda x: -degrees[x])
    S2 = greedy(order2)
    
    best_set = max([S1, S2], key=len)
    best_size = len(best_set)
    
    # 4. Local Search: 1-Swap and 2-Swap
    # We want to escape local optima.
    # Move: Remove x from S, Add y to S.
    # Check validity of new set.
    
    # Precompute for faster check?
    # Validity of S U {y} \ {x}:
    # 1. y must not form a bad triple with two elements in (S \ {x}).
    # 2. x is removed, so any triple involving x is fine (unless y replaces it? No, triples are specific).
    # Actually, we just check the new set.
    
    def is_valid_set(S_set):
        # Check all bad triples
        for a, b, c in bad_triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True
        
    # This is too slow if done naively for every swap.
    # Let's use the conflict list check.
    # To check if y can be added to S_current:
    #   Check conflict[y]: for each (u,v) in conflict[y], if u in S_current and v in S_current -> invalid.
    
    # Let's implement a fast swap search.
    # We maintain S as a set.
    # We try all pairs (x in S, y not in S).
    # To speed up, we can precompute for each y, the pairs (u,v).
    # Checking "u in S and v in S" is fast if we iterate conflict[y].
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all 1-swaps
        # To make it faster, iterate over x in S
        # For each x, try all y not in S
        # This is O(|S| * (N-|S|) * avg_conflict_degree)
        
        # Optimization:
        # Instead of checking all y, we can prioritize y that "unlock" many things?
        # Let's just do a full sweep.
        
        S_list = list(best_set)
        candidates_to_add = [y for y in range(1, n + 1) if y not in best_set]
        
        for x in S_list:
            if time.time() > deadline - 5:
                break
            # Remove x
            S_trial = set(best_set)
            S_trial.remove(x)
            
            # Try to add any y
            # We can try to add all valid y's?
            # Or just the best y?
            # Let's try to add the best y (one that minimizes future conflicts? or just first valid?)
            # To maximize size, we should add AS MANY as possible after removing x?
            # No, a swap is 1-for-1.
            # But we can do: Remove x, then greedily add ALL valid y's.
            # This is a "Remove 1, Fill Max" move.
            
            S_trial_filled = set(S_trial)
            for y in candidates_to_add:
                if y in S_trial_filled:
                    continue
                # Check if y is valid in S_trial_filled
                valid = True
                for u, v in conflict[y]:
                    if u in S_trial_filled and v in S_trial_filled:
                        valid = False
                        break
                if valid:
                    S_trial_filled.add(y)
            
            if len(S_trial_filled) > len(best_set):
                best_set = S_trial_filled
                best_size = len(best_set)
                improved = True
                # Update candidates_to_add? It changes.
                # Break out of x loop to restart?
                break
        
        # If no improvement from "Remove 1, Fill Max", try pure 1-swaps?
        # The "Remove 1, Fill Max" is stronger than 1-swap because it can add multiple.
        # It is effectively a local optimum for single removals.
        
    # 5. Simulated Annealing / Random Restarts
    # If stuck, perturb and re-greedy.
    if time.time() < deadline - 5:
        import random
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            
            # Perturb: Remove random 5-10 elements
            k = random.randint(5, 10)
            S_trial = set(best_set)
            to_remove = random.sample(list(S_trial), min(k, len(S_trial)))
            for x in to_remove:
                S_trial.remove(x)
            
            # Greedily fill
            # Randomize order for diversity
            order_rand = list(range(1, n + 1))
            random.shuffle(order_rand)
            
            for y in order_rand:
                if y in S_trial:
                    continue
                valid = True
                for u, v in conflict[y]:
                    if u in S_trial and v in S_trial:
                        valid = False
                        break
                if valid:
                    S_trial.add(y)
            
            if len(S_trial) > len(best_set):
                best_set = S_trial
                best_size = len(best_set)
                
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
