"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit (was 85s, caused timeouts)
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    # lcm(i, j) = i * j // gcd(i, j)
    # We only care about triples where lcm(a,b) == lcm(a,c) == lcm(b,c)
    
    # First, let's find all bad triples.
    # A triple (a,b,c) with a<b<c is bad if:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    
    # Optimization: 
    # lcm(a,b) = L implies a|L and b|L.
    # So a,b,c must all divide L.
    # Also L must be a multiple of lcm(a,b).
    
    # Let's just iterate a<b<c and check.
    # N=128, C(128,3) = ~340k triples. This is very fast.
    bad_triples = set()
    # To speed up, we can precompute LCMs.
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
            
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm[a][b]
            # c must be > b
            # lcm(a,c) == lab implies c must divide lab (since a divides lab, lcm(a,c) is a multiple of lab? No, lcm(a,c) is multiple of a and c. If lcm(a,c)=lab, then c|lab)
            # Also lcm(b,c) == lab implies c|lab.
            # So c must be a divisor of lab, and c > b.
            # We can just iterate c from b+1 to N and check.
            for c in range(b+1, N+1):
                if lcm[a][c] == lab and lcm[b][c] == lab:
                    bad_triples.add(frozenset([a, b, c]))
    
    # For each element x, store the pairs (y,z) such that {x,y,z} is a bad triple.
    # This allows O(1) or O(k) check for can_add.
    conflicts = {x: [] for x in range(1, N+1)}
    for bt in bad_triples:
        # bt is a frozenset of 3 elements
        # Unpack to a, b, c for easier manipulation
        a, b, c = list(bt)
        conflicts[a].append(frozenset([b, c]))
        conflicts[b].append(frozenset([a, c]))
        conflicts[c].append(frozenset([a, b]))
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        # S is a set of ints
        for pair in conflicts[x]:
            # pair is a frozenset of two elements y, z
            # Check if both y and z are in S
            if pair.issubset(S):
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Heuristic: Start with empty set or small random set
    # We will run multiple restarts.
    
    def greedy_fill(S, order):
        """Add elements from 'order' to S if possible."""
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)

    # Local search helper: try to improve S by removing one element and re-filling
    def local_search(S, max_iterations=100):
        """
        Attempts to improve S by 1-removal and re-greedy-fill.
        Accepts same-size moves with probability to escape local minima.
        Returns True if S was improved (size increased) or a same-size move was accepted.
        """
        current_size = len(S)
        if current_size == 0:
            return False
            
        S_list = list(S)
        random.shuffle(S_list)
        
        for u in S_list:
            if time.time() > deadline - 3:
                break
                
            # Remove u
            S.remove(u)
            
            # Try to re-fill with random order
            candidates = [x for x in range(1, N+1) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S):
                    S.add(x)
            
            new_size = len(S)
            if new_size > current_size:
                return True
            elif new_size == current_size:
                # Accept same-size move with some probability to escape local minima
                if random.random() < 0.3:
                    return True
                else:
                    # Revert: try next u
                    pass
            else:
                # Size decreased, revert
                S.add(u)
                
        return False

    # Multiple restarts with different strategies
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        S = set()
        
        # Strategy: sometimes start with a random subset, sometimes empty
        if r % 3 == 0:
            # Start with a small random seed set
            seed_size = random.randint(1, 5)
            seed_elements = random.sample(elements, seed_size)
            for x in seed_elements:
                if can_add(x, S):
                    S.add(x)
        
        order = elements[:]
        random.shuffle(order)
        
        greedy_fill(S, order)
        
        # Local search phase
        improved = True
        while improved and time.time() < deadline - 3:
            improved = local_search(S)
            
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
            # Try to add any remaining element
            for x in elements:
                if x not in best_set and can_add(x, best_set):
                    best_set.add(x)
                    best_size = len(best_set)
                
    # Final polish
    if time.time() < deadline - 1 and best_set:
        for x in elements:
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                
    if not best_set:
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
