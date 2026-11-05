import time
import math
import random

N = 128

# Precompute forbidden triples as (a, b, c) with a < b < c
forbidden_triples = []

def precompute_triples():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    forbidden_triples.append((a - 1, b - 1, c - 1))

precompute_triples()

def popcount(x):
    return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = 0  # Bitmask
    
    # Precompute for each variable, the list of triples it appears in
    # triple_vars[i] = list of (j, k) such that (i, j, k) is a forbidden triple
    # where i < j < k
    triple_vars = [[] for _ in range(N)]
    for (a, b, c) in forbidden_triples:
        triple_vars[a].append((b, c))
        triple_vars[b].append((a, c))
        triple_vars[c].append((a, b))
    
    # Count of triples for each variable
    triple_count = [len(triple_vars[i]) for i in range(N)]
    
    def greedy(order):
        S = 0
        excl = 0
        S_bits = []
        for x in order:
            if (S >> x) & 1: continue
            if (excl >> x) & 1: continue
            S |= (1 << x)
            S_bits.append(x)
            for u in S_bits[:-1]:
                # Find all w such that (min(x,u), max(x,u), w) is forbidden
                lo = min(x, u)
                hi = max(x, u)
                for (v, w) in triple_vars[lo]:
                    if v == hi:
                        excl |= (1 << w)
                    elif w == hi:
                        excl |= (1 << v)
        return S, excl

    # Get initial solution via greedy
    best_greedy = 0
    S, _ = greedy(range(N))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
    
    S, _ = greedy(range(N-1, -1, -1))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
        
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        order = list(range(N))
        random.shuffle(order)
        S, _ = greedy(order)
        if popcount(S) > popcount(best_greedy):
            best_greedy = S
            
    best_size = popcount(best_greedy)
    best_set = best_greedy
    
    # WalkSAT-style local search
    # State: assignment bitmask
    # Objective: minimize number of violated triples, then maximize size
    
    def count_violations(assign):
        """Count number of forbidden triples that are fully included in assign."""
        count = 0
        for (a, b, c) in forbidden_triples:
            if (assign >> a) & 1 and (assign >> b) & 1 and (assign >> c) & 1:
                count += 1
        return count
    
    def walk_sat(assign, max_steps=2000):
        """WalkSAT-style local search to find a valid assignment."""
        current = assign
        current_violations = count_violations(current)
        
        if current_violations == 0:
            return current, 0
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            if current_violations == 0:
                return current, 0
            
            # Find all variables that appear in at least one violated triple
            violated_vars = set()
            for (a, b, c) in forbidden_triples:
                if (current >> a) & 1 and (current >> b) & 1 and (current >> c) & 1:
                    violated_vars.add(a)
                    violated_vars.add(b)
                    violated_vars.add(c)
            
            if not violated_vars:
                return current, 0
            
            # Choose a variable to flip
            # With probability p, choose randomly from violated vars
            # With probability 1-p, choose the one that minimizes violations after flip
            p = 0.3  # noise parameter
            
            if random.random() < p:
                x = random.choice(list(violated_vars))
            else:
                # For each violated var, compute delta in violations if flipped
                best_delta = float('inf')
                best_x = None
                for x in violated_vars:
                    # Try flipping x
                    new_assign = current ^ (1 << x)
                    new_viol = count_violations(new_assign)
                    delta = new_viol - current_violations
                    if delta < best_delta:
                        best_delta = delta
                        best_x = x
                x = best_x
            
            # Flip x
            current = current ^ (1 << x)
            current_violations = count_violations(current)
        
        return current, current_violations
    
    # Phase 1: Use local search to improve from greedy solutions
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Start from best known or a random perturbation
        if restart == 0:
            start_assign = best_set
        else:
            # Perturb best_set: flip some random bits
            start_assign = best_set
            for _ in range(random.randint(2, 6)):
                x = random.randint(0, N - 1)
                start_assign ^= (1 << x)
        
        result_assign, viol = walk_sat(start_assign, max_steps=1500)
        
        if viol == 0:
            sz = popcount(result_assign)
            if sz > best_size:
                best_size = sz
                best_set = result_assign
    
    # Phase 2: More aggressive search - try to build up from scratch
    # Use a different strategy: start with all variables, remove violations
    for restart in range(100):
        if time.time() > deadline - 3:
            break
        
        # Start with a random subset
        if restart % 3 == 0:
            # Start with all, remove randomly
            assign = (1 << N) - 1
        elif restart % 3 == 1:
            # Start with random half
            assign = 0
            for i in range(N):
                if random.random() < 0.5:
                    assign |= (1 << i)
        else:
            # Start from best and perturb more
            assign = best_set
            for _ in range(random.randint(5, 15)):
                x = random.randint(0, N - 1)
                assign ^= (1 << x)
        
        result_assign, viol = walk_sat(assign, max_steps=3000)
        
        if viol == 0:
            sz = popcount(result_assign)
            if sz > best_size:
                best_size = sz
                best_set = result_assign
    
    # Final verification and polish
    # Try to add any remaining variables that don't create violations
    if time.time() < deadline - 1:
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for x in range(N):
                if (best_set >> x) & 1:
                    continue
                # Check if adding x creates any violation
                can_add = True
                for (a, b, c) in forbidden_triples:
                    if x in (a, b, c):
                        others = [v for v in (a, b, c) if v != x]
                        if (best_set >> others[0]) & 1 and (best_set >> others[1]) & 1:
                            can_add = False
                            break
                if can_add:
                    best_set |= (1 << x)
                    changed = True
    
    # Convert best_set bitmask to list
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    if not result:
        result = [1]
        
    return result
