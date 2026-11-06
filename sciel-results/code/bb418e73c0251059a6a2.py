import time
import random
from math import gcd

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Precompute all bad triples: (a, b, c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.add((a, b, c))
    
    # Build conflict structure: for each element, the set of pairs (x,y) such that
    # {x, y, elem} is a bad triple. Adding elem to S is invalid if any (x,y) in
    # conflict_pairs[elem] has both x,y in S.
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # Also build a set for O(1) membership: for each x, a set of frozensets of pairs
    conflict_pair_sets = {x: set() for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for (p, q) in conflict_pairs[x]:
            conflict_pair_sets[x].add(frozenset((p, q)))
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for (p, q) in conflict_pairs[x]:
            if p in S and q in S:
                return False
        return True
    
    def is_valid(S):
        """Check if set S contains no bad triple."""
        for x in S:
            for (p, q) in conflict_pairs[x]:
                if p in S and q in S:
                    return False
        return True
    
    # Greedy fill: given a starting set, greedily add elements until maximal
    def greedy_fill(S, order=None):
        S = set(S)
        if order is None:
            order = elements[:]
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S
    
    best = set()
    best_size = 0
    
    # Phase 1: Multiple randomized greedy starts
    num_starts = 300
    for start in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        order = elements[:]
        random.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        # Fill to maximal
        S = greedy_fill(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Phase 2: Local search on the best set
    # Try 1-out 1-in swaps and 1-out 2+ in swaps
    current = set(best)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to add any single missing element
        missing = [x for x in elements if x not in current]
        random.shuffle(missing)
        for v in missing:
            if can_add(v, current):
                current.add(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-out 1-in swaps
        removal_candidates = list(current)
        random.shuffle(removal_candidates)
        
        for u in removal_candidates:
            if time.time() > deadline - 5:
                break
            
            temp = current - {u}
            
            # Try to add a different element
            for v in missing:
                if v == u:
                    continue
                if can_add(v, temp):
                    # Check if this swap leads to a larger set after filling
                    trial = temp | {v}
                    trial = greedy_fill(trial)
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 1-out 2+ in: remove one, try to add multiple
        for u in removal_candidates:
            if time.time() > deadline - 5:
                break
            
            temp = current - {u}
            working = set(temp)
            added = []
            
            for v in missing:
                if v == u:
                    continue
                if can_add(v, working):
                    working.add(v)
                    added.append(v)
            
            if len(added) > 1:
                current = working
                improved = True
                break
    
    # Phase 3: Simulated annealing style exploration
    if time.time() < deadline - 10:
        current_sa = set(best)
        sa_best = set(best)
        sa_best_size = best_size
        
        # Temperature schedule
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        while T > T_min and time.time() < deadline - 5:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove a random element
                if current_sa:
                    to_remove = random.choice(list(current_sa))
                    trial = current_sa - {to_remove}
                    # Try to fill
                    trial = greedy_fill(trial)
                    delta = len(trial) - len(current_sa)
                    if delta >= 0 or random.random() < (1 - (1 / (1 + delta))) if delta < 0 else 1:
                        # Accept
                        if delta < 0:
                            accept_prob = 1.0 / (1.0 + abs(delta) * T)
                            if random.random() < accept_prob:
                                current_sa = trial
                        else:
                            current_sa = trial
            elif move_type < 0.7:
                # 1-out 1-in swap
                if current_sa:
                    to_remove = random.choice(list(current_sa))
                    missing_sa = [x for x in elements if x not in current_sa]
                    if missing_sa:
                        to_add = random.choice(missing_sa)
                        temp = current_sa - {to_remove}
                        if can_add(to_add, temp):
                            trial = temp | {to_add}
                            trial = greedy_fill(trial)
                            delta = len(trial) - len(current_sa)
                            if delta >= 0:
                                current_sa = trial
                            else:
                                accept_prob = 1.0 / (1.0 + abs(delta) * T)
                                if random.random() < accept_prob:
                                    current_sa = trial
            else:
                # 2-out 2+ in
                if len(current_sa) >= 2:
                    to_remove = set(random.sample(list(current_sa), 2))
                    temp = current_sa - to_remove
                    working = set(temp)
                    added = 0
                    missing_sa = [x for x in elements if x not in current_sa]
                    random.shuffle(missing_sa)
                    for v in missing_sa:
                        if v in to_remove:
                            continue
                        if can_add(v, working):
                            working.add(v)
                            added += 1
                    if added > 2:
                        trial = working
                        delta = len(trial) - len(current_sa)
                        if delta >= 0:
                            current_sa = trial
                        else:
                            accept_prob = 1.0 / (1.0 + abs(delta) * T)
                            if random.random() < accept_prob:
                                current_sa = trial
            
            if len(current_sa) > sa_best_size:
                sa_best_size = len(current_sa)
                sa_best = set(current_sa)
            
            T *= cooling
        
        if sa_best_size > best_size:
            best_size = sa_best_size
            best = sa_best
    
    # Final fill
    if time.time() < deadline - 2:
        best = greedy_fill(best)
    
    if not best:
        best = {1}
    
    return list(best)
