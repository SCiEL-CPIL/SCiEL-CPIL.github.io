import time
import random
from math import lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128

    # 1. Pre-compute bad triples and conflict structures
    # conflict_map[v] = set of (u, w) such that {v, u, w} is a bad triple
    conflict_map = defaultdict(set)
    bad_triples = []
    
    # Optimization: We only need to check triples where lcm(a,b) = lcm(a,c) = lcm(b,c)
    # This is equivalent to: a,b,c are all divisors of L, and L is the lcm of any pair.
    # Actually, just iterate all triples. N=128, N^3/6 ~ 350k checks. Very fast.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    conflict_map[a].add((b, c))
                    conflict_map[b].add((a, c))
                    conflict_map[c].add((a, b))
                    bad_triples.append((a, b, c))

    if time.time() > deadline:
        return list(range(64, 129))

    # Helper to count bad triples in a set
    def count_bad_triples(S):
        count = 0
        S_list = list(S)
        # Efficient check: for each pair in S, check if any c in S is a conflict
        # Or just iterate bad_triples and check if all 3 are in S.
        # N=128, bad_triples count is likely small (hundreds/thousands).
        # Let's precompute a set of bad triples for O(1) lookup? No, S changes.
        # Iterating bad_triples is O(|BT|). |BT| is small.
        for t in bad_triples:
            if t[0] in S and t[1] in S and t[2] in S:
                count += 1
        return count

    # Helper: can add v to S without creating a new bad triple?
    # A new bad triple is created if there exist u, w in S such that {v, u, w} is bad.
    # This is exactly: for any (u, w) in conflict_map[v], if u in S and w in S, then NO.
    def can_add(v, S):
        for u, w in conflict_map[v]:
            if u in S and w in S:
                return False
        return True

    # Greedy initialization
    def greedy_init():
        S = set()
        # Order by degree (number of conflicts)
        order = sorted(range(1, N + 1), key=lambda x: len(conflict_map[x]))
        for v in order:
            if can_add(v, S):
                S.add(v)
        return S

    best_set = greedy_init()
    best_size = len(best_set)
    
    # WalkSAT / Local Search
    # Objective: Minimize U (unsatisfied clauses = bad triples in S), Maximize |S|
    # Score = -U * 1000 + |S|
    
    def get_score(S):
        U = count_bad_triples(S)
        return -U * 1000 + len(S)

    def walksat(S, max_iters=20000):
        current_S = set(S)
        current_score = get_score(current_S)
        best_local_S = set(current_S)
        best_local_score = current_score
        
        # Track U for faster updates? 
        # For N=128, count_bad_triples is fast enough.
        
        temp = 1.0
        cooling = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # If valid, try to add
            U = count_bad_triples(current_S)
            if U == 0:
                # Try to add a random element
                candidates = [v for v in range(1, N + 1) if v not in current_S]
                if candidates:
                    v = random.choice(candidates)
                    if can_add(v, current_S):
                        new_S = current_S | {v}
                        new_score = get_score(new_S)
                        # Always accept improvement (size +1)
                        current_S = new_S
                        current_score = new_score
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_S = set(current_S)
                        continue
                    else:
                        # Adding v creates conflicts. 
                        # We might accept it to escape local optimum if we are stuck?
                        # But if U=0, we are in a valid set. We want to stay valid if possible.
                        # If we can't add, we are at a local max for this valid set.
                        # We should "kick" by removing an element or flipping a 0 to 1 (creating conflict) 
                        # to allow different additions later.
                        pass
            
            # If invalid or stuck, perform a flip
            # Pick a random bad triple to resolve
            if U > 0:
                # Find a random bad triple in current_S
                # Efficient: pick random v in S, pick random (u,w) in conflict_map[v] that are in S
                # Or just iterate bad_triples
                found_triple = None
                # Shuffled copy for random selection
                bt_copy = bad_triples[:]
                random.shuffle(bt_copy)
                for t in bt_copy:
                    if t[0] in current_S and t[1] in current_S and t[2] in current_S:
                        found_triple = t
                        break
                
                if found_triple:
                    # Pick a random variable in the triple to flip (1 -> 0)
                    v = random.choice(found_triple)
                    new_S = current_S - {v}
                    new_score = get_score(new_S)
                    delta = new_score - current_score
                    # Accept if improvement or with prob exp(delta/temp)
                    if delta > 0 or random.random() < min(1.0, exp(delta / temp)):
                        current_S = new_S
                        current_score = new_score
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_S = set(current_S)
            else:
                # U == 0, but we couldn't add. We are stuck.
                # Kick: Remove a random element from S to allow different structure?
                # Or flip a 0 to 1 even if it creates a conflict (to try to resolve it differently?)
                # Let's remove a random element to "kick"
                if current_S:
                    v = random.choice(list(current_S))
                    new_S = current_S - {v}
                    new_score = get_score(new_S)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < min(1.0, exp(delta / temp)):
                        current_S = new_S
                        current_score = new_score
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_S = set(current_S)
            
            temp *= cooling

        return best_local_S

    import math
    def exp(x):
        if x > 50: return 1.0
        if x < -50: return 0.0
        return math.exp(x)

    # Run multiple restarts
    random.seed(42)
    
    # Initial best
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Try to improve best_set with walksat
    if time.time() < deadline - 10:
        improved = walksat(best_set, max_iters=5000)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)

    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Diversified initialization
        S_init = set()
        # Random order with some bias
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Sort by degree + noise
        order.sort(key=lambda x: len(conflict_map[x]) + random.random() * 5)
        
        for v in order:
            if can_add(v, S_init):
                S_init.add(v)
        
        # Local search
        S_improved = walksat(S_init, max_iters=5000)
        
        if len(S_improved) > best_size:
            best_set = S_improved
            best_size = len(best_set)
            # If we found a new best, maybe run more search on it?
            # For now, just record.

    # Final polish: Try to add more to best_set
    def try_add_more(S):
        improved = True
        while improved:
            improved = False
            for v in range(1, N + 1):
                if v not in S and can_add(v, S):
                    S.add(v)
                    improved = True
        return S

    best_set = try_add_more(best_set)

    return list(best_set)
