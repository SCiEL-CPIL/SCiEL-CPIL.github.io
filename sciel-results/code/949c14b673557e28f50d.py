import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    n = 512
    random.seed(12345)
    
    # Precompute all 4-term arithmetic progressions
    # conflict_mask[x] is a numpy array of booleans: True for elements that, together with x,
    # form a 4-AP with two other elements in the set.
    # Specifically, if x is in S, we cannot have any triple (a,b,c) in conflict_mask[x] 
    # such that all a,b,c are in S.
    # More precisely: for each x, store a list of triples (y1, y2, y3) such that {x, y1, y2, y3} is a 4-AP.
    # We will use a bitmask representation for speed.
    
    # Let's represent the set S as a boolean numpy array of length n+1 (index 0 unused).
    # For each element x, we need to check if adding x to S creates a 4-AP.
    # This happens if there exists a triple (y1, y2, y3) in conflict_triples[x] 
    # such that y1, y2, y3 are all in S.
    
    # Precompute conflict_triples for each x.
    conflict_triples = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflict_triples[a].append((b, c, e))
            conflict_triples[b].append((a, c, e))
            conflict_triples[c].append((a, b, e))
            conflict_triples[e].append((a, b, c))
    
    def can_add(x, S_mask):
        """Check if adding x to S (represented by S_mask) creates a 4-AP."""
        for y1, y2, y3 in conflict_triples[x]:
            if S_mask[y1] and S_mask[y2] and S_mask[y3]:
                return False
        return True

    def has_4ap(S_mask):
        """Verify if S_mask contains a 4-AP."""
        for a in range(1, n + 1):
            if not S_mask[a]:
                continue
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                if S_mask[a + d] and S_mask[a + 2 * d] and S_mask[a + 3 * d]:
                    return True
        return False

    def ternary_01_set(n):
        """Generate all numbers in [1, n] whose base-3 representation contains only 0s and 1s."""
        result = set()
        for mask in range(1, 1 << 6):
            val = 0
            for bit in range(6):
                if mask & (1 << bit):
                    val += 3 ** bit
            if 1 <= val <= n:
                result.add(val)
        return result

    def greedy_construct(S_init, order):
        """Greedy construction starting from S_init, adding elements in 'order'."""
        S_mask = np.zeros(n + 1, dtype=bool)
        for x in S_init:
            S_mask[x] = True
        S_size = len(S_init)
        for x in order:
            if S_mask[x]:
                continue
            if can_add(x, S_mask):
                S_mask[x] = True
                S_size += 1
        return S_mask, S_size

    def local_search(S_mask, deadline, max_rounds=1000):
        """
        Local search: try to remove one element and add two others to increase size.
        """
        S_mask = S_mask.copy()
        current_size = np.sum(S_mask)
        no_improvement_count = 0
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
                
            # Try 1-2 move: remove 1, add 2
            if current_size == 0:
                break
            to_remove = np.random.choice(np.where(S_mask)[0])
            S_minus = S_mask.copy()
            S_minus[to_remove] = False
            S_minus_size = current_size - 1
            
            # Try to find 2 elements to add
            candidates = np.where(~S_minus)[0]
            candidates = candidates[candidates > 0]
            random.shuffle(candidates)
            
            added = []
            for c in candidates:
                if len(added) == 2:
                    break
                if can_add(c, S_minus | (np.zeros(n + 1, dtype=bool) if not added else np.concatenate([S_minus, [False]]) if False else S_minus)):
                    # Need to check against S_minus plus already added
                    temp_mask = S_minus.copy()
                    for a in added:
                        temp_mask[a] = True
                    if can_add(c, temp_mask):
                        added.append(c)
                    
            if len(added) == 2:
                S_mask[to_remove] = False
                for a in added:
                    S_mask[a] = True
                current_size = S_minus_size + 2
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > 50:
                    # Try a 1-1 swap to diversify
                    to_remove = np.random.choice(np.where(S_mask)[0])
                    S_minus = S_mask.copy()
                    S_minus[to_remove] = False
                    candidates = np.where(~S_minus)[0]
                    candidates = candidates[candidates > 0]
                    random.shuffle(candidates)
                    found_swap = False
                    for c in candidates:
                        if can_add(c, S_minus):
                            S_mask[to_remove] = False
                            S_mask[c] = True
                            found_swap = True
                            break
                    if not found_swap:
                        break
                        
        return S_mask, current_size

    def kick_and_repair(S_mask, deadline):
        """
        Random kick: remove a few random elements, then repair greedily.
        """
        S_mask = S_mask.copy()
        if np.sum(S_mask) == 0:
            return S_mask, 0
        
        # Remove 1-3 random elements
        kick_size = random.randint(1, 3)
        present = np.where(S_mask)[0]
        to_remove = random.sample(list(present), min(kick_size, len(present)))
        for x in to_remove:
            S_mask[x] = False
        
        # Repair: greedily add elements back
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_repaired, size = greedy_construct(set(np.where(S_mask)[0]), order)
        
        # Local search on the repaired set
        S_final, size = local_search(S_repaired, deadline, max_rounds=200)
        
        return S_final, size

    # Initial sets
    best_set = np.zeros(n + 1, dtype=bool)
    best_size = 0
    
    # 1. Greedy from empty
    order = list(range(1, n + 1))
    random.shuffle(order)
    greedy_set, greedy_size = greedy_construct(set(), order)
    if greedy_size > best_size:
        best_set = greedy_set
        best_size = greedy_size
        
    # 2. Greedy from ternary
    ternary = ternary_01_set(n)
    order2 = list(range(1, n + 1))
    random.shuffle(order2)
    ternary_greedy, ternary_size = greedy_construct(ternary, order2)
    if ternary_size > best_size:
        best_set = ternary_greedy
        best_size = ternary_size
        
    # 3. Local search on best
    if time.time() < deadline:
        best_set, best_size = local_search(best_set, deadline, max_rounds=2000)
        
    # 4. Multiple restarts with kicks
    for attempt in range(30):
        if time.time() > deadline:
            break
            
        # Random start
        if attempt % 3 == 0:
            S_init = set()
        elif attempt % 3 == 1:
            S_init = ternary_01_set(n)
        else:
            S_init = set(random.sample(range(1, n + 1), 30))
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        current, current_size = greedy_construct(S_init, order)
        
        # Local search
        if current_size >= best_size:
            current, current_size = local_search(current, deadline, max_rounds=1000)
        else:
            # Even if not better, do a kick-and-repair to explore
            current, current_size = kick_and_repair(current, deadline)
            
        if current_size > best_size:
            best_set = current
            best_size = current_size
            
    # 5. Final kick-and-repair on the best set
    if time.time() < deadline - 5.0:
        kicked, kicked_size = kick_and_repair(best_set, deadline)
        if kicked_size > best_size:
            best_set = kicked
            best_size = kicked_size
            
    # 6. Final local search on the best set
    if time.time() < deadline:
        best_set, best_size = local_search(best_set, deadline, max_rounds=500)
        
    # Final verification
    if has_4ap(best_set):
        # Fallback to safe ternary set
        ternary = ternary_01_set(n)
        best_set = np.zeros(n + 1, dtype=bool)
        for x in ternary:
            best_set[x] = True
        best_size = len(ternary)
        
    return sorted(np.where(best_set)[0].tolist())
