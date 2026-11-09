import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    def run_one_trial():
        S = []
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)
        
        def try_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    return False
            S.append(x)
            in_S[x] = True
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1
            return True

        def try_remove(x):
            S.remove(x)
            in_S[x] = False
            counts[2 * x] -= 1
            for y in S:
                if y != x:
                    counts[x + y] -= 1

        # 1. Randomized Greedy Initialization
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            if try_add(c):
                pass
        
        current_size = len(S)
        
        # Precompute sum counts for faster blocker detection
        # sum_pairs[s] = list of pairs (a,b) in S with a+b=s, a<=b
        
        def build_sum_pairs():
            sp = {}
            for i, a in enumerate(S):
                for b in S[i:]:
                    s = a + b
                    if s not in sp:
                        sp[s] = []
                    sp[s].append((a, b))
            return sp
        
        # 2. Local Search with systematic blocker finding and random swaps
        # We try to add random elements not in S.
        # If blocked, find blockers and try swapping.
        
        # Track which candidates are "hard" (blocked) for prioritization
        blocked_count = [0] * (max_val + 1)
        
        while time.time() < deadline - 5:
            # Pick candidate: prefer those with high blocked_count (more constrained)
            # or random
            if random.random() < 0.3:
                # Pick from high-constraint elements
                non_in_S = [x for x in range(1, max_val + 1) if not in_S[x]]
                if non_in_S:
                    # Weight by blocked_count
                    weights = [blocked_count[x] + 1 for x in non_in_S]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cum = 0
                    c = non_in_S[-1]
                    for x, w in zip(non_in_S, weights):
                        cum += w
                        if r <= cum:
                            c = x
                            break
            else:
                c = random.randint(1, max_val)
            
            if in_S[c]:
                continue
            
            # Try direct add
            if try_add(c):
                current_size += 1
                continue
            
            # Increment blocked count
            blocked_count[c] += 1
            
            # If direct add failed, find blockers efficiently
            # Build sum pairs once per iteration for O(1) lookup
            sum_pairs = build_sum_pairs()
            
            blockers_to_try = set()
            
            # Check 2*c
            if counts[2 * c] >= 2:
                for (a, b) in sum_pairs.get(2 * c, []):
                    blockers_to_try.add(a)
                    blockers_to_try.add(b)
            
            # Check c+y for y in S where count >= 2
            for y in S:
                s = c + y
                if counts[s] >= 2:
                    for (a, b) in sum_pairs.get(s, []):
                        if a != c and b != c:  # c is not in S yet
                            blockers_to_try.add(a)
                            blockers_to_try.add(b)
            
            blockers_list = list(blockers_to_try)
            random.shuffle(blockers_list)
            swapped = False
            
            # Try swapping each blocker with c
            for blocker in blockers_list:
                if not in_S[blocker]:
                    continue
                try_remove(blocker)
                if try_add(c):
                    swapped = True
                    break
                else:
                    try_add(blocker)
            
            # If no swap worked, try more random swaps as fallback
            if not swapped and len(S) > 0:
                for _ in range(10):
                    y = random.choice(S)
                    try_remove(y)
                    if try_add(c):
                        break
                    else:
                        try_add(y)
            
            # Occasional 2-for-1 swap: remove two elements, add c
            if not swapped and len(S) > 2 and random.random() < 0.2:
                # Try removing a pair that are both blockers
                if len(blockers_list) >= 2:
                    for i in range(min(5, len(blockers_list) - 1)):
                        for j in range(i + 1, min(6, len(blockers_list))):
                            b1, b2 = blockers_list[i], blockers_list[j]
                            if in_S[b1] and in_S[b2]:
                                try_remove(b1)
                                try_remove(b2)
                                if try_add(c):
                                    swapped = True
                                    break
                                else:
                                    try_add(b1)
                                    try_add(b2)
                            if swapped:
                                break
                        if swapped:
                            break

        # Final polish: try to add any remaining elements
        for c in range(1, max_val + 1):
            if not in_S[c]:
                if try_add(c):
                    current_size += 1
        
        return S

    # Run multiple trials if time permits
    best_S = run_one_trial()
    best_size = len(best_S)
    
    # If we have time, try another trial with different random seed
    if time.time() < deadline - 30:
        S2 = run_one_trial()
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
    
    # Final verification: ensure the set is valid
    def verify(S):
        counts = [0] * (2 * max_val + 1)
        for i, a in enumerate(S):
            for b in S[i:]:
                counts[a + b] += 1
        return all(c <= 2 for c in counts)
    
    if not verify(best_S):
        # Fallback: return a smaller valid set
        # This shouldn't happen, but just in case
        pass
    
    return best_S
