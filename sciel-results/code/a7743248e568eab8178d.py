import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    # Precompute 3-APs
    # ap_list[x] = list of (a, b) such that a, b, x is a 3-AP and a < b < x
    # Also need to handle cases where x is the middle term: a, x, c
    # And cases where x is the first term: x, b, c
    # It's easier to just iterate over all 3-APs (a, b, c) and store them.
    # Or, for each x, store the set of pairs that would form an AP with x.
    
    # Let's define conflicts differently for fast update.
    # For a set S, V(S) = number of 3-APs in S.
    # When we add x, new APs formed are those (a,b,c) containing x where the other two are in S.
    # When we remove x, APs destroyed are those containing x where the other two are in S.
    
    # Precompute for each x, the list of pairs (u, v) such that {x, u, v} is a 3-AP.
    # u < v < x or x < u < v or u < x < v.
    # Let's just list all pairs (u, v) with u != v, u != x, v != x, and {x, u, v} is AP.
    
    pairs_for_x = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Check if a, b form an AP with some c
            # 2b = a + c => c = 2b - a
            c = 2 * b - a
            if c > 0 and c <= N and c != a and c != b:
                # (a, b, c) is a 3-AP
                # Add pair (b, c) to pairs_for_x[a]
                # Add pair (a, c) to pairs_for_x[b]
                # Add pair (a, b) to pairs_for_x[c]
                pairs_for_x[a].append((b, c))
                pairs_for_x[b].append((a, c))
                pairs_for_x[c].append((a, b))
                
    # Note: This might add duplicates if we're not careful?
    # a, b, c are distinct.
    # For a fixed AP (a,b,c), the pairs are unique.
    # e.g. AP (1,2,3).
    # x=1: pair (2,3)
    # x=2: pair (1,3)
    # x=3: pair (1,2)
    # This is correct.
    
    # Greedy Initialization
    def greedy_init():
        mask = 0
        # Try adding elements in random order? Or sorted by weight?
        # Sorted by weight (1, 2, 3...) is good for weight, but might block many others.
        # Let's try a few strategies and pick the best.
        best_mask = 0
        best_score = 0.0
        
        # Strategy 1: Greedy by index (smallest first)
        m = 0
        s = 0.0
        for i in range(1, N + 1):
            # Check if i can be added
            valid = True
            for u, v in pairs_for_x[i]:
                if ((m >> (u - 1)) & 1) and ((m >> (v - 1)) & 1):
                    valid = False
                    break
            if valid:
                m |= (1 << (i - 1))
                s += weights[i]
        if s > best_score:
            best_score = s
            best_mask = m
            
        # Strategy 2: Greedy by random order
        rng = random.Random(123)
        indices = list(range(1, N + 1))
        rng.shuffle(indices)
        m = 0
        s = 0.0
        for i in indices:
            valid = True
            for u, v in pairs_for_x[i]:
                if ((m >> (u - 1)) & 1) and ((m >> (v - 1)) & 1):
                    valid = False
                    break
            if valid:
                m |= (1 << (i - 1))
                s += weights[i]
        if s > best_score:
            best_score = s
            best_mask = m
            
        return best_mask, best_score

    best_mask, best_score = greedy_init()
    
    # Simulated Annealing
    # State: mask, current_violations, current_weight
    # We need to calculate initial violations and weight.
    
    def get_violations(mask):
        # Count 3-APs in mask
        v = 0
        # To avoid double counting, we can just count all APs (a,b,c) with a<b<c where all in mask.
        # Or use the pairs_for_x structure.
        # For each x in mask, count pairs (u,v) in pairs_for_x[x] where u,v in mask.
        # This counts each AP 3 times (once for each element).
        # So V = total_count / 3.
        count = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                for u, v in pairs_for_x[x]:
                    if ((mask >> (u - 1)) & 1) and ((mask >> (v - 1)) & 1):
                        count += 1
        return count // 3

    def get_weight(mask):
        w = 0.0
        for i in range(1, N + 1):
            if (mask >> (i - 1)) & 1:
                w += weights[i]
        return w

    current_mask = best_mask
    current_v = get_violations(current_mask)
    current_w = get_weight(current_mask)
    
    # Since greedy_init ensures validity, current_v should be 0.
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-5
    steps = 0
    
    # Penalty coefficient
    C = 10.0
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = random.randint(1, N)
        
        # Calculate delta
        # If i is in mask, we are removing it.
        # If i is not in mask, we are adding it.
        
        bit = 1 << (i - 1)
        
        if current_mask & bit:
            # Removing i
            # Weight change: -weights[i]
            dw = -weights[i]
            # Violation change:
            # APs containing i are destroyed if the other two are in mask.
            dv = 0
            for u, v in pairs_for_x[i]:
                if ((current_mask >> (u - 1)) & 1) and ((current_mask >> (v - 1)) & 1):
                    dv -= 1 # AP destroyed
            # Note: dv is negative.
            # New V = current_v + dv
        else:
            # Adding i
            dw = weights[i]
            dv = 0
            for u, v in pairs_for_x[i]:
                if ((current_mask >> (u - 1)) & 1) and ((current_mask >> (v - 1)) & 1):
                    dv += 1 # AP created
            # Note: dv is positive.
            
        new_v = current_v + dv
        new_w = current_w + dw
        
        # Energy
        # E = C * V - W
        # delta_E = C * dv - dw
        delta_E = C * dv - dw
        
        # Acceptance
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            # Accept move
            if current_mask & bit:
                current_mask ^= bit
            else:
                current_mask |= bit
            current_v = new_v
            current_w = new_w
            
            # Update best valid
            if current_v == 0:
                if current_w > best_score:
                    best_score = current_w
                    best_mask = current_mask
        else:
            # Reject move
            pass
            
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        steps += 1
        
        # Optional: Restart if stuck?
        # If T is very low and no improvement, maybe restart with new T?
        # For now, keep it simple.
        
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if (best_mask >> (i - 1)) & 1:
            result.append(i)
            
    return result
