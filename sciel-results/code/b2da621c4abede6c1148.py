import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 82s deadline to stay well under 90s limit
    n = 512

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # Map each element to the list of APs it participates in
    ap_map = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap = (a, b, c, e)
            ap_map[a].append(ap)
            ap_map[b].append(ap)
            ap_map[c].append(ap)
            ap_map[e].append(ap)
            
    # Precompute for each x, the list of "other 3 elements" tuples for each AP containing x
    # This avoids the if/elif chain in can_add
    ap_others = {i: [] for i in range(1, n + 1)}
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_others[a].append((b, c, e))
            ap_others[b].append((a, c, e))
            ap_others[c].append((a, b, e))
            ap_others[e].append((a, b, c))
    
    # Use bitset for faster membership checks
    # Represent a set as a numpy boolean array of length n+1
    def set_to_bits(S):
        bits = np.zeros(n + 1, dtype=bool)
        for x in S:
            bits[x] = True
        return bits
    
    def can_add(x, S_bits):
        """Check if adding x to S creates a 4-AP. S_bits is a numpy bool array."""
        for other in ap_others[x]:
            if S_bits[other[0]] and S_bits[other[1]] and S_bits[other[2]]:
                return False
        return True

    def has_4ap(S):
        """Verify if S contains a 4-AP."""
        for a in S:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in S and (a + 2 * d) in S and (a + 3 * d) in S:
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
        S = set(S_init)
        S_bits = set_to_bits(S)
        for x in order:
            if x in S:
                continue
            if can_add(x, S_bits):
                S.add(x)
                S_bits[x] = True
        return S

    def local_search(S, deadline, max_rounds=1000):
        """
        Fast local search: try 1-2 moves (remove 1, add 2) and 1-1 swaps.
        Uses precomputed conflict counts for efficiency.
        """
        S = set(S)
        S_bits = set_to_bits(S)
        current_size = len(S)
        no_improvement_count = 0
        
        # Precompute all candidates list
        all_elements = list(range(1, n + 1))
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            if not S:
                break
            
            # Try 1-2 move: remove 1, add 2
            to_remove = random.choice(list(S))
            S_minus_bits = S_bits.copy()
            S_minus_bits[to_remove] = False
            
            # Find all candidates that can be added to S_minus
            valid_candidates = []
            for c in all_elements:
                if S_minus_bits[c]:
                    continue
                if can_add(c, S_minus_bits):
                    valid_candidates.append(c)
            
            if len(valid_candidates) >= 2:
                # Try to find a compatible pair from valid_candidates
                # For efficiency, check pairs in random order
                random.shuffle(valid_candidates)
                found_pair = False
                temp_bits = S_minus_bits.copy()
                for i in range(min(len(valid_candidates), 40)):
                    c1 = valid_candidates[i]
                    if temp_bits[c1]:
                        continue
                    if not can_add(c1, temp_bits):
                        continue
                    temp_bits[c1] = True
                    for j in range(i + 1, min(i + 41, len(valid_candidates))):
                        c2 = valid_candidates[j]
                        if temp_bits[c2]:
                            continue
                        if can_add(c2, temp_bits):
                            S = S - {to_remove} | {c1, c2}
                            S_bits = temp_bits
                            S_bits[c2] = True
                            current_size = len(S)
                            no_improvement_count = 0
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if not found_pair:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            if no_improvement_count > 30:
                # Try 1-1 swap to diversify
                to_remove = random.choice(list(S))
                S_minus_bits = S_bits.copy()
                S_minus_bits[to_remove] = False
                
                found_swap = False
                for c in all_elements:
                    if S_minus_bits[c]:
                        continue
                    if can_add(c, S_minus_bits):
                        S = S - {to_remove} | {c}
                        S_bits = S_minus_bits
                        S_bits[c] = True
                        found_swap = True
                        break
                
                if not found_swap:
                    no_improvement_count += 1
                else:
                    no_improvement_count = max(0, no_improvement_count - 10)
                
                if no_improvement_count > 80:
                    break
        
        return S

    def kick_and_repair(S, deadline):
        """
        Random kick: remove a few random elements, then repair greedily.
        This helps escape local minima.
        """
        S = set(S)
        if not S:
            return S
        
        # Remove 1-4 random elements
        kick_size = random.randint(1, 4)
        to_remove = random.sample(list(S), min(kick_size, len(S)))
        S_kicked = S - set(to_remove)
        
        # Repair: greedily add elements back with random order
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_repaired = greedy_construct(S_kicked, order)
        
        # Local search on the repaired set
        S_final = local_search(S_repaired, deadline, max_rounds=150)
        
        return S_final

    # Initial sets
    best_set = ternary_01_set(n)
    best_size = len(best_set)
    
    # 1. Greedy from empty
    random.seed(42)
    order = list(range(1, n + 1))
    random.shuffle(order)
    greedy_set = greedy_construct(set(), order)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(best_set)
        
    # 2. Greedy from ternary
    order2 = list(range(1, n + 1))
    random.shuffle(order2)
    ternary_greedy = greedy_construct(ternary_01_set(n), order2)
    if len(ternary_greedy) > best_size:
        best_set = ternary_greedy
        best_size = len(best_set)
        
    # 3. Local search on best
    if time.time() < deadline:
        best_set = local_search(best_set, deadline, max_rounds=2000)
        best_size = len(best_set)
        
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
        
        current = greedy_construct(S_init, order)
        
        # Local search
        if len(current) >= best_size:
            current = local_search(current, deadline, max_rounds=1000)
        else:
            # Even if not better, do a kick-and-repair to explore
            current = kick_and_repair(current, deadline)
            
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # 5. Final kick-and-repair on the best set
    if time.time() < deadline - 5.0:
        kicked = kick_and_repair(best_set, deadline)
        if len(kicked) > best_size:
            best_set = kicked
            best_size = len(best_set)
            
    # 6. Final local search on the best set
    if time.time() < deadline:
        best_set = local_search(best_set, deadline, max_rounds=1000)
        best_size = len(best_set)
        
    # Final verification
    if has_4ap(best_set):
        # Fallback to safe ternary set
        best_set = ternary_01_set(n)
        
    return sorted(best_set)
