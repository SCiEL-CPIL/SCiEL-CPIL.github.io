import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 84.0
    n = 512

    # Precompute all 4-APs
    # For each element x, store the list of (other1, other2, other3) triples
    # such that {x, other1, other2, other3} forms a 4-AP
    ap_others = [[] for _ in range(n + 1)]
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

    # Convert to numpy arrays for fast vectorized checking
    # For each x, store a 2D array of shape (num_aps, 3)
    ap_others_np = [np.array(ap_others[x], dtype=np.int32) if ap_others[x] else np.empty((0, 3), dtype=np.int32) for x in range(n + 1)]

    def set_to_bits(S):
        bits = np.zeros(n + 1, dtype=bool)
        for x in S:
            bits[x] = True
        return bits

    def can_add(x, S_bits):
        """Check if adding x to S creates a 4-AP using vectorized numpy."""
        if not ap_others_np[x].any():
            return True
        others = ap_others_np[x]
        # For each AP containing x, check if all 3 others are in S
        # S_bits[others] gives shape (num_aps, 3) boolean
        all_in = S_bits[others].all(axis=1)
        return not all_in.any()

    def can_add_many(candidates, S_bits):
        """Vectorized check for multiple candidates at once."""
        results = np.zeros(len(candidates), dtype=bool)
        for i, x in enumerate(candidates):
            if not ap_others_np[x].any():
                results[i] = True
            else:
                others = ap_others_np[x]
                all_in = S_bits[others].all(axis=1)
                results[i] = not all_in.any()
        return results

    def has_4ap(S):
        """Verify if S contains a 4-AP."""
        S_set = set(S)
        for a in S:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in S_set and (a + 2 * d) in S_set and (a + 3 * d) in S_set:
                    return True
        return False

    def ternary_01_set(n):
        """Generate all numbers in [1, n] whose base-3 representation contains only 0s and 1s."""
        result = set()
        for mask in range(1, 1 << 9):
            val = 0
            for bit in range(9):
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

    def local_search(S, deadline, max_rounds=5000):
        """
        Fast local search with 1-2 moves and 1-1 swaps using vectorized checks.
        """
        S = set(S)
        S_bits = set_to_bits(S)
        current_size = len(S)
        no_improvement_count = 0
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
            
            # Find all candidates that can be added to S_minus using vectorized check
            candidates = [c for c in all_elements if not S_minus_bits[c]]
            if len(candidates) > 0:
                valid_mask = can_add_many(candidates, S_minus_bits)
                valid_candidates = [c for c, v in zip(candidates, valid_mask) if v]
            else:
                valid_candidates = []
            
            if len(valid_candidates) >= 2:
                random.shuffle(valid_candidates)
                found_pair = False
                temp_bits = S_minus_bits.copy()
                limit = min(len(valid_candidates), 60)
                for i in range(limit):
                    c1 = valid_candidates[i]
                    if temp_bits[c1]:
                        continue
                    if not can_add(c1, temp_bits):
                        continue
                    temp_bits[c1] = True
                    for j in range(i + 1, min(i + 50, len(valid_candidates))):
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
            
            if no_improvement_count > 50:
                # Try 1-1 swap to diversify
                to_remove = random.choice(list(S))
                S_minus_bits = S_bits.copy()
                S_minus_bits[to_remove] = False
                
                candidates = [c for c in all_elements if not S_minus_bits[c]]
                if candidates:
                    valid_mask = can_add_many(candidates, S_minus_bits)
                    valid_candidates = [c for c, v in zip(candidates, valid_mask) if v]
                    random.shuffle(valid_candidates)
                    found_swap = False
                    for c in valid_candidates[:30]:
                        if can_add(c, S_minus_bits):
                            S = S - {to_remove} | {c}
                            S_bits = S_minus_bits
                            S_bits[c] = True
                            found_swap = True
                            break
                
                if not found_swap:
                    no_improvement_count += 1
                else:
                    no_improvement_count = max(0, no_improvement_count - 15)
                
                if no_improvement_count > 100:
                    break
        
        return S

    def kick_and_repair(S, deadline):
        """
        Random kick: remove a few random elements, then repair greedily.
        """
        S = set(S)
        if not S:
            return S
        
        kick_size = random.randint(1, 5)
        to_remove = random.sample(list(S), min(kick_size, len(S)))
        S_kicked = S - set(to_remove)
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_repaired = greedy_construct(S_kicked, order)
        
        S_final = local_search(S_repaired, deadline, max_rounds=200)
        return S_final

    # Initial sets
    best_set = ternary_01_set(n)
    best_size = len(best_set)
    
    random.seed(42)
    
    # 1. Greedy from empty
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
        best_set = local_search(best_set, deadline, max_rounds=3000)
        best_size = len(best_set)
        
    # 4. Multiple restarts
    for attempt in range(40):
        if time.time() > deadline:
            break
            
        if attempt % 3 == 0:
            S_init = set()
        elif attempt % 3 == 1:
            S_init = ternary_01_set(n)
        else:
            S_init = set(random.sample(range(1, n + 1), random.randint(20, 40)))
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        current = greedy_construct(S_init, order)
        
        if len(current) >= best_size:
            current = local_search(current, deadline, max_rounds=1500)
        else:
            current = kick_and_repair(current, deadline)
            
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # 5. Final kick-and-repair on the best set
    if time.time() < deadline - 5.0:
        kicked = kick_and_repair(best_set, deadline)
        if len(kicked) > best_size:
            best_set = kicked
            best_size = len(kicked)
            
    # 6. Final local search on the best set
    if time.time() < deadline:
        best_set = local_search(best_set, deadline, max_rounds=2000)
        best_size = len(best_set)
        
    # Final verification
    if has_4ap(best_set):
        best_set = ternary_01_set(n)
        
    return sorted(best_set)
