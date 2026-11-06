import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses a fast bitset-based DP for verification and a local search strategy.
    """
    N = 512
    MAX_SIZE = 48
    
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s wall-clock limit

    # Pre-compute candidate list
    candidates = list(range(1, N + 1))

    def is_valid(S):
        """
        Returns True if no element in S is the mean of any subset of >=2 other elements.
        Uses integer bitset DP for subset sums.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True
        
        total_sum = sum(S_list)
        
        for idx_a, a in enumerate(S_list):
            others = [x for i, x in enumerate(S_list) if i != idx_a]
            m_others = len(others)
            
            if m_others < 2:
                continue
                
            max_sum_others = total_sum - a
            
            # DP: dp[k] is a bitset of achievable sums using exactly k elements from others
            dp = [0] * (m_others + 1)
            dp[0] = 1  # Sum 0 with 0 elements
            
            for x in others:
                # Update dp in reverse to avoid using x multiple times in same step
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            # Check if a is the mean of any subset of size k >= 2
            for k in range(2, m_others + 1):
                target_sum = k * a
                if target_sum > max_sum_others:
                    break
                if (dp[k] >> target_sum) & 1:
                    return False
        
        return True

    def get_violation(S):
        """
        Returns (a, T) where a is the violating element and T is the subset causing violation.
        Returns (None, None) if no violation.
        T is a list of elements.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return None, None
        
        total_sum = sum(S_list)
        
        for idx_a, a in enumerate(S_list):
            others = [x for i, x in enumerate(S_list) if i != idx_a]
            m_others = len(others)
            
            if m_others < 2:
                continue
                
            max_sum_others = total_sum - a
            
            # Quick check for k=2 (pairs)
            others_set = set(others)
            for i in range(len(others)):
                x = others[i]
                y_needed = 2 * a - x
                if y_needed > x and y_needed in others_set:
                    return a, [x, y_needed]
            
            # If no pair, check k >= 3 using DP
            dp = [0] * (m_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            for k in range(3, m_others + 1):
                target_sum = k * a
                if target_sum > max_sum_others:
                    break
                if (dp[k] >> target_sum) & 1:
                    # We know a violation exists, but reconstructing T is expensive.
                    # For repair, we just need to know 'a' is involved or remove something.
                    # Returning T=None signals "violation exists but T unknown".
                    # We can try to reconstruct T by backtracking, but for now, 
                    # we can just return a and a random subset of others of size k
                    # if we want to be more specific, but for repair, removing 'a' 
                    # or a random element often works.
                    # Let's try to find one specific subset for better repair.
                    # Backtracking DP is complex. Instead, we'll just return a and None.
                    return a, None
        
        return None, None

    def greedy_construct(order):
        """
        Builds a set by adding elements in the given order if valid.
        """
        S = set()
        for x in order:
            if len(S) >= MAX_SIZE:
                break
            S_trial = S.copy()
            S_trial.add(x)
            if is_valid(S_trial):
                S = S_trial
        return S

    best_set = set()
    best_size = 0

    # Phase 1: Greedy Initialization with multiple deterministic and random orders
    # 1. Descending
    S_desc = greedy_construct(range(N, 0, -1))
    if len(S_desc) > best_size:
        best_set = S_desc
        best_size = len(S_desc)
        if best_size == MAX_SIZE:
            return list(S_desc)

    # 2. Ascending
    S_asc = greedy_construct(range(1, N + 1))
    if len(S_asc) > best_size:
        best_set = S_asc
        best_size = len(S_asc)
        if best_size == MAX_SIZE:
            return list(S_asc)

    # 3. Random Orders
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        random.shuffle(candidates)
        S_rand = greedy_construct(candidates)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
            if best_size == MAX_SIZE:
                return list(S_rand)

    # Phase 2: Local Search / Repair / Expansion
    S = best_set.copy()
    
    # If we didn't reach MAX_SIZE, try to expand/swap
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Check if current S is valid
        if not is_valid(S):
            # Repair S
            for step in range(100):
                if time.time() - start_time > time_limit:
                    break
                if is_valid(S):
                    break
                a, T = get_violation(S)
                if a is None:
                    break
                r = random.random()
                if T is not None and len(T) > 0 and r < 0.7:
                    # Remove an element from the violating subset
                    victim = random.choice(T)
                    S.remove(victim)
                elif r < 0.9:
                    # Remove the mean element
                    S.remove(a)
                else:
                    # Remove a random other element
                    others = [x for x in S if x != a]
                    if others:
                        S.remove(random.choice(others))
                    else:
                        S.remove(a)
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
            if best_size == MAX_SIZE:
                return list(S)
        
        # If we have a valid set, try to add or swap to increase size
        if len(S) < MAX_SIZE:
            # Try adding random elements
            for step in range(500):
                if time.time() - start_time > time_limit or len(S) >= MAX_SIZE:
                    break
                
                remaining = [x for x in candidates if x not in S]
                if not remaining:
                    break
                
                c = random.choice(remaining)
                S_trial = S.copy()
                S_trial.add(c)
                
                if is_valid(S_trial):
                    S = S_trial
                    if len(S) > best_size:
                        best_set = S.copy()
                        best_size = len(S)
                        if best_size == MAX_SIZE:
                            return list(S)
                else:
                    # Try to swap: remove one element from S to make room for c?
                    # Or remove a violating element from S_trial and keep c?
                    # If S_trial is invalid, adding c caused a violation.
                    # We can try to remove an element from S (not c) to fix it, 
                    # effectively swapping an old element for c.
                    a, T = get_violation(S_trial)
                    if a is not None:
                        # Candidates for removal: elements in T (if known) or a, or random from S
                        removal_candidates = set()
                        if T:
                            removal_candidates.update(T)
                        removal_candidates.add(a)
                        # Also consider removing c itself (revert)
                        # But we want to try swaps.
                        
                        # Try removing one of the candidates from S (not c)
                        # If we remove x from S and add c, the new set is S - {x} + {c}
                        # This is exactly S_trial - {x} if x != c.
                        # Since c is in S_trial, and x is in S, x is in S_trial.
                        # So S_trial - {x} is the swapped set.
                        
                        for x in list(removal_candidates):
                            if x == c:
                                continue
                            S_swap = S_trial.copy()
                            S_swap.remove(x)
                            if is_valid(S_swap):
                                # Accept swap with some probability
                                if random.random() < 0.5:
                                    S = S_swap
                                    break
                        else:
                            # If no swap worked, just revert or try another c
                            pass

    # Final check
    if best_set:
        if is_valid(best_set):
            return list(best_set)
    
    # Fallback (should not happen)
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
