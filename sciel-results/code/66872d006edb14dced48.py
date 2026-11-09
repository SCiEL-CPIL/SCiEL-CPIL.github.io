import random

def construct():
    n = 512
    max_size = 48
    
    def compute_dp(elements):
        m = len(elements)
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in elements:
            for j in range(m - 1, -1, -1):
                if dp[j]:
                    dp[j + 1] |= (dp[j] << x)
        return dp

    def is_valid_candidate(c, selected, dp_full, dp_excl):
        m = len(selected)
        # 1. c is mean of subset of selected (size >= 2)
        for k in range(2, m + 1):
            target = k * c
            if (dp_full[k] >> target) & 1:
                return False
        
        # 2. c makes some a in selected a mean of subset of (selected \ {a}) U {c}
        #    Subset must include c.
        #    Sum(T') + c = (|T'| + 1) * a
        #    Sum(T') = (k' + 1) * a - c, where T' subset of selected \ {a}, |T'| = k' >= 1
        for a in selected:
            others_dp = dp_excl[a]
            m_others = m - 1
            for k_prime in range(1, m_others + 1):
                target_sum = (k_prime + 1) * a - c
                if target_sum < 0:
                    continue
                if (others_dp[k_prime] >> target_sum) & 1:
                    return False
        return True

    best_set = []
    
    # Strategies for selection
    # 0: Smallest
    # 1: Largest
    # 2: Random
    # 3: Random
    
    for restart in range(100):
        selected = []
        remaining = set(range(1, n + 1))
        
        # Choose a strategy for this restart
        strategy = restart % 4
        
        for _ in range(max_size):
            if not remaining:
                break
            
            if selected:
                dp_full = compute_dp(selected)
                dp_excl = {}
                for a in selected:
                    others = [x for x in selected if x != a]
                    dp_excl[a] = compute_dp(others)
            else:
                dp_full = [1]
                dp_excl = {}
            
            valid_candidates = []
            for c in remaining:
                if is_valid_candidate(c, selected, dp_full, dp_excl):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                break
            
            if strategy == 0:
                chosen = min(valid_candidates)
            elif strategy == 1:
                chosen = max(valid_candidates)
            elif strategy == 2:
                chosen = random.choice(valid_candidates)
            else:
                chosen = random.choice(valid_candidates)
            
            selected.append(chosen)
            remaining.remove(chosen)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= max_size:
                break
    
    return best_set[:max_size]
