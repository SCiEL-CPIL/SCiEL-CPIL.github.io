import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Precompute for fast validity checking
    # For each a in 1..n, precompute which pairs (b,c) with b<=c have (b+c) % a == 0
    # But that's too much. Instead, we'll use a different approach.
    
    def is_valid_addition(x, S_list):
        """Check if adding x to S_list (sorted list) maintains the property.
        S_list is a sorted list of current elements.
        """
        # For all triples (a,b,c) in S_list ∪ {x} with a < b <= c, a ∤ (b+c)
        # New triples must involve x.
        
        # Case A: x is the smallest (x < b <= c, b,c in S_list)
        # Need: for all b <= c in S_list with b > x, (b+c) % x != 0
        for i in range(len(S_list)):
            if S_list[i] > x:
                b = S_list[i]
                for j in range(i, len(S_list)):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        
        # Case B: x is the middle (a < x <= c, a,c in S_list, a < x <= c)
        # Need: for all a in S_list with a < x, and c in S_list with c >= x, (x+c) % a != 0
        for a in S_list:
            if a < x:
                for c in S_list:
                    if c >= x:
                        if (x + c) % a == 0:
                            return False
        
        # Case C: x is the largest (a < b < x, a,b in S_list)
        # Need: for all a < b in S_list with b < x, (b+x) % a != 0
        for i in range(len(S_list)):
            if S_list[i] < x:
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    if b < x:
                        if (b + x) % a == 0:
                            return False
        
        return True
    
    def is_valid_full(S_list):
        """Full validation of a set."""
        S_set = set(S_list)
        for a in S_list:
            for i in range(len(S_list)):
                b = S_list[i]
                if b <= a:
                    continue
                for j in range(i, len(S_list)):
                    c = S_list[j]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def greedy_from_large_to_small():
        S = []
        for x in range(n, 0, -1):
            if is_valid_addition(x, S):
                S.append(x)
        return S
    
    def greedy_random_order():
        S = []
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_addition(x, S):
                S.append(x)
        return S
    
    def greedy_weighted_order(weights):
        """Greedy with weighted random ordering."""
        S = []
        order = list(range(1, n + 1))
        # Weighted shuffle: higher weight = more likely to be picked earlier
        for i in range(len(order) - 1, 0, -1):
            total_w = sum(weights[order[j] - 1] for j in range(i + 1))
            if total_w <= 0:
                r = random.random() * (i + 1)
                idx = int(r)
            else:
                r = random.random() * total_w
                cum = 0
                idx = i
                for j in range(i + 1):
                    cum += weights[order[j] - 1]
                    if r <= cum:
                        idx = j
                        break
            order[i], order[idx] = order[idx], order[i]
        for x in order:
            if is_valid_addition(x, S):
                S.append(x)
        return S
    
    def local_search_add(S):
        """Try to add excluded elements in descending order."""
        S_set = set(S)
        excluded = sorted(set(range(1, n + 1)) - S_set, reverse=True)
        S_list = sorted(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            new_excluded = []
            for x in excluded:
                if is_valid_addition(x, S_list):
                    S_list.append(x)
                    S_list.sort()
                    improved = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
        return S_list
    
    def local_search_swap(S, max_rounds=50):
        """Try swap moves: remove one element, add another (or two)."""
        S_list = sorted(S)
        S_set = set(S_list)
        excluded = sorted(set(range(1, n + 1)) - S_set, reverse=True)
        best_S = S_list[:]
        best_len = len(best_S)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            
            # Try single swaps: remove a, add b
            for i, a in enumerate(S_list):
                if time.time() > deadline - 10:
                    break
                temp_S = S_list[:i] + S_list[i+1:]
                for b in excluded:
                    # Quick check: can we add b to temp_S?
                    if is_valid_addition(b, temp_S):
                        new_S = temp_S + [b]
                        new_S.sort()
                        # Verify it's valid
                        if is_valid_full(new_S):
                            if len(new_S) > best_len:
                                best_S = new_S
                                best_len = len(new_S)
                                S_list = new_S
                                S_set = set(S_list)
                                excluded = sorted(set(range(1, n + 1)) - S_set, reverse=True)
                                improved = True
                            break
            
            # Try adding excluded elements
            if time.time() > deadline - 10:
                break
            for x in excluded[:]:
                if is_valid_addition(x, S_list):
                    S_list.append(x)
                    S_list.sort()
                    S_set.add(x)
                    excluded.remove(x)
                    improved = True
                    if len(S_list) > best_len:
                        best_S = S_list[:]
                        best_len = len(best_S)
            
            if not improved:
                break
        
        return best_S
    
    def local_search_remove_add(S):
        """Remove some elements, then greedily add back."""
        S_list = sorted(S)
        S_set = set(S_list)
        excluded = sorted(set(range(1, n + 1)) - S_set, reverse=True)
        best_S = S_list[:]
        best_len = len(best_S)
        
        # Try removing small elements and re-adding
        for remove_count in [1, 2]:
            if time.time() > deadline - 10:
                break
            if remove_count == 1:
                candidates = list(range(len(S_list)))
            else:
                candidates = [(i, j) for i in range(len(S_list)) for j in range(i+1, len(S_list))]
            
            random.shuffle(candidates)
            for cand in candidates[:100]:  # Limit tries
                if time.time() > deadline - 10:
                    break
                if remove_count == 1:
                    idx = cand
                    temp_S = S_list[:idx] + S_list[idx+1:]
                else:
                    i, j = cand
                    if i > j:
                        i, j = j, i
                    temp_S = S_list[:i] + S_list[i+1:j] + S_list[j+1:]
                
                # Greedily add from excluded
                new_S = temp_S[:]
                for x in excluded:
                    if is_valid_addition(x, new_S):
                        new_S.append(x)
                new_S.sort()
                
                if len(new_S) > best_len:
                    best_S = new_S
                    best_len = len(new_S)
                    S_list = new_S
                    S_set = set(S_list)
                    excluded = sorted(set(range(1, n + 1)) - S_set, reverse=True)
        
        return best_S
    
    best_S = []
    
    # Strategy 1: Greedy from large to small
    S1 = greedy_from_large_to_small()
    best_S = S1[:]
    
    # Improve with local search
    if time.time() < deadline - 10:
        S1_imp = local_search_add(S1)
        if len(S1_imp) > len(best_S):
            best_S = S1_imp
    
    if time.time() < deadline - 15:
        S1_swap = local_search_swap(best_S, max_rounds=20)
        if len(S1_swap) > len(best_S):
            best_S = S1_swap
    
    # Strategy 2: Random greedy restarts
    if time.time() < deadline - 10:
        for trial in range(300):
            if time.time() > deadline - 5:
                break
            S2 = greedy_random_order()
            if len(S2) > len(best_S):
                best_S = S2[:]
            # Quick local search on promising solutions
            if len(S2) >= len(best_S) and time.time() < deadline - 10:
                S2_imp = local_search_add(S2)
                if len(S2_imp) > len(best_S):
                    best_S = S2_imp
                if time.time() < deadline - 15:
                    S2_swap = local_search_swap(best_S, max_rounds=10)
                    if len(S2_swap) > len(best_S):
                        best_S = S2_swap
    
    # Strategy 3: Weighted greedy
    if time.time() < deadline - 10:
        # Try different weightings
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            # Weight larger numbers more heavily
            weights = [1.0 + (x / n) * 2.0 for x in range(1, n + 1)]
            # Add some randomness to weights
            weights = [w * (0.5 + random.random()) for w in weights]
            S3 = greedy_weighted_order(weights)
            if len(S3) > len(best_S):
                best_S = S3[:]
            if time.time() < deadline - 10:
                S3_imp = local_search_add(S3)
                if len(S3_imp) > len(best_S):
                    best_S = S3_imp
    
    # Final local search
    if time.time() < deadline - 5:
        best_S = local_search_add(best_S)
    
    if time.time() < deadline - 5:
        best_S = local_search_swap(best_S, max_rounds=30)
    
    # Final remove-add search
    if time.time() < deadline - 5:
        best_S = local_search_remove_add(best_S)
    
    return sorted(best_S)
