import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1213121"
    if n == 4:
        # Standard: 121312141213121413121314131214131214131214131214
        # Let's use the known optimal for n=4 which is length 33
        # Actually, let's just use the standard construction for n=4 and optimize
        pass

    # 1. Generate standard superpermutation
    # Standard construction: S_n = concat( S_{n-1} with n inserted )
    # This yields length sum_{k=1}^n k!
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of 1..m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap between end of 'new' and start of 'block'
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    best_word = word
    best_len = len(word)

    # Optimization: Multi-start Greedy Deletion
    # The standard construction is often not the shortest possible superpermutation.
    # We can shorten it by removing characters that do not destroy any permutation window.
    # The order in which we check/remove characters affects the final length.
    
    def get_perm_indices(w):
        perm_indices = defaultdict(list)
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                perm_indices[window].append(i)
        return perm_indices

    def is_valid_superperm(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == n * (1) * 2 * 3 * 4 * 5 * 6 * 7 # n!
        # More generally:
        # import math
        # return len(seen) == math.factorial(n)

    def try_greedy_deletion(w, order_type, deadline):
        """
        Attempts to shorten w by greedily removing safe characters.
        order_type: 'left', 'right', 'random', 'center', 'symbol_rotation'
        """
        w = list(w)
        improved = True
        # Pre-compute factorial for validity check
        import math
        target_perms = math.factorial(n)
        
        while improved and time.time() < deadline:
            improved = False
            perm_indices = get_perm_indices(w)
            L = len(w)
            
            # Determine order of indices to check
            if order_type == "right":
                indices_to_check = list(range(L - 1, -1, -1))
            elif order_type == "left":
                indices_to_check = list(range(L))
            elif order_type == "center":
                mid = L // 2
                indices_to_check = sorted(range(L), key=lambda i: abs(i - mid))
            elif order_type == "symbol_rotation":
                # Heuristic: prioritize indices where the symbol is "least recently used" 
                # in the context of the window, or just a different randomization
                # For now, let's use a random shuffle but biased towards symbols
                # that appear less frequently in the immediate neighborhood?
                # Actually, let's just use random for now as it's robust.
                indices_to_check = list(range(L))
                random.shuffle(indices_to_check)
            else: # random
                indices_to_check = list(range(L))
                random.shuffle(indices_to_check)
            
            # We iterate through the indices. If we delete one, we restart the pass
            # because indices shift. To be efficient, we can try to do multiple deletions
            # in one pass if we process from right to left (indices to the left are unaffected).
            # However, 'random' and 'left' require restarts or careful index management.
            # For simplicity and correctness with arbitrary orders, we restart after each deletion.
            # Given n=7, L~6000, and only ~8 deletions needed, this is fast.
            
            for i in indices_to_check:
                if time.time() >= deadline:
                    break
                
                # Check if index i is safe to remove
                # Affected windows start at k such that max(0, i-n+1) <= k <= min(i, L-n)
                start_k = max(0, i - n + 1)
                end_k = min(i, L - n)
                
                at_risk = False
                for k in range(start_k, end_k + 1):
                    p = tuple(w[k:k + n])
                    if len(set(p)) == n:
                        # Check if p appears anywhere else
                        indices = perm_indices[p]
                        # If all indices of p are in [start_k, end_k], it is at risk
                        has_safe_index = False
                        for idx in indices:
                            if idx < start_k or idx > end_k:
                                has_safe_index = True
                                break
                        if not has_safe_index:
                            at_risk = True
                            break
                
                if not at_risk:
                    # Perform deletion
                    w = w[:i] + w[i+1:]
                    improved = True
                    # Restart the loop to re-evaluate safety of other indices
                    # This is safe because we recompute perm_indices in the next iteration of while
                    break
            
        return w

    # Try different orders to escape local minima
    # We will run several rounds with different strategies.
    # 'right' is often the most effective for the standard construction.
    orders = ["right", "left", "center", "random", "random", "random", "right", "left", "symbol_rotation", "symbol_rotation"]
    
    for order in orders:
        if time.time() >= deadline - 5: # Leave 5s for final checks
            break
        
        candidate = try_greedy_deletion(best_word, order, deadline - 5)
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_word = candidate
            # If we found a shorter one, we can try to polish it further with other orders
            # The loop continues, so it will try the next order on the new best_word.
        
        # If we hit the theoretical lower bound or known best, we can stop early
        # Known best for n=7 is 5905.
        if best_len <= 5905:
            break

    return "".join(map(str, best_word))
