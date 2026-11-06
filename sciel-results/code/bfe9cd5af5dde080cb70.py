import time
import sys

def construct():
    deadline = time.time() + 115
    best_max = [float('inf')]
    best_set = [None]
    
    # Target differences 1..30
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
        
    # We need a Sidon set. 
    # Min size n: n(n-1)/2 >= 30 => n=9 (36 diffs).
    
    def dfs(elements, used_mask):
        if time.time() > deadline:
            return
        
        # Check if we covered all target diffs
        if (used_mask & target_mask) == target_mask:
            current_max = elements[-1]
            if current_max < best_max[0]:
                best_max[0] = current_max
                best_set[0] = list(elements)
            return
        
        # Pruning: Lower bound on final max element
        # The largest missing difference d_max must be formed by some pair (x, y) in the final set.
        # Since elements are sorted, y > x. y - x = d_max => y = x + d_max.
        # The smallest possible y is min(elements) + d_max? No, x can be any element.
        # Actually, y must be at least min(elements) + d_max if x is min(elements).
        # But x could be larger.
        # However, y must be at least max(elements) if we add elements? No, we only add larger elements.
        # So the final max element M must satisfy:
        # There exist a, b in FinalSet such that b - a = d_max.
        # Since FinalSet contains current elements, and we only add larger elements:
        # If d_max is not yet formed by current elements (it isn't, by definition of missing),
        # then either:
        # 1. One of a, b is current, one is new.
        #    If a is current, b = a + d_max. Since b > a, b is new (or larger current).
        #    If b is current, a = b - d_max. Since a < b, a is smaller. But we only add larger elements.
        #    So if b is in current elements, a must also be in current elements (since we can't add smaller).
        #    So if d_max is missing, it cannot be formed by two current elements.
        #    Thus, at least one of a, b must be a NEW element (larger than current max).
        #    Let new elements be $> M_{curr}$.
        #    Case 1: b is new. Then b >= M_{curr} + 1. a is current. b - a = d_max => b = a + d_max.
        #            So we need an element $a \in \text{current}$ such that $a + d_{max} > M_{curr}$ and we can add $a+d_{max}$.
        #            Actually, we just need to be able to add it.
        #            The final max will be at least $\max(M_{curr}, \max_{a \in \text{current}} (a + d_{max}))$?
        #            No, we can choose which a to use. We want to minimize the final max.
        #            So we can pick the smallest $a$ such that $a+d_{max}$ is a valid candidate?
        #            Wait, $a$ is fixed. We can only add $x = a + d_{max}$.
        #            So if we use pair $(a, a+d_{max})$, the max becomes $\max(M_{curr}, a+d_{max})$.
        #            We can choose which $a$ to use? No, the set is fixed.
        #            But we can choose to NOT use this pair yet?
        #            The pair must exist in the final set.
        #            So the final max $M_{final}$ must be $\ge \min_{a \in \text{current}} (a + d_{max})$?
        #            No, $a$ must be in the final set. Current elements are in the final set.
        #            So for each $a \in \text{current}$, if we use $a$ as the smaller element, we need $a+d_{max}$ in the set.
        #            If we don't use any current $a$ as the smaller element for $d_{max}$, then both $a, b$ are new?
        #            If both are new, $b > a > M_{curr}$. Then $b - a = d_{max} \implies b = a + d_{max} > M_{curr} + d_{max}$.
        #            So $M_{final} \ge M_{curr} + d_{max}$.
        #            
        #            So, $M_{final} \ge \min( \min_{a \in \text{current}} (a + d_{max}), M_{curr} + d_{max} )$.
        #            Actually, it's simpler:
        #            $M_{final} \ge \max(M_{curr}, \min_{a \in \text{current}} (a + d_{max}))$?
        #            No. If we pick $a_{min} = \min(\text{current})$, we need $a_{min} + d_{max}$ in the set.
        #            Is it possible to cover $d_{max}$ without using any current element as the smaller term?
        #            Yes, if we add two new elements $x, y$ with $y - x = d_{max}$.
        #            Then $y > x > M_{curr} \implies y \ge M_{curr} + 1 + d_{max}$.
        #            So $M_{final} \ge M_{curr} + 1 + d_{max}$.
        #            
        #            So the lower bound is $\min( \min_{a \in \text{current}} (a + d_{max}), M_{curr} + 1 + d_{max} )$.
        #            Wait, $\min_{a \in \text{current}} (a + d_{max})$ is just $\min(\text{current}) + d_{max}$.
        #            Since $\min(\text{current}) \le M_{curr}$, $\min(\text{current}) + d_{max} \le M_{curr} + d_{max} < M_{curr} + 1 + d_{max}$.
        #            So the tightest lower bound is $\min(\text{current}) + d_{max}$.
        #            BUT, we can only add elements $> M_{curr}$.
        #            If $\min(\text{current}) + d_{max} \le M_{curr}$, then the element $b = \min(\text{current}) + d_{max}$ is already in the range of current elements?
        #            No, if $b \le M_{curr}$, then $b$ must be in current elements to be in the final set (since we don't add smaller elements).
        #            If $b$ is not in current elements, we can't add it.
        #            So if $\min(\text{current}) + d_{max} \le M_{curr}$ and $\min(\text{current}) + d_{max} \notin \text{current}$, then we CANNOT cover $d_{max}$ using $\min(\text{current})$ as the smaller element.
        #            We would have to use a larger $a$ or two new elements.
        #            
        #            This logic is getting complicated. Let's use a simpler bound:
        #            $M_{final} \ge M_{curr}$.
        #            And $M_{final} \ge \max_{d \in \text{missing}} (\min(\text{current}) + d)$? No.
        #            
        #            Let's just use: if $M_{curr} \ge \text{best\_max}$, prune.
        #            And rely on the search to find the minimum.

        current_max = elements[-1]
        if current_max >= best_max[0]:
            return
            
        # Calculate lower bound for next elements
        # We need to cover missing diffs.
        missing_mask = target_mask & ~used_mask
        if missing_mask:
            d_max = missing_mask.bit_length() - 1
            # The smallest element in the set is 0.
            # To form d_max, we need x - y = d_max.
            # Since we only add larger elements, the smaller element y must be in current set 
            # OR y is a new element (but then x is even larger).
            # If y is in current set, x = y + d_max.
            # To minimize final max, we want x to be as small as possible, but x > current_max.
            # So we need some y in current set such that y + d_max > current_max is NOT required, 
            # but x must be added.
            # Actually, if y + d_max <= current_max, then x is not "new" in the sense of being > current_max.
            # But if x is not in current set, we can't add it.
            # So if for all y in current set, (y + d_max not in current set) AND (y + d_max <= current_max),
            # then we MUST use two new elements to form d_max.
            # This implies final max >= current_max + 1 + d_max.
            
            # Simple bound: 
            # If we use a current y, we add x = y + d_max.
            # We can only add x if x > current_max.
            # So we need a y such that y + d_max > current_max.
            # Smallest such x is min(y + d_max for y in current if y + d_max > current_max).
            # If no such y exists, we must use two new elements.
            
            valid_x_candidates = []
            for y in elements:
                x = y + d_max
                if x > current_max:
                    valid_x_candidates.append(x)
            
            if valid_x_candidates:
                min_x_from_current = min(valid_x_candidates)
                # We will have to add at least this element (or something larger) to cover d_max?
                # Not necessarily this specific x, but the max element will be at least min_x_from_current?
                # Yes, because any x formed with current y must be > current_max.
                # And the smallest such x is min_x_from_current.
                # So final max >= min_x_from_current.
                lb = min_x_from_current
            else:
                # Must use two new elements.
                # Smallest new element is current_max + 1.
                # Next is current_max + 1 + d_max.
                lb = current_max + 1 + d_max
            
            if lb >= best_max[0]:
                return
        
        # Try adding next element
        # We try x from current_max + 1 to best_max[0] - 1
        # Optimization: We can skip x if it doesn't help cover missing diffs?
        # No, we need to explore.
        
        for x in range(current_max + 1, best_max[0]):
            if time.time() > deadline:
                return
            
            # Check if x can be added
            valid = True
            new_diffs = 0
            for y in elements:
                d = x - y
                if (used_mask >> d) & 1:
                    valid = False
                    break
                new_diffs |= (1 << d)
            
            if not valid:
                continue
                
            # Add x
            elements.append(x)
            used_mask_new = used_mask | new_diffs
            dfs(elements, used_mask_new)
            elements.pop()

    # Initial call
    # We can start with 0.
    # To improve speed, we can also try starting with other small sets?
    # No, 0 is required.
    dfs([0], 0)
    
    if best_set[0] is None:
        # Fallback if search fails (should not happen)
        return [0]
        
    return best_set[0]
