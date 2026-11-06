import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map elements to APs containing them
    elem_to_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s:
                count += 1
        return count

    def is_valid_add(s, x):
        """Check if adding x to s creates a 4-AP."""
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if b in s and c in s and d in s:
                    return False
            elif x == b:
                if a in s and c in s and d in s:
                    return False
            elif x == c:
                if a in s and b in s and d in s:
                    return False
            else:
                if a in s and b in s and c in s:
                    return False
        return True

    def delta_add(s, x):
        """Change in 3-AP count if we add x to s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s:
                    delta += 1
            elif x == b:
                if a in s and c in s:
                    delta += 1
            else:
                if a in s and b in s:
                    delta += 1
        return delta

    def delta_remove(s, x):
        """Change in 3-AP count if we remove x from s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s:
                    delta += 1
            elif x == b:
                if a in s and c in s:
                    delta += 1
            else:
                if a in s and b in s:
                    delta += 1
        return -delta # Loss is negative

    def is_valid_swap(s, x_out, x_in):
        """Check if swapping x_out for x_in is valid (no 4-APs)."""
        # We need to check if x_in creates a 4-AP with (s - {x_out})
        # Equivalent to: for all 4-APs containing x_in, the other 3 elements must not be in (s - {x_out})
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            others = []
            if x_in == a:
                others = [b, c, d]
            elif x_in == b:
                others = [a, c, d]
            elif x_in == c:
                others = [a, b, d]
            else:
                others = [a, b, c]
            
            # If all 'others' are in s, and none of them is x_out, then a 4-AP is formed.
            # Actually, if all 'others' are in s, we must ensure that x_out is NOT one of them 
            # such that its removal breaks the AP? No.
            # The new set is S' = S \ {x_out} U {x_in}.
            # A 4-AP is formed in S' if it contains x_in and 3 other elements from S'.
            # The 3 other elements must be in S \ {x_out}.
            # So, if all 'others' are in S, and x_out is not in 'others', then the AP is intact in S'.
            # If x_out IS in 'others', then one of the required elements is removed, so the AP is broken.
            
            all_in_s = True
            x_out_in_others = False
            for y in others:
                if y not in s:
                    all_in_s = False
                    break
                if y == x_out:
                    x_out_in_others = True
            
            if all_in_s and not x_out_in_others:
                return False
        return True

    def greedy_construct(random_seed):
        random.seed(random_seed)
        s = set()
        available = list(range(1, N + 1))
        
        while available:
            best_x = None
            best_delta = -1
            # To add some randomness, we can pick from the top K candidates
            candidates = []
            for x in available:
                if is_valid_add(s, x):
                    d = delta_add(s, x)
                    candidates.append((d, x))
            
            if not candidates:
                break
                
            # Sort by delta descending
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            # Pick the best, or one of the top few to diversify
            top_k = min(5, len(candidates))
            if random.random() < 0.8:
                # Greedy pick
                best_x = candidates[0][1]
            else:
                # Random pick from top K
                best_x = random.choice([c[1] for c in candidates[:top_k]])
            
            s.add(best_x)
            available.remove(best_x)
            
        return s

    def local_search(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try all swaps
            # Optimization: Only consider swaps that might improve score
            for x_out in list(s):
                loss = -delta_remove(s, x_out) # loss is positive
                for x_in in range(1, N + 1):
                    if x_in in s:
                        continue
                    if not is_valid_swap(s, x_out, x_in):
                        continue
                    # Calculate gain of adding x_in to s - {x_out}
                    # Gain is delta_add(s - {x_out}, x_in)
                    # But delta_add(s, x_in) might be different if x_out was part of an AP with x_in
                    # Let's calculate precisely.
                    # New set S' = S \ {x_out} U {x_in}
                    # Delta = count_3aps(S') - count_3aps(S)
                    # This is expensive. Let's approximate or calculate carefully.
                    
                    # Actually, let's just calculate the change.
                    # Removing x_out loses 'loss' APs.
                    # Adding x_in gains 'gain' APs.
                    # Gain is number of 3-APs containing x_in where other 2 are in S \ {x_out}.
                    gain = 0
                    for ap_idx in elem_to_3aps[x_in]:
                        a, b, c = three_aps[ap_idx]
                        if x_in == a:
                            if b in s and c in s and b != x_out and c != x_out:
                                gain += 1
                        elif x_in == b:
                            if a in s and c in s and a != x_out and c != x_out:
                                gain += 1
                        else:
                            if a in s and b in s and a != x_out and b != x_out:
                                gain += 1
                            
                    if gain > loss:
                        s.remove(x_out)
                        s.add(x_in)
                        improved = True
                        break # Restart loop with new set
                if improved:
                    break
        return s

    best_s = set()
    best_score = -1
    
    # Run multiple greedy constructions
    for i in range(50):
        if time.time() > deadline:
            break
            
        s = greedy_construct(i)
        score = count_3aps(s)
        
        if score > best_score:
            best_score = score
            best_s = set(s)
            
        # Local search
        s = local_search(s)
        score = count_3aps(s)
        
        if score > best_score:
            best_score = score
            best_s = set(s)
            
    return sorted(best_s)
