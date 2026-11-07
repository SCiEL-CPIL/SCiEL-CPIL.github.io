import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55  # Leave 5s margin for safety

    # Precompute comparability: for each mask, which other masks are comparable (strict subset/superset)
    # Use a list of sets for fast membership testing
    comparables = [set() for _ in range(total_masks)]
    
    # For each mask s, find all supersets (strict) and subsets (strict)
    # Supersets of s: t such that (s & ~t) == 0 and t != s
    # Subsets of s: t such that (t & ~s) == 0 and t != s
    
    # Efficient approach: for each s, enumerate all supersets and subsets
    # Number of supersets of s = 2^(12 - popcount(s)) - 1
    # Number of subsets of s = 2^(popcount(s)) - 1
    # Total work is manageable.
    
    for s in range(total_masks):
        # Find all strict supersets
        # A strict superset t of s satisfies: t = s | u where u is a non-empty subset of ~s
        complement = (~s) & (total_masks - 1)
        # Enumerate all non-empty subsets of complement
        u = complement
        while u:
            t = s | u
            comparables[s].add(t)
            comparables[t].add(s)
            u = (u - 1) & complement
        
        # Find all strict subsets
        # A strict subset t of s satisfies: t = s & u where u is a proper subset of s (u != s)
        # Equivalently, t is a subset of s and t != s
        # Enumerate all proper subsets of s
        u = s
        while u:
            t = u
            if t != s:
                # t is a strict subset of s
                # But we already added (s, t) when processing t's supersets? No, we added (t, s) as superset.
                # Actually, when we processed t, we would have added s as a superset of t.
                # So comparables[t] already contains s, and comparables[s] already contains t.
                # Wait, no. When we process t, we enumerate supersets of t, which includes s.
                # So comparables[t].add(s) and comparables[s].add(t) would be done.
                # But here we're processing s, so we should only add subsets that haven't been added yet?
                # Actually, the superset enumeration already handles both directions.
                # If t is a subset of s, then s is a superset of t.
                # When we processed t, we enumerated supersets of t, including s, and added both directions.
                # So we don't need to do the subset enumeration separately!
                pass
            u = (u - 1) & s
    
    # Wait, the above only does supersets. Let me verify:
    # For pair (s, t) where s < t (s is strict subset of t):
    # When processing s, we enumerate supersets of s, which includes t. We add t to comparables[s] and s to comparables[t].
    # So both directions are covered. Good.
    # But the subset enumeration loop does nothing useful. Let me remove it.
    
    # Actually, let me redo this more carefully.
    # The superset enumeration for s adds all strict supersets of s.
    # For any pair (a, b) where a is a strict subset of b:
    #   When processing a, b is enumerated as a superset, so b is added to comparables[a] and a to comparables[b].
    # So we only need the superset enumeration.
    
    # Let me rebuild comparables cleanly.
    comparables = [set() for _ in range(total_masks)]
    for s in range(total_masks):
        complement = (~s) & (total_masks - 1)
        u = complement
        while u:
            t = s | u
            comparables[s].add(t)
            comparables[t].add(s)
            u = (u - 1) & complement
    
    # Now comparables[s] contains all t such that s and t are comparable (one is strict subset of the other).
    
    # Greedy initialization with precomputed comparability
    def greedy_init(seed=None):
        if seed is not None:
            random.seed(seed)
        
        fam = []
        fam_set = set()
        
        # Start with a few random or structured sets
        # Try starting with empty set and full set for good structure
        candidates = list(range(total_masks))
        random.shuffle(candidates)
        
        # Pick first few greedily but with some randomness
        for _ in range(min(5, m)):
            best_mask = -1
            best_score = -1
            # Sample a few candidates to add randomness
            sample = random.sample(candidates, min(50, len(candidates)))
            for s in sample:
                if s in fam_set:
                    continue
                score = sum(1 for x in fam if x in comparables[s])
                if score > best_score:
                    best_score = score
                    best_mask = s
            if best_mask == -1:
                # Pick random
                best_mask = random.choice(candidates)
            fam.append(best_mask)
            fam_set.add(best_mask)
        
        # Now greedily add the rest
        while len(fam) < m:
            best_mask = -1
            best_score = -1
            for s in range(total_masks):
                if s in fam_set:
                    continue
                score = sum(1 for x in fam if x in comparables[s])
                if score > best_score:
                    best_score = score
                    best_mask = s
            if best_mask == -1:
                break
            fam.append(best_mask)
            fam_set.add(best_mask)
        
        return fam, fam_set
    
    def calc_score(fam):
        """Calculate number of comparable pairs in family."""
        count = 0
        n_fam = len(fam)
        for i in range(n_fam):
            for j in range(i + 1, n_fam):
                if fam[i] in comparables[fam[j]]:
                    count += 1
        return count
    
    best_family = None
    best_score_val = -1
    
    # Try multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        fam, fam_set = greedy_init(seed=num_restarts * 12345)
        
        # Local search: try single swaps
        current_family = fam[:]
        current_fam_set = fam_set.copy()
        current_score = calc_score(current_family)
        
        if current_score > best_score_val:
            best_score_val = current_score
            best_family = current_family[:]
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Compute degrees for all elements in current_family
            # deg[i] = number of elements in current_family comparable to current_family[i]
            degs = [0] * len(current_family)
            for i in range(len(current_family)):
                for j in range(len(current_family)):
                    if i != j and current_family[i] in comparables[current_family[j]]:
                        degs[i] += 1
            
            # Try swapping each element with a better candidate
            indices = list(range(len(current_family)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                A = current_family[i]
                dA = degs[i]
                
                best_gain = 0
                best_B = -1
                
                # Try all candidates not in current_family
                for B in range(total_masks):
                    if B in current_fam_set:
                        continue
                    
                    # Compute dB' = number of elements in current_family \ {A} comparable to B
                    # = (number of elements in current_family comparable to B) - (1 if A comparable to B else 0)
                    count_comp = 0
                    for x in current_family:
                        if x in comparables[B]:
                            count_comp += 1
                    
                    # If A is comparable to B, subtract 1 (since we're removing A)
                    if A in comparables[B]:
                        count_comp -= 1
                    
                    gain = count_comp - dA
                    if gain > best_gain:
                        best_gain = gain
                        best_B = B
                
                if best_B != -1 and best_gain > 0:
                    # Perform swap
                    current_fam_set.remove(A)
                    current_fam_set.add(best_B)
                    current_family[i] = best_B
                    
                    current_score += best_gain // 2
                    
                    if current_score > best_score_val:
                        best_score_val = current_score
                        best_family = current_family[:]
                    
                    improved = True
                    # Break and restart the outer while to recompute degrees
                    break
            
            # If we broke out of the for loop due to a swap, continue the while loop
            # If no improvement was made, the while loop will exit
        
        # If stuck at local optimum, try a kick
        if time.time() < deadline:
            # Random kick: replace 2-3 random elements
            kick_size = random.choice([2, 3])
            kick_indices = random.sample(range(len(current_family)), kick_size)
            kick_elements = [current_family[i] for i in kick_indices]
            
            for idx, old_elem in zip(kick_indices, kick_elements):
                current_fam_set.remove(old_elem)
                new_elem = random.randint(0, total_masks - 1)
                while new_elem in current_fam_set:
                    new_elem = random.randint(0, total_masks - 1)
                current_fam_set.add(new_elem)
                current_family[idx] = new_elem
            
            current_score = calc_score(current_family)
            if current_score > best_score_val:
                best_score_val = current_score
                best_family = current_family[:]
            
            # Do a few more local search steps after kick
            for _ in range(3):
                if time.time() > deadline:
                    break
                improved = False
                degs = [0] * len(current_family)
                for i in range(len(current_family)):
                    for j in range(len(current_family)):
                        if i != j and current_family[i] in comparables[current_family[j]]:
                            degs[i] += 1
                
                indices = list(range(len(current_family)))
                random.shuffle(indices)
                
                for i in indices:
                    A = current_family[i]
                    dA = degs[i]
                    best_gain = 0
                    best_B = -1
                    
                    for B in range(total_masks):
                        if B in current_fam_set:
                            continue
                        count_comp = 0
                        for x in current_family:
                            if x in comparables[B]:
                                count_comp += 1
                        if A in comparables[B]:
                            count_comp -= 1
                        gain = count_comp - dA
                        if gain > best_gain:
                            best_gain = gain
                            best_B = B
                    
                    if best_B != -1 and best_gain > 0:
                        current_fam_set.remove(A)
                        current_fam_set.add(best_B)
                        current_family[i] = best_B
                        current_score += best_gain // 2
                        if current_score > best_score_val:
                            best_score_val = current_score
                            best_family = current_family[:]
                        improved = True
                        break
                if not improved:
                    break
    
    # Convert to lists of integers
    return _convert(best_family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
