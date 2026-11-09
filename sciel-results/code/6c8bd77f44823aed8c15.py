import time
import random
import sys

def construct():
    n = 12
    m = 40
    deadline = time.time() + 50.0  # Safety margin for 90s limit
    
    # 1. Generate all masks
    all_masks = list(range(1 << n))
    
    # Helper to check if a is subset of b (a <= b)
    def is_subset(a, b):
        return (a & b) == a

    # Helper to check if a and b are comparable
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # 2. Construct Initial Family: 3 Maximal Chains + 1 Set
    # We need 3 disjoint chains of length 13.
    # A maximal chain in 2^12 has 13 elements.
    # We can generate chains by starting from a set and adding/removing bits.
    
    def get_maximal_chain(start_mask):
        # This is a simple heuristic to get a chain.
        # A better way: 
        # Chain: 0, 1, 3, 7, 15... (adding bits in order)
        # But we need disjoint chains.
        pass

    # Simple deterministic initialization of 3 chains
    # Chain 1: Add bits 0..11 in order
    c1 = []
    mask = 0
    c1.append(mask)
    for i in range(n):
        mask |= (1 << i)
        c1.append(mask)
        
    # Chain 2: Add bits 1..11, then 0? No, must be disjoint.
    # Let's just pick random maximal chains and check for disjointness.
    
    def generate_random_chain():
        # Start with a random set, extend up and down to make it maximal
        # Or simpler: pick a random permutation of bits, build chain
        bits = list(range(n))
        random.shuffle(bits)
        chain = [0]
        mask = 0
        for b in bits:
            mask |= (1 << b)
            chain.append(mask)
        return chain

    family_masks = []
    used = set()
    
    # Generate 3 disjoint chains
    attempts = 0
    while len(family_masks) < 39 and attempts < 1000:
        attempts += 1
        c = generate_random_chain()
        # Check if disjoint from current family
        if not any(x in used for x in c):
            family_masks.extend(c)
            used.update(c)
            
    # If we failed to get 3 chains (unlikely), fill with random sets
    if len(family_masks) < 39:
        for mask in all_masks:
            if len(family_masks) >= 39:
                break
            if mask not in used:
                family_masks.append(mask)
                used.add(mask)
                
    # Add 1 more set
    for mask in all_masks:
        if mask not in used:
            family_masks.append(mask)
            used.add(mask)
            break
            
    if len(family_masks) < m:
        # Fallback
        for mask in all_masks:
            if len(family_masks) >= m:
                break
            if mask not in used:
                family_masks.append(mask)
                used.add(mask)

    # 3. Simulated Annealing / Local Search
    # State: list of 40 masks
    # Score: count of comparable pairs
    
    def calc_score(fam):
        score = 0
        for i in range(len(fam)):
            mi = fam[i]
            for j in range(i+1, len(fam)):
                mj = fam[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    score += 1
        return score

    current_score = calc_score(family_masks)
    best_family = family_masks[:]
    best_score = current_score
    
    temp = 10.0
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline - 5:
        # Pick a random index to replace
        i = random.randint(0, m - 1)
        old_mask = family_masks[i]
        
        # Pick a random new mask not in family
        new_mask = random.choice(all_masks)
        if new_mask in used:
            # Try a few times to find a new one
            for _ in range(10):
                new_mask = random.choice(all_masks)
                if new_mask not in used:
                    break
            if new_mask in used:
                continue
                
        # Calculate delta
        # Remove old_mask: subtract pairs with all other 39
        delta = 0
        for j in range(m):
            if j == i:
                continue
            mj = family_masks[j]
            # Old contribution
            if (old_mask & mj) == old_mask or (old_mask & mj) == mj:
                delta -= 1
            # New contribution
            if (new_mask & mj) == new_mask or (new_mask & mj) == mj:
                delta += 1
                
        if delta > 0 or random.random() < pow(2.71828, delta / temp):
            # Accept
            family_masks[i] = new_mask
            used.remove(old_mask)
            used.add(new_mask)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = family_masks[:]
                
        temp *= cooling
        
        if temp < 0.1:
            temp = 1.0 # Restart if stuck? Or just keep going.
            # Better: if no improvement for a long time, restart temp
            
    # Final Polish: Greedy Local Search
    # Try to improve by swapping any pair? No, just single replacement.
    # Run a few passes of pure greedy (accept only if delta > 0)
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for i in range(m):
            if time.time() > deadline - 1:
                break
            old_mask = family_masks[i]
            best_new_mask = None
            best_delta = 0
            
            # Sample a subset of candidates to keep it fast? 
            # Or iterate all? 4096 * 40 = 163k checks per pass.
            # 163k checks is ~0.05s. So we can do full scan.
            
            for new_mask in all_masks:
                if new_mask in used:
                    continue
                    
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    mj = family_masks[j]
                    if (old_mask & mj) == old_mask or (old_mask & mj) == mj:
                        delta -= 1
                    if (new_mask & mj) == new_mask or (new_mask & mj) == mj:
                        delta += 1
                        
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
                    
            if best_new_mask is not None and best_delta > 0:
                family_masks[i] = best_new_mask
                used.remove(old_mask)
                used.add(best_new_mask)
                current_score += best_delta
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family_masks[:]
                    
    # Convert best_family to lists
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
