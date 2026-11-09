import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    
    # Precompute comparability matrix for all 4096 subsets
    # comp[a][b] = True if a is comparable with b (one is strict subset of other)
    # Since we need to check many times, precompute as a 2D array
    num_masks = 1 << n
    
    # For each mask, precompute which other masks are comparable
    # Use a bitset approach: for each mask s, store a set of comparable masks
    # Actually, let's just precompute a 2D boolean array
    # 4096 x 4096 = 16M entries, each 1 byte = 16MB, acceptable
    
    # More efficient: for each mask, precompute list of comparable masks
    # But that's variable length. Let's use the direct check but optimize it.
    
    # Actually, the bottleneck is the inner loop over 39 family members.
    # Let's precompute for each mask s, a sorted list of all masks comparable to s.
    # Then for a given family, we can quickly count how many family members are comparable to s.
    
    # Better approach: maintain a frequency array and use inclusion-exclusion?
    # No, let's just optimize the current approach with numpy or careful Python.
    
    # Let's try a different angle: precompute for each mask, its "comparable set" as a bitset
    # Then count = popcount(comparable_set[s] & family_bitset)
    
    # Precompute comparable bitsets
    # comp_bits[s] is a Python int where bit j is set if mask j is comparable to mask s
    comp_bits = [0] * num_masks
    
    for s in range(num_masks):
        cb = 0
        for t in range(num_masks):
            if s == t:
                continue
            if (s & t) == s or (s & t) == t:
                cb |= (1 << t)
        comp_bits[s] = cb
    
    def count_comparables(family):
        count = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                fj = family[j]
                if (fi & fj) == fi or (fi & fj) == fj:
                    count += 1
        return count
    
    def greedy_init():
        family = []
        family_set = set()
        
        family.append(0)
        family_set.add(0)
        
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            
            for s in range(1 << n):
                if s in family_set:
                    continue
                
                gain = 0
                for f in family:
                    if (f & s) == f or (f & s) == s:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
        
        return family
    
    family = greedy_init()
    current_score = count_comparables(family)
    
    # SA with optimized delta using comp_bits
    T = 10.0
    T_min = 0.01
    alpha = 0.9999  # Very slow cooling
    max_iter_per_restart = 200000
    num_restarts = 3
    
    best_family = family[:]
    best_score = current_score
    
    # Maintain a set for O(1) membership check
    family_set = set(family)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        T = 10.0
        current_score = count_comparables(family)
        
        for it in range(max_iter_per_restart):
            if time.time() > deadline:
                break
            
            i = random.randint(0, m - 1)
            old_s = family[i]
            
            # Smart candidate generation
            if random.random() < 0.7:
                ref_idx = random.randint(0, m - 1)
                if ref_idx == i:
                    ref_idx = (ref_idx + 1) % m
                ref_s = family[ref_idx]
                
                if random.random() < 0.5:
                    new_s = ref_s
                    num_add = random.randint(1, 3)
                    for _ in range(num_add):
                        bit = random.randint(0, n - 1)
                        new_s |= (1 << bit)
                else:
                    new_s = ref_s
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        bit = random.randint(0, n - 1)
                        new_s &= ~(1 << bit)
            else:
                new_s = random.randint(0, (1 << n) - 1)
            
            if new_s == old_s:
                continue
            
            if new_s in family_set:
                continue
            
            # Fast delta using precomputed comp_bits
            # Count how many family members (excluding i) are comparable to new_s and old_s
            # Use the bitset trick: build a bitset of family members, then popcount intersection
            
            # Build family bitset excluding position i
            # Actually, let's just iterate - it's only 39 elements
            
            count_new = 0
            count_old = 0
            
            for j, f in enumerate(family):
                if j == i:
                    continue
                if (f & new_s) == f or (f & new_s) == new_s:
                    count_new += 1
                if (f & old_s) == f or (f & old_s) == old_s:
                    count_old += 1
            
            delta = count_new - count_old
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                family[i] = new_s
                family_set.remove(old_s)
                family_set.add(new_s)
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            
            T *= alpha
            if T < T_min:
                T = T_min
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(m):
                old_s = family[i]
                best_local_gain = 0
                best_local_s = -1
                
                # Try systematic candidates: all supersets and subsets of old_s
                # and some random ones
                candidates = set()
                
                # All immediate supersets
                for bit in range(n):
                    if not (old_s & (1 << bit)):
                        candidates.add(old_s | (1 << bit))
                
                # All immediate subsets
                for bit in range(n):
                    if old_s & (1 << bit):
                        candidates.add(old_s & ~(1 << bit))
                
                # Some random candidates
                for _ in range(10):
                    candidates.add(random.randint(0, (1 << n) - 1))
                
                for new_s in candidates:
                    if new_s == old_s:
                        continue
                    if new_s in family_set:
                        continue
                    
                    count_new = 0
                    for j, f in enumerate(family):
                        if j == i:
                            continue
                        if (f & new_s) == f or (f & new_s) == new_s:
                            count_new += 1
                    
                    count_old = 0
                    for j, f in enumerate(family):
                        if j == i:
                            continue
                        if (f & old_s) == f or (f & old_s) == old_s:
                            count_old += 1
                    
                    local_gain = count_new - count_old
                    if local_gain > best_local_gain:
                        best_local_gain = local_gain
                        best_local_s = new_s
                
                if best_local_s != -1 and best_local_gain > 0:
                    family[i] = best_local_s
                    family_set.remove(old_s)
                    family_set.add(best_local_s)
                    current_score += best_local_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
    
    family = best_family
    
    result = []
    for s in family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
        
    return result
