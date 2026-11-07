import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit

    def time_left():
        return deadline - time.time()

    # Precompute comparability bitsets
    # neighbors_bitset[i] has bit j set if i and j are comparable (i != j)
    neighbors_bitset = [0] * total_masks
    for i in range(total_masks):
        mask = 0
        # Subsets
        sub = i
        while True:
            if sub != i:
                mask |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & i
        
        # Supersets
        comp_bits = (~i) & (total_masks - 1)
        c = comp_bits
        while True:
            s = i | c
            if s != i:
                mask |= (1 << s)
            if c == 0:
                break
            c = (c - 1) & comp_bits
        neighbors_bitset[i] = mask

    def popcount(x):
        return bin(x).count("1")

    def calc_score(family_list):
        """Calculate number of comparable pairs."""
        score = 0
        L = len(family_list)
        for i in range(L):
            for j in range(i + 1, L):
                if (neighbors_bitset[family_list[i]] >> family_list[j]) & 1:
                    score += 1
        return score

    def greedy_construct():
        """Build family greedily."""
        family = [0]
        used = {0}
        cur = 0
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)
            used.add(cur)
        
        while len(family) < m:
            best_gain = -1
            best_mask = None
            for s in range(total_masks):
                if s in used:
                    continue
                gain = 0
                for f in family:
                    if (neighbors_bitset[f] >> s) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
        return family

    def simulated_annealing(start_family):
        """Run SA on the start family."""
        family = list(start_family)
        family_set = set(family)
        family_bitset = 0
        for f in family:
            family_bitset |= (1 << f)
        
        current_score = calc_score(family)
        best_score = current_score
        best_family = list(family)
        
        T = 10.0
        T_end = 0.01
        cooling_rate = 0.9999
        
        while time_left() > 5:
            # Pick a random element to remove
            a = random.choice(family)
            
            # Pick a random mask to add
            b = random.randint(0, total_masks - 1)
            if b in family_set:
                continue
            
            # Calculate delta using bitsets
            # family_without_a_bitset
            fam_wo_a = family_bitset & ~(1 << a)
            
            # Loss: number of neighbors of a in fam_wo_a
            # neighbors_bitset[a] & fam_wo_a
            loss = popcount(neighbors_bitset[a] & fam_wo_a)
            
            # Gain: number of neighbors of b in fam_wo_a
            gain = popcount(neighbors_bitset[b] & fam_wo_a)
            
            delta = gain - loss
            
            accept = False
            if delta > 0:
                accept = True
            else:
                if delta < -100:
                    prob = 0.0
                else:
                    prob = math.exp(delta / T)
                if random.random() < prob:
                    accept = True
            
            if accept:
                # Perform swap
                family_set.remove(a)
                family_set.add(b)
                # Update family list
                idx = family.index(a)
                family[idx] = b
                # Update bitset
                family_bitset = (family_bitset & ~(1 << a)) | (1 << b)
                
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
                
                T *= cooling_rate
                if T < T_end:
                    T = T_end
            else:
                T *= cooling_rate
                if T < T_end:
                    T = T_end

        return best_family, best_score

    # Generate a good initial family
    initial_family = greedy_construct()
    
    # Run SA
    best_family, best_score = simulated_annealing(initial_family)
    
    # Convert to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
