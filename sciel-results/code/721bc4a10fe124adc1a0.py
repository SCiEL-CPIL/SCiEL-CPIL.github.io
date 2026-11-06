import random
import time

def construct():
    # Precomputation
    # triples_by_element[x] = list of (d, a, m, b) where x is in {a, m, b}
    triples_by_element = {x: [] for x in range(1, 513)}
    
    # Generate all triples (a, a+d, a+2d) with a, a+2d in [1, 512]
    for d in range(1, 257): # max d such that 2d < 512
        for a in range(1, 513 - 2*d + 1):
            m = a + d
            b = a + 2*d
            triple = (d, a, m, b)
            triples_by_element[a].append(triple)
            triples_by_element[m].append(triple)
            triples_by_element[b].append(triple)
    
    def get_dense_seeds(n=32):
        counts = []
        for x in range(1, 513):
            c_mid = min(x - 1, 512 - x)
            c_first = (512 - x) // 2
            c_last = (x - 1) // 2
            total = c_mid + c_first + c_last
            counts.append((total, x))
        counts.sort(reverse=True)
        return [x for _, x in counts[:n]]
    
    # Initial Set
    current_set = set(get_dense_seeds(32))
    
    # Initialize count_d
    count_d = [0] * 513
    # Calculate initial score
    # For each d, count how many triples are fully in current_set
    # We can do this by iterating through all triples and checking membership
    # But it's faster to use the triples_by_element? No, just iterate all triples.
    # There are ~ sum_{d=1}^{256} (512 - 2d) triples. ~ 256 * 256 = 65k triples.
    # 65k checks is fast.
    
    all_triples = []
    for d in range(1, 257):
        for a in range(1, 513 - 2*d + 1):
            m = a + d
            b = a + 2*d
            all_triples.append((d, a, m, b))
            
    for d, a, m, b in all_triples:
        if a in current_set and m in current_set and b in current_set:
            count_d[d] += 1
            
    current_score = sum(1 for d in range(1, 513) if count_d[d] > 0)
    
    best_set = set(current_set)
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110
    
    temp = 5.0
    cooling = 0.9999
    random.seed(42)
    
    all_nums = list(range(1, 513))
    
    while time.time() - start_time < time_limit:
        # Pick random out
        out = random.choice(list(current_set))
        
        # Pick random in
        # To avoid building list of candidates every time (slow), 
        # we can pick a random x from 1..512 and retry if in current_set.
        # Or maintain a list of available numbers? 
        # Since set size is 32 out of 512, probability of collision is low (~6%).
        # Let's just pick random x.
        while True:
            in_elem = random.randint(1, 512)
            if in_elem not in current_set:
                break
                
        # Save state for potential undo
        # We don't need to save the whole set, just the changes to count_d and score
        # But we need to know which count_d changed.
        # Let's just record the deltas.
        
        prev_score = current_score
        
        # Remove 'out'
        # For each triple containing 'out', if other two in set, decrement count_d
        for d, a, m, b in triples_by_element[out]:
            # Check if other two are in current_set
            # 'out' is one of a, m, b
            if out == a:
                if m in current_set and b in current_set:
                    count_d[d] -= 1
                    if count_d[d] == 0:
                        current_score -= 1
            elif out == m:
                if a in current_set and b in current_set:
                    count_d[d] -= 1
                    if count_d[d] == 0:
                        current_score -= 1
            else: # out == b
                if a in current_set and m in current_set:
                    count_d[d] -= 1
                    if count_d[d] == 0:
                        current_score -= 1
                        
        current_set.discard(out)
        
        # Add 'in_elem'
        current_set.add(in_elem)
        
        for d, a, m, b in triples_by_element[in_elem]:
            if in_elem == a:
                if m in current_set and b in current_set:
                    count_d[d] += 1
                    if count_d[d] == 1: # was 0
                        current_score += 1
            elif in_elem == m:
                if a in current_set and b in current_set:
                    count_d[d] += 1
                    if count_d[d] == 1:
                        current_score += 1
            else: # in_elem == b
                if a in current_set and m in current_set:
                    count_d[d] += 1
                    if count_d[d] == 1:
                        current_score += 1
                        
        delta = current_score - prev_score
        
        # Accept/Reject
        if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
            # Accept
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            # Reject: Undo
            # Remove 'in_elem'
            for d, a, m, b in triples_by_element[in_elem]:
                if in_elem == a:
                    if m in current_set and b in current_set:
                        count_d[d] -= 1
                        if count_d[d] == 0:
                            current_score -= 1
                elif in_elem == m:
                    if a in current_set and b in current_set:
                        count_d[d] -= 1
                        if count_d[d] == 0:
                            current_score -= 1
                else:
                    if a in current_set and m in current_set:
                        count_d[d] -= 1
                        if count_d[d] == 0:
                            current_score -= 1
                            
            current_set.discard(in_elem)
            
            # Add 'out'
            current_set.add(out)
            
            for d, a, m, b in triples_by_element[out]:
                if out == a:
                    if m in current_set and b in current_set:
                        count_d[d] += 1
                        if count_d[d] == 1:
                            current_score += 1
                elif out == m:
                    if a in current_set and b in current_set:
                        count_d[d] += 1
                        if count_d[d] == 1:
                            current_score += 1
                else:
                    if a in current_set and m in current_set:
                        count_d[d] += 1
                        if count_d[d] == 1:
                            current_score += 1
                            
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    return sorted(best_set)
