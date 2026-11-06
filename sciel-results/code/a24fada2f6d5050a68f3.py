import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 55.0
    best_set = []
    best_size = 0
    
    def run_greedy(order):
        S = []
        pair_sums = set()
        triple_sums = set()
        S_list = []
        
        for x in order:
            # Check pair sums
            if 2*x in pair_sums:
                continue
            
            valid = True
            new_pairs = [2*x]
            for y in S:
                s = x + y
                if s in pair_sums:
                    valid = False
                    break
                new_pairs.append(s)
            
            if not valid:
                continue

            # Check triple sums
            if 3*x in triple_sums:
                continue
            
            valid = True
            new_triples = [3*x]
            
            # Check 2*x+y
            for y in S:
                s = 2*x + y
                if s in triple_sums:
                    valid = False
                    break
                new_triples.append(s)
            
            if not valid:
                continue
            
            # Check x+y+z for all y <= z in S
            n = len(S)
            for y_idx in range(n):
                y = S[y_idx]
                for z_idx in range(y_idx, n):
                    z = S[z_idx]
                    s = x + y + z
                    if s in triple_sums:
                        valid = False
                        break
                    new_triples.append(s)
                if not valid:
                    break
            
            if not valid:
                continue

            # Accept
            S.append(x)
            pair_sums.update(new_pairs)
            triple_sums.update(new_triples)
        
        return S

    def local_search(S, max_iter=50):
        """Try to add more elements by looking for candidates that fit."""
        S = list(S)
        pair_sums = set()
        triple_sums = set()
        
        # Rebuild sets from S
        for i in range(len(S)):
            for j in range(i, len(S)):
                pair_sums.add(S[i] + S[j])
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    triple_sums.add(S[i] + S[j] + S[k])
        
        improved = True
        while improved and len(S) < 80:
            improved = False
            # Try adding each unused number
            used = set(S)
            for x in range(1, 4097):
                if x in used:
                    continue
                if 2*x in pair_sums:
                    continue
                if 3*x in triple_sums:
                    continue
                
                valid = True
                for y in S:
                    if (x + y) in pair_sums or (2*x + y) in triple_sums:
                        valid = False
                        break
                if not valid:
                    continue
                
                n = len(S)
                for i in range(n):
                    for j in range(i, n):
                        if (x + S[i] + S[j]) in triple_sums:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Try to add x
                S.append(x)
                for y in S:
                    pair_sums.add(x + y)
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        triple_sums.add(S[i] + S[j] + x)
                improved = True
                break
        
        return S

    # Initial deterministic runs
    for order in [range(1, 4097), range(4096, 0, -1), 
                  list(range(1, 4097, 2)), list(range(2, 4097, 2))]:
        current_set = run_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Try local search on best set
    if len(best_set) > 0:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved

    # Randomized runs with different strategies
    while time.time() - start_time < time_limit:
        strategy = random.random()
        
        if strategy < 0.3:
            # Pure random
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
        elif strategy < 0.5:
            # Random with bias toward smaller numbers
            candidates = []
            for x in range(1, 4097):
                weight = 1.0 / math.sqrt(x)
                if random.random() < weight / 100:
                    candidates.append(x)
            random.shuffle(candidates)
            if len(candidates) < 100:
                candidates = list(range(1, 4097))
                random.shuffle(candidates)
        elif strategy < 0.7:
            # Random with bias toward larger numbers
            candidates = []
            for x in range(1, 4097):
                weight = math.sqrt(x) / 64
                if random.random() < weight / 100:
                    candidates.append(x)
            random.shuffle(candidates)
            if len(candidates) < 100:
                candidates = list(range(1, 4097))
                random.shuffle(candidates)
        else:
            # Mix of random and sorted
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
            # Insert some in sorted order
            sorted_part = sorted(candidates[:400])
            random_part = candidates[400:]
            random.shuffle(random_part)
            candidates = sorted_part + random_part
        
        current_set = run_greedy(candidates)
        
        # Try local search if promising
        if len(current_set) >= best_size - 2:
            improved = local_search(current_set)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        elif len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return best_set
