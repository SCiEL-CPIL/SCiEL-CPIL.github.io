import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def is_valid(n, colors):
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check Red K4
        for comb4 in combinations(range(n), 4):
            a, b, c, d = comb4
            if (colors[triple_to_idx[(a,b,c)]] == 0 and
                colors[triple_to_idx[(a,b,d)]] == 0 and
                colors[triple_to_idx[(a,c,d)]] == 0 and
                colors[triple_to_idx[(b,c,d)]] == 0):
                return False
        
        # Check Blue K5
        for comb5 in combinations(range(n), 5):
            a, b, c, d, e = comb5
            ts = [
                (a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e),
                (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)
            ]
            is_blue = True
            for t in ts:
                if colors[triple_to_idx[t]] != 1:
                    is_blue = False
                    break
            if is_blue:
                return False
        return True

    # Greedy sequential construction
    def greedy_construct(n, seed):
        rng = random.Random(seed)
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        colors = [None] * num_triples
        
        # For each triple, we'll track how many "dangerous" configurations it participates in
        # A red K4 forms when all 4 triples in a 4-set are red
        # A blue K5 forms when all 10 triples in a 5-set are blue
        
        # We process triples in lexicographic order
        for idx in range(num_triples):
            if time.time() > deadline - 5.0:
                return None, None
            
            a, b, c = triples[idx]
            
            # Count how many red K4s would form if we color this triple red
            red_k4_count = 0
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                # Check if the other 3 triples in {a,b,c,d} are already red
                i_abd = triple_to_idx[(a,b,d)]
                i_acd = triple_to_idx[(a,c,d)]
                i_bcd = triple_to_idx[(b,c,d)]
                
                # Only count if all three are already colored red
                if (colors[i_abd] == 0 and colors[i_acd] == 0 and colors[i_bcd] == 0):
                    red_k4_count += 1
            
            # Count how many blue K5s would form if we color this triple blue
            blue_k5_count = 0
            others = [v for v in range(n) if v != a and v != b and v != c]
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    d = others[i]
                    e = others[j]
                    # Check if the other 9 triples in {a,b,c,d,e} are already blue
                    # (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e)
                    # (b,c,d), (b,c,e), (b,d,e), (c,d,e)
                    all_blue = True
                    for (x, y, z) in [(a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e),
                                       (b,c,d), (b,c,e), (b,d,e), (c,d,e)]:
                        idx2 = triple_to_idx[(x,y,z)]
                        if colors[idx2] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        blue_k5_count += 1
            
            # Choose color that minimizes new violations
            # Weight blue K5 more heavily since K5 is larger
            red_cost = red_k4_count
            blue_cost = blue_k5_count * 2
            
            if red_cost < blue_cost:
                colors[idx] = 0
            elif blue_cost < red_cost:
                colors[idx] = 1
            else:
                # Tie: random choice
                colors[idx] = rng.randint(0, 1)
        
        return colors, triple_to_idx

    # Try greedy construction for various n
    best_result = None
    for n in range(34, 8, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple seeds for greedy construction
        for seed in range(20):
            if time.time() > deadline - 5.0:
                break
            
            colors, triple_to_idx = greedy_construct(n, seed)
            if colors is None:
                continue
            
            # Check if valid
            if is_valid(n, colors):
                if best_result is None or n > best_result['n']:
                    best_result = {'n': n, 'colors': colors}
                break  # Found valid coloring for this n, try smaller n
        
        # If we found a valid coloring for this n, we could stop
        # But let's try to find larger n first
        if best_result is not None and best_result['n'] == n:
            # We found a valid coloring for this n, but maybe we can do better
            # Continue trying smaller n? No, we want to maximize n
            # Actually, we're iterating from 34 down, so if we found one at n,
            # we should return it since we can't do better than 34
            if n >= 34:
                break
    
    # If greedy didn't work, fall back to local search
    if best_result is None:
        for n in range(34, 9, -1):
            if time.time() > deadline - 10.0:
                break
            
            triples = list(combinations(range(n), 3))
            num_triples = len(triples)
            triple_to_idx = {t: i for i, t in enumerate(triples)}
            
            max_restarts = 10
            for r in range(max_restarts):
                if time.time() > deadline - 5.0:
                    break
                
                colors = [random.randint(0, 1) for _ in range(num_triples)]
                
                if is_valid(n, colors):
                    best_result = {'n': n, 'colors': colors}
                    break
                
                # Local search with limited steps
                improved = True
                steps = 0
                max_steps = 500
                while improved and steps < max_steps and time.time() < deadline - 5.0:
                    improved = False
                    indices = list(range(num_triples))
                    random.shuffle(indices)
                    
                    for idx in indices:
                        if time.time() > deadline - 5.0:
                            break
                        
                        a, b, c = triples[idx]
                        old_color = colors[idx]
                        new_color = 1 - old_color
                        
                        delta = 0
                        
                        # Check K4s
                        for d in range(n):
                            if d == a or d == b or d == c:
                                continue
                            i2 = triple_to_idx[(a,b,d)]
                            i3 = triple_to_idx[(a,c,d)]
                            i4 = triple_to_idx[(b,c,d)]
                            
                            if colors[i2] == 0 and colors[i3] == 0 and colors[i4] == 0:
                                if old_color == 0:
                                    delta -= 1
                                else:
                                    delta += 1
                        
                        # Check K5s
                        others = [v for v in range(n) if v != a and v != b and v != c]
                        for i in range(len(others)):
                            for j in range(i+1, len(others)):
                                d = others[i]
                                e = others[j]
                                is_blue_other = (
                                    colors[triple_to_idx[(a,b,d)]] == 1 and
                                    colors[triple_to_idx[(a,b,e)]] == 1 and
                                    colors[triple_to_idx[(a,c,d)]] == 1 and
                                    colors[triple_to_idx[(a,c,e)]] == 1 and
                                    colors[triple_to_idx[(a,d,e)]] == 1 and
                                    colors[triple_to_idx[(b,c,d)]] == 1 and
                                    colors[triple_to_idx[(b,c,e)]] == 1 and
                                    colors[triple_to_idx[(b,d,e)]] == 1 and
                                    colors[triple_to_idx[(c,d,e)]] == 1
                                )
                                
                                if is_blue_other:
                                    if old_color == 1:
                                        delta -= 1
                                    else:
                                        delta += 1
                                    
                        if delta < 0:
                            colors[idx] = new_color
                            improved = True
                            
                        steps += 1
                        if steps > 1000:
                            break
                    
                    if is_valid(n, colors):
                        best_result = {'n': n, 'colors': colors}
                        break
            
            if best_result is not None:
                break
    
    # Fallback: algebraic construction for small n
    if best_result is None:
        n = 9
        best_result = {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    
    return best_result
