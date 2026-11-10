import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    
    # All possible triples
    all_triples = list(combinations(points, 3))
    
    # Precompute masks for all triples
    triple_masks = {}
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks[t] = m
    
    # Precompute pairs for each triple
    triple_pairs = {}
    for t in all_triples:
        pairs = set()
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pairs.add(key)
        triple_pairs[t] = pairs
    
    def get_alpha(triples):
        """Compute independence number via branch-and-bound hitting set."""
        if not triples:
            return n
        
        t_masks = [triple_masks[t] for t in triples]
        num_triples = len(t_masks)
        
        # Upper bound on hitting set size
        all_pts = 0
        for tm in t_masks:
            all_pts |= tm
        upper_bound = bin(all_pts).count('1')
        best = [upper_bound]
        
        # Precompute for each point, bitmask of triples it hits
        point_hit_mask = [0] * n
        for idx, tm in enumerate(t_masks):
            for p in range(n):
                if tm & (1 << p):
                    point_hit_mask[p] |= (1 << idx)
        
        # Sort triples by size (all size 3) but by frequency to improve pruning
        # Actually, let's just use the order as-is but pick the most constrained triple first
        
        def solve(remaining_mask, cur_size):
            if cur_size >= best[0]:
                return
            if remaining_mask == 0:
                best[0] = cur_size
                return
            
            # Pick triple with fewest uncovered points (most constrained)
            first_bit = remaining_mask & (-remaining_mask)
            first_idx = first_bit.bit_length() - 1
            first_triple = t_masks[first_idx]
            
            # Branch on each point in this triple
            for p in range(n):
                if not (first_triple & (1 << p)):
                    continue
                new_remaining = remaining_mask & ~point_hit_mask[p]
                solve(new_remaining, cur_size + 1)
                if best[0] <= cur_size + 1:
                    return
        
        full_mask = (1 << num_triples) - 1
        solve(full_mask, 0)
        return n - best[0]
    
    def is_linear(triples):
        seen_pairs = set()
        for t in triples:
            for key in triple_pairs[t]:
                if key in seen_pairs:
                    return False
                seen_pairs.add(key)
        return True
    
    def greedy_construct():
        """Greedy construction prioritizing balanced point coverage."""
        triples = []
        used_pairs = set()
        point_count = [0] * n
        
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            valid = True
            for key in triple_pairs[t]:
                if key in used_pairs:
                    valid = False
                    break
            if valid:
                triples.append(t)
                used_pairs.update(triple_pairs[t])
                for p in t:
                    point_count[p] += 1
        
        return triples
    
    def local_search(triples, deadline):
        """Simulated annealing with 1-move and 2-move swaps."""
        if not triples:
            return triples, get_alpha(triples)
        
        current_alpha = get_alpha(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        
        current = triples[:]
        cur_alpha = current_alpha
        
        temperature = 1.0
        cooling = 0.99
        
        max_steps = 15000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if step % 20 == 0:
                temperature *= cooling
            
            # Choose move type: 1-swap (80%) or 2-swap (20%)
            if random.random() < 0.8:
                # 1-swap: remove one, add one
                if len(current) < 2:
                    continue
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                
                # Find compatible candidates
                new_used_pairs = set()
                for t in current:
                    if t == removed:
                        continue
                    new_used_pairs.update(triple_pairs[t])
                
                valid_cands = []
                for t in all_triples:
                    if t == removed or t in current:
                        continue
                    if not (triple_pairs[t] & new_used_pairs):
                        valid_cands.append(t)
                
                if not valid_cands:
                    continue
                
                added = random.choice(valid_cands)
                new_triples = current[:idx] + current[idx+1:] + [added]
            else:
                # 2-swap: remove two, add two
                if len(current) < 4:
                    continue
                idx1, idx2 = random.sample(range(len(current)), 2)
                removed1 = current[idx1]
                removed2 = current[idx2]
                
                new_used_pairs = set()
                for t in current:
                    if t == removed1 or t == removed2:
                        continue
                    new_used_pairs.update(triple_pairs[t])
                
                valid_cands = []
                for t in all_triples:
                    if t == removed1 or t == removed2 or t in current:
                        continue
                    if not (triple_pairs[t] & new_used_pairs):
                        valid_cands.append(t)
                
                if len(valid_cands) < 2:
                    continue
                
                added1, added2 = random.sample(valid_cands, 2)
                # Check that added1 and added2 don't conflict with each other
                if triple_pairs[added1] & triple_pairs[added2]:
                    continue
                
                new_triples = [t for t in current if t != removed1 and t != removed2]
                new_triples.extend([added1, added2])
            
            new_alpha = get_alpha(new_triples)
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_alpha - cur_alpha
            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
            else:
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current = new_triples
                    cur_alpha = new_alpha
        
        # Final polish: try to add more triples
        while time.time() < deadline - 2:
            used_pairs_final = set()
            for t in best_triples:
                used_pairs_final.update(triple_pairs[t])
            
            added_any = False
            for t in all_triples:
                if t in best_triples:
                    continue
                if not (triple_pairs[t] & used_pairs_final):
                    test_triples = best_triples + [t]
                    test_alpha = get_alpha(test_triples)
                    if test_alpha <= best_alpha:
                        best_triples = test_triples
                        best_alpha = test_alpha
                        used_pairs_final.update(triple_pairs[t])
                        added_any = True
                        break
            
            if not added_any:
                break
        
        return best_triples, best_alpha
    
    # Main: multiple random restarts
    best_triples = []
    best_alpha = n
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        triples = greedy_construct()
        if not triples:
            continue
        
        triples, alpha = local_search(triples, deadline)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        if best_alpha <= 4:
            break
    
    # Use remaining time for more restarts
    while time.time() < deadline - 3:
        triples = greedy_construct()
        if not triples:
            continue
        triples, alpha = local_search(triples, deadline)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
    
    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    used_pairs = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print("INVALID: Shared pair", t)
                    break
                used_pairs.add(key)
    else:
        print("Valid linear system")
